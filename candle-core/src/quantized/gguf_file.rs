//! Support for the [GGUF file format](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md).
//!
//! Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

use super::{GgmlDType, QTensor};
use crate::{Context, Device, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;

pub const DEFAULT_ALIGNMENT: u64 = 32;

// Caps mirror ggml-org/llama.cpp#19856 (GGUF_MAX_STRING_LENGTH,
// GGUF_MAX_ARRAY_ELEMENTS) and GGML_MAX_DIMS from ggml.h. The depth cap
// bounds recursion through nested `Value::Array` so a crafted file cannot
// overflow the stack with a chain of arrays-of-arrays.
const GGUF_MAX_STRING_LENGTH: u64 = 1 << 30;
const GGUF_MAX_ARRAY_ELEMENTS: u64 = 1 << 30;
const GGUF_MAX_TENSOR_DIMS: u32 = 4;
const GGUF_MAX_VALUE_DEPTH: usize = 64;

/// Allocation and header bounds used while parsing a GGUF directory.
///
/// [`Content::read`] retains the format-wide defaults. Callers loading a more
/// narrowly specified artifact can use [`Content::read_with_limits`] to reject
/// oversized declarations before allocating their strings, arrays, or maps.
#[derive(Debug, Clone, Copy)]
pub struct ContentReadLimits {
    pub max_tensor_count: u64,
    pub max_metadata_count: u64,
    pub max_string_length: u64,
    pub max_array_elements: u64,
    pub max_header_bytes: u64,
}

impl Default for ContentReadLimits {
    fn default() -> Self {
        Self {
            max_tensor_count: GGUF_MAX_ARRAY_ELEMENTS,
            max_metadata_count: GGUF_MAX_ARRAY_ELEMENTS,
            max_string_length: GGUF_MAX_STRING_LENGTH,
            max_array_elements: GGUF_MAX_ARRAY_ELEMENTS,
            max_header_bytes: u64::MAX,
        }
    }
}

// `file_size` is the byte length captured once up front, so this avoids
// seeking to the end and back on every length-prefixed read.
fn remaining_bytes<R: std::io::Seek>(reader: &mut R, file_size: u64) -> Result<u64> {
    let cur = reader.stream_position()?;
    Ok(file_size.saturating_sub(cur))
}

fn read_length<R: std::io::Read>(reader: &mut R, magic: &VersionedMagic) -> Result<u64> {
    match magic {
        VersionedMagic::GgufV1 => Ok(reader.read_u32::<LittleEndian>()? as u64),
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => Ok(reader.read_u64::<LittleEndian>()?),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Gguf,
}

impl TryFrom<u32> for Magic {
    type Error = crate::Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x46554747 | 0x47475546 => Self::Gguf,
            _ => crate::bail!("unknown magic 0x{value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Gguf, 1) => Self::GgufV1,
            (Magic::Gguf, 2) => Self::GgufV2,
            (Magic::Gguf, 3) => Self::GgufV3,
            _ => crate::bail!("gguf: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }

    fn length_prefix_size(&self) -> u64 {
        match self {
            Self::GgufV1 => 4,
            Self::GgufV2 | Self::GgufV3 => 8,
        }
    }
}

#[derive(Debug)]
pub struct TensorInfo {
    pub ggml_dtype: GgmlDType,
    pub shape: crate::Shape,
    pub offset: u64,
}

impl TensorInfo {
    pub fn read<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        tensor_data_offset: u64,
        device: &Device,
    ) -> Result<QTensor> {
        let tensor_elems = self.shape.elem_count();
        let block_size = self.ggml_dtype.block_size();
        if !tensor_elems.is_multiple_of(block_size) {
            crate::bail!(
            "the number of elements {tensor_elems} is not divisible by the block size {block_size}"
        )
        }
        let size_in_bytes = tensor_elems / block_size * self.ggml_dtype.type_size();
        let tensor_start = tensor_data_offset.saturating_add(self.offset);
        let file_size = reader.seek(std::io::SeekFrom::End(0))?;
        let remaining = file_size.saturating_sub(tensor_start);
        if size_in_bytes as u64 > remaining {
            crate::bail!(
                "tensor needs {size_in_bytes} bytes at offset {tensor_start}, only {remaining} remaining in file"
            )
        }
        let mut raw_data = vec![0u8; size_in_bytes];
        reader.seek(std::io::SeekFrom::Start(tensor_start))?;
        reader.read_exact(&mut raw_data)?;
        super::ggml_file::qtensor_from_ggml(
            self.ggml_dtype,
            &raw_data,
            self.shape.dims().to_vec(),
            device,
        )
    }
}

#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

fn read_string<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    magic: &VersionedMagic,
    file_size: u64,
    limits: &ContentReadLimits,
) -> Result<String> {
    let len = read_length(reader, magic)?;
    if len > limits.max_string_length {
        crate::bail!(
            "gguf: string length {len} exceeds max {}",
            limits.max_string_length
        )
    }
    let remaining = remaining_bytes(reader, file_size)?;
    if len > remaining {
        crate::bail!("gguf: string length {len} exceeds remaining file bytes {remaining}")
    }
    let mut v = vec![0u8; len as usize];
    reader.read_exact(&mut v)?;
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while let Some(0) = v.last() {
        v.pop();
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    Ok(String::from_utf8_lossy(&v).into_owned())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    // The value is a 8-bit unsigned integer.
    U8,
    // The value is a 8-bit signed integer.
    I8,
    // The value is a 16-bit unsigned little-endian integer.
    U16,
    // The value is a 16-bit signed little-endian integer.
    I16,
    // The value is a 32-bit unsigned little-endian integer.
    U32,
    // The value is a 32-bit signed little-endian integer.
    I32,
    // The value is a 64-bit unsigned little-endian integer.
    U64,
    // The value is a 64-bit signed little-endian integer.
    I64,
    // The value is a 32-bit IEEE754 floating point number.
    F32,
    // The value is a 64-bit IEEE754 floating point number.
    F64,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String,
    // The value is an array of other values, with the length and type prepended.
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array,
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
        }
    }

    pub fn to_u8(&self) -> Result<u8> {
        match self {
            Self::U8(v) => Ok(*v),
            v => crate::bail!("not a u8 {v:?}"),
        }
    }

    pub fn to_i8(&self) -> Result<i8> {
        match self {
            Self::I8(v) => Ok(*v),
            v => crate::bail!("not a i8 {v:?}"),
        }
    }

    pub fn to_u16(&self) -> Result<u16> {
        match self {
            Self::U16(v) => Ok(*v),
            v => crate::bail!("not a u16 {v:?}"),
        }
    }

    pub fn to_i16(&self) -> Result<i16> {
        match self {
            Self::I16(v) => Ok(*v),
            v => crate::bail!("not a i16 {v:?}"),
        }
    }

    pub fn to_u32(&self) -> Result<u32> {
        match self {
            Self::U32(v) => Ok(*v),
            v => crate::bail!("not a u32 {v:?}"),
        }
    }

    pub fn to_i32(&self) -> Result<i32> {
        match self {
            Self::I32(v) => Ok(*v),
            v => crate::bail!("not a i32 {v:?}"),
        }
    }

    /// This will also automatically upcast any integral types which will not truncate.
    pub fn to_u64(&self) -> Result<u64> {
        match self {
            Self::U64(v) => Ok(*v),
            // Autoupcast cases here
            Self::U8(v) => Ok(*v as u64),
            Self::U16(v) => Ok(*v as u64),
            Self::U32(v) => Ok(*v as u64),
            Self::Bool(v) => Ok(*v as u64),
            v => crate::bail!("not a u64 or upcastable to u64 {v:?}"),
        }
    }

    pub fn to_i64(&self) -> Result<i64> {
        match self {
            Self::I64(v) => Ok(*v),
            v => crate::bail!("not a i64 {v:?}"),
        }
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            v => crate::bail!("not a f32 {v:?}"),
        }
    }

    pub fn to_f64(&self) -> Result<f64> {
        match self {
            Self::F64(v) => Ok(*v),
            v => crate::bail!("not a f64 {v:?}"),
        }
    }

    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            v => crate::bail!("not a bool {v:?}"),
        }
    }

    pub fn to_vec(&self) -> Result<&Vec<Value>> {
        match self {
            Self::Array(v) => Ok(v),
            v => crate::bail!("not a vec {v:?}"),
        }
    }

    pub fn to_string(&self) -> Result<&String> {
        match self {
            Self::String(v) => Ok(v),
            v => crate::bail!("not a string {v:?}"),
        }
    }

    fn read<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
        depth: usize,
        file_size: u64,
        limits: &ContentReadLimits,
    ) -> Result<Self> {
        if depth > GGUF_MAX_VALUE_DEPTH {
            crate::bail!("gguf: value nesting depth exceeds max {GGUF_MAX_VALUE_DEPTH}")
        }
        let v = match value_type {
            ValueType::U8 => Self::U8(reader.read_u8()?),
            ValueType::I8 => Self::I8(reader.read_i8()?),
            ValueType::U16 => Self::U16(reader.read_u16::<LittleEndian>()?),
            ValueType::I16 => Self::I16(reader.read_i16::<LittleEndian>()?),
            ValueType::U32 => Self::U32(reader.read_u32::<LittleEndian>()?),
            ValueType::I32 => Self::I32(reader.read_i32::<LittleEndian>()?),
            ValueType::U64 => Self::U64(reader.read_u64::<LittleEndian>()?),
            ValueType::I64 => Self::I64(reader.read_i64::<LittleEndian>()?),
            ValueType::F32 => Self::F32(reader.read_f32::<LittleEndian>()?),
            ValueType::F64 => Self::F64(reader.read_f64::<LittleEndian>()?),
            ValueType::Bool => match reader.read_u8()? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => crate::bail!("unexpected bool value {b}"),
            },
            ValueType::String => Self::String(read_string(reader, magic, file_size, limits)?),
            ValueType::Array => {
                let value_type = reader.read_u32::<LittleEndian>()?;
                let value_type = ValueType::from_u32(value_type)?;
                let len = read_length(reader, magic)?;
                if len > limits.max_array_elements {
                    crate::bail!(
                        "gguf: array length {len} exceeds max {}",
                        limits.max_array_elements
                    )
                }
                let needed = len.saturating_mul(value_type.min_disk_size(magic));
                let remaining = remaining_bytes(reader, file_size)?;
                if needed > remaining {
                    crate::bail!(
                        "gguf: array of {len} elements needs at least {needed} bytes, only {remaining} remaining"
                    )
                }
                let mut vs = Vec::new();
                for _ in 0..len {
                    vs.push(Value::read(
                        reader,
                        value_type,
                        magic,
                        depth + 1,
                        file_size,
                        limits,
                    )?)
                }
                Self::Array(vs)
            }
        };
        Ok(v)
    }

    fn write<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        match self {
            &Self::U8(v) => w.write_u8(v)?,
            &Self::I8(v) => w.write_i8(v)?,
            &Self::U16(v) => w.write_u16::<LittleEndian>(v)?,
            &Self::I16(v) => w.write_i16::<LittleEndian>(v)?,
            &Self::U32(v) => w.write_u32::<LittleEndian>(v)?,
            &Self::I32(v) => w.write_i32::<LittleEndian>(v)?,
            &Self::U64(v) => w.write_u64::<LittleEndian>(v)?,
            &Self::I64(v) => w.write_i64::<LittleEndian>(v)?,
            &Self::F32(v) => w.write_f32::<LittleEndian>(v)?,
            &Self::F64(v) => w.write_f64::<LittleEndian>(v)?,
            &Self::Bool(v) => w.write_u8(u8::from(v))?,
            Self::String(v) => write_string(w, v.as_str())?,
            Self::Array(v) => {
                // The `Value` type does not enforce that all the values in an Array have the same
                // type.
                let value_type = if v.is_empty() {
                    // Doesn't matter, the array is empty.
                    ValueType::U32
                } else {
                    let value_type: std::collections::HashSet<_> =
                        v.iter().map(|elem| elem.value_type()).collect();
                    if value_type.len() != 1 {
                        crate::bail!("multiple value-types in the same array {value_type:?}")
                    }
                    value_type.into_iter().next().context("empty value_type")?
                };
                w.write_u32::<LittleEndian>(value_type.to_u32())?;
                w.write_u64::<LittleEndian>(v.len() as u64)?;
                for elem in v.iter() {
                    elem.write(w)?
                }
            }
        }
        Ok(())
    }
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self> {
        let v = match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            v => crate::bail!("unrecognized value-type {v:#08x}"),
        };
        Ok(v)
    }

    fn to_u32(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }

    /// Minimum on-disk size of one value of this type, used to reject array
    /// lengths that exceed the remaining file size before allocating.
    fn min_disk_size(&self, magic: &VersionedMagic) -> u64 {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String => magic.length_prefix_size(),
            Self::Array => 4 + magic.length_prefix_size(),
        }
    }
}

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(reader: &mut R) -> Result<Self> {
        Self::read_with_limits(reader, ContentReadLimits::default())
    }

    pub fn read_with_limits<R: std::io::Seek + std::io::Read>(
        reader: &mut R,
        limits: ContentReadLimits,
    ) -> Result<Self> {
        // Capture the file size once so the bounds checks below don't have to
        // seek to the end and back on every length-prefixed read.
        let start = reader.stream_position()?;
        let file_size = reader.seek(std::io::SeekFrom::End(0))?;
        reader.seek(std::io::SeekFrom::Start(start))?;
        let header_cap = start.saturating_add(limits.max_header_bytes);
        let header_limit = header_cap.min(file_size);

        let magic = VersionedMagic::read(reader)?;
        let tensor_count = read_length(reader, &magic)?;
        let metadata_kv_count = read_length(reader, &magic)?;

        if tensor_count > limits.max_tensor_count {
            crate::bail!(
                "gguf: tensor_count {tensor_count} exceeds max {}",
                limits.max_tensor_count
            )
        }
        if metadata_kv_count > limits.max_metadata_count {
            crate::bail!(
                "gguf: metadata_kv_count {metadata_kv_count} exceeds max {}",
                limits.max_metadata_count
            )
        }

        // Reject header-declared counts that can't fit in the file at minimum size.
        // Per-entry minima: a metadata kv is at least `key_len_prefix + u32 value_type
        // + 1 byte value`; a tensor info is at least `name_len_prefix + u32 n_dims
        // + u32 dtype + u64 offset`.
        let prefix = magic.length_prefix_size();
        let min_per_kv = prefix + 4 + 1;
        let min_per_tensor = prefix + 4 + 4 + 8;
        let needed = metadata_kv_count
            .saturating_mul(min_per_kv)
            .saturating_add(tensor_count.saturating_mul(min_per_tensor));
        let remaining = remaining_bytes(reader, header_limit)?;
        if needed > remaining {
            crate::bail!(
                "gguf: header declares {tensor_count} tensors and {metadata_kv_count} metadata entries, needs at least {needed} bytes, only {remaining} remaining"
            )
        }

        let mut metadata = HashMap::new();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, &magic, header_limit, &limits)?;
            let value_type = reader.read_u32::<LittleEndian>()?;
            let value_type = ValueType::from_u32(value_type)?;
            let value = Value::read(reader, value_type, &magic, 0, header_limit, &limits)?;
            if metadata.insert(key.clone(), value).is_some() {
                crate::bail!("gguf: duplicate metadata key '{key}'")
            }
        }
        let mut tensor_infos = HashMap::new();
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader, &magic, header_limit, &limits)?;
            let n_dimensions = reader.read_u32::<LittleEndian>()?;
            if n_dimensions > GGUF_MAX_TENSOR_DIMS {
                crate::bail!(
                    "gguf: tensor '{tensor_name}' has {n_dimensions} dimensions, max is {GGUF_MAX_TENSOR_DIMS}"
                )
            }

            let mut dimensions: Vec<usize> = match magic {
                VersionedMagic::GgufV1 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader.read_u32_into::<LittleEndian>(&mut dimensions)?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
                VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader.read_u64_into::<LittleEndian>(&mut dimensions)?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
            };

            dimensions.reverse();
            let ggml_dtype = reader.read_u32::<LittleEndian>()?;
            let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
            let offset = reader.read_u64::<LittleEndian>()?;
            if tensor_infos
                .insert(
                    tensor_name.clone(),
                    TensorInfo {
                        shape: crate::Shape::from(dimensions),
                        offset,
                        ggml_dtype,
                    },
                )
                .is_some()
            {
                crate::bail!("gguf: duplicate tensor name '{tensor_name}'")
            }
        }
        let position = reader.stream_position()?;
        if position > header_limit {
            crate::bail!("gguf: header exceeds max {} bytes", limits.max_header_bytes)
        }
        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::U64(v)) => *v,
            Some(Value::I8(v)) if *v > 0 => *v as u64,
            Some(Value::I16(v)) if *v > 0 => *v as u64,
            Some(Value::I32(v)) if *v > 0 => *v as u64,
            Some(Value::I64(v)) if *v > 0 => *v as u64,
            Some(value) => crate::bail!(
                "gguf: general.alignment must be a positive integer, got {:?}",
                value.value_type()
            ),
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            crate::bail!("gguf: general.alignment {alignment} must be a positive power of two")
        }
        let tensor_data_offset = position
            .checked_add(alignment - 1)
            .ok_or_else(|| crate::Error::Msg("gguf: tensor data alignment overflow".into()))?
            / alignment
            * alignment;
        if tensor_data_offset > header_cap {
            crate::bail!(
                "gguf: aligned header exceeds max {} bytes",
                limits.max_header_bytes
            )
        }
        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }

    pub fn tensor<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<QTensor> {
        let tensor_info = match self.tensor_infos.get(name) {
            Some(tensor_info) => tensor_info,
            None => crate::bail!("cannot find tensor info for {name}"),
        };
        tensor_info.read(reader, self.tensor_data_offset, device)
    }
}

fn write_string<W: std::io::Write>(w: &mut W, str: &str) -> Result<()> {
    let bytes = str.as_bytes();
    w.write_u64::<LittleEndian>(bytes.len() as u64)?;
    w.write_all(bytes)?;
    Ok(())
}

pub fn write<W: std::io::Seek + std::io::Write>(
    w: &mut W,
    metadata: &[(&str, &Value)],
    tensors: &[(&str, &QTensor)],
) -> Result<()> {
    w.write_u32::<LittleEndian>(0x46554747)?;
    w.write_u32::<LittleEndian>(2)?; // version 2.
    w.write_u64::<LittleEndian>(tensors.len() as u64)?;
    w.write_u64::<LittleEndian>(metadata.len() as u64)?;
    for (name, value) in metadata.iter() {
        write_string(w, name)?;
        w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
        value.write(w)?;
    }
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors.iter() {
        write_string(w, name)?;
        let dims = tensor.shape().dims();
        w.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &dim in dims.iter().rev() {
            w.write_u64::<LittleEndian>(dim as u64)?;
        }
        w.write_u32::<LittleEndian>(tensor.dtype().to_u32())?;
        w.write_u64::<LittleEndian>(offset as u64)?;
        offsets.push(offset);
        let size_in_bytes = tensor.storage_size_in_bytes();
        let padding = 31 - (31 + size_in_bytes) % 32;
        offset += size_in_bytes + padding;
    }
    let pos = w.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    w.write_all(&vec![0u8; padding])?;
    let tensor_start_pos = w.stream_position()? as usize;
    for (offset, (_name, tensor)) in offsets.iter().zip(tensors.iter()) {
        let pos = w.stream_position()? as usize;
        if tensor_start_pos + offset != pos {
            crate::bail!(
                "internal error, unexpected current position {tensor_start_pos} {offset} {pos}"
            )
        }
        let data = tensor.data()?;
        let size_in_bytes = data.len();
        w.write_all(&data)?;
        let padding = 31 - (31 + size_in_bytes) % 32;
        w.write_all(&vec![0u8; padding])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    fn version_two_prefix(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.write_u32::<LittleEndian>(0x46554747).unwrap();
        bytes.write_u32::<LittleEndian>(2).unwrap();
        bytes.write_u64::<LittleEndian>(tensor_count).unwrap();
        bytes.write_u64::<LittleEndian>(metadata_count).unwrap();
        bytes
    }

    fn append_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.write_u64::<LittleEndian>(value.len() as u64).unwrap();
        bytes.write_all(value.as_bytes()).unwrap();
    }

    #[test]
    fn duplicate_metadata_keys_are_rejected() {
        let mut bytes = version_two_prefix(0, 2);
        for value in [1u32, 2u32] {
            append_string(&mut bytes, "duplicate");
            bytes
                .write_u32::<LittleEndian>(ValueType::U32.to_u32())
                .unwrap();
            bytes.write_u32::<LittleEndian>(value).unwrap();
        }
        let error = Content::read(&mut Cursor::new(bytes))
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate metadata key 'duplicate'"));
    }

    #[test]
    fn duplicate_tensor_names_are_rejected() {
        let mut bytes = version_two_prefix(2, 0);
        for offset in [0u64, 32u64] {
            append_string(&mut bytes, "duplicate");
            bytes.write_u32::<LittleEndian>(1).unwrap();
            bytes.write_u64::<LittleEndian>(1).unwrap();
            bytes.write_u32::<LittleEndian>(0).unwrap();
            bytes.write_u64::<LittleEndian>(offset).unwrap();
        }
        let error = Content::read(&mut Cursor::new(bytes))
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate tensor name 'duplicate'"));
    }

    #[test]
    fn invalid_alignment_is_rejected_without_panicking() {
        for alignment in [0u32, 3u32] {
            let mut bytes = version_two_prefix(0, 1);
            append_string(&mut bytes, "general.alignment");
            bytes
                .write_u32::<LittleEndian>(ValueType::U32.to_u32())
                .unwrap();
            bytes.write_u32::<LittleEndian>(alignment).unwrap();
            let error = Content::read(&mut Cursor::new(bytes))
                .unwrap_err()
                .to_string();
            assert!(error.contains("alignment") && error.contains("power of two"));
        }
    }

    #[test]
    fn caller_limits_reject_counts_and_strings_before_allocation() {
        let count_limits = ContentReadLimits {
            max_tensor_count: 1,
            max_metadata_count: 1,
            ..ContentReadLimits::default()
        };
        let error =
            Content::read_with_limits(&mut Cursor::new(version_two_prefix(2, 0)), count_limits)
                .unwrap_err()
                .to_string();
        assert!(error.contains("tensor_count 2 exceeds max 1"));

        let mut oversized_string = version_two_prefix(0, 1);
        oversized_string.write_u64::<LittleEndian>(9).unwrap();
        oversized_string.extend_from_slice(&[0; 9]);
        let string_limits = ContentReadLimits {
            max_string_length: 8,
            ..ContentReadLimits::default()
        };
        let error = Content::read_with_limits(&mut Cursor::new(oversized_string), string_limits)
            .unwrap_err()
            .to_string();
        assert!(error.contains("string length 9 exceeds max 8"));
    }
}
