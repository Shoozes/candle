use candle_core::{npy::NpzTensors, pickle::PthTensors, Device, Tensor};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

type TestResult = Result<(), Box<dyn std::error::Error>>;
type Entries = Vec<(String, Vec<u8>)>;

struct Sandbox(PathBuf);

impl Sandbox {
    fn new() -> std::io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "candle-archive-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }

    fn finish(self) -> std::io::Result<()> {
        std::fs::remove_dir_all(&self.0)?;
        assert!(!self.0.exists());
        Ok(())
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        if self.0.exists() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn archive(entries: &[(String, Vec<u8>)]) -> zip::result::ZipResult<Vec<u8>> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, data) in entries {
        writer.start_file(name, options)?;
        writer.write_all(data)?;
    }
    Ok(writer.finish()?.into_inner())
}

fn pth_entries() -> Result<Entries, Box<dyn std::error::Error>> {
    let mut zip = zip::ZipArchive::new(Cursor::new(include_bytes!("test.pt")))?;
    let mut entries = Vec::new();
    for i in 0..zip.len() {
        let mut file = zip.by_index(i)?;
        let name = file.name()?.into_owned();
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        entries.push((name, data));
    }
    Ok(entries)
}

fn npy() -> Vec<u8> {
    let mut header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }".to_owned();
    while !(10 + header.len() + 1).is_multiple_of(64) {
        header.push(' ');
    }
    header.push('\n');
    let mut data = b"\x93NUMPY\x01\x00".to_vec();
    data.extend_from_slice(&(header.len() as u16).to_le_bytes());
    data.extend_from_slice(header.as_bytes());
    for value in [1.25f32, -2.5] {
        data.extend_from_slice(&value.to_le_bytes());
    }
    data
}

fn cp437_names(mut data: Vec<u8>) -> Vec<u8> {
    // All fixture names start with ASCII x; replace it with CP437 é.
    for (signature, flags_offset, name_offset) in [(b"PK\x03\x04", 6, 30), (b"PK\x01\x02", 8, 46)] {
        let starts: Vec<_> = data
            .windows(4)
            .enumerate()
            .filter_map(|(i, bytes)| (bytes == signature).then_some(i))
            .collect();
        assert!(!starts.is_empty());
        for start in starts {
            assert_eq!(data[start + name_offset], b'x');
            data[start + flags_offset + 1] &= !8;
            data[start + name_offset] = 0x82;
        }
    }
    data
}

fn assert_npz_payload_error(path: &Path) -> TestResult {
    assert!(Tensor::read_npz(path).is_err());
    assert!(Tensor::read_npz_by_name(path, &["x"]).is_err());
    let lazy = NpzTensors::new(path)?;
    assert!(lazy.get("x").is_err());
    Ok(())
}

#[test]
fn npz_roundtrip_named_lazy_and_unicode() -> TestResult {
    let dir = Sandbox::new()?;
    let path = dir.0.join("tensors.npz");
    let values = Tensor::new(&[1.25f32, -2.5], &Device::Cpu)?;
    let empty = Tensor::zeros((0, 2), candle_core::DType::F32, &Device::Cpu)?;
    Tensor::write_npz(&[("α/weights", &values), ("empty", &empty)], &path)?;
    let before = std::fs::read(&path)?;
    let all = Tensor::read_npz(&path)?;
    assert_eq!(all[0].0, "α/weights");
    assert_eq!(all[0].1.to_vec1::<f32>()?, [1.25, -2.5]);
    assert_eq!(all[1].1.dims(), [0, 2]);
    let named = Tensor::read_npz_by_name(&path, &["empty", "α/weights", "α/weights"])?;
    assert_eq!(named[0].dims(), [0, 2]);
    assert_eq!(named[2].to_vec1::<f32>()?, [1.25, -2.5]);
    assert!(Tensor::read_npz_by_name(&path, &["absent"]).is_err());
    let lazy = NpzTensors::new(&path)?;
    assert_eq!(lazy.get_shape_and_dtype("α/weights")?.0.dims(), [2]);
    assert_eq!(
        lazy.get("α/weights")?.unwrap().to_vec1::<f32>()?,
        [1.25, -2.5]
    );
    assert!(lazy.get("absent")?.is_none());
    assert_eq!(before, std::fs::read(&path)?);
    dir.finish()?;
    Ok(())
}

#[test]
fn truncated_archives_return_errors_without_changing_inputs() -> TestResult {
    let dir = Sandbox::new()?;
    let bytes = archive(&[("x.npy".into(), npy())])?;
    for length in [0, 3, bytes.len() / 2, bytes.len() - 8] {
        let path = dir.0.join("truncated.zip");
        std::fs::write(&path, &bytes[..length])?;
        assert!(Tensor::read_npz(&path).is_err());
        assert!(NpzTensors::new(&path).is_err());
        assert!(PthTensors::new(&path, None).is_err());
        assert_eq!(std::fs::read(&path)?, bytes[..length]);
    }
    dir.finish()?;
    Ok(())
}

#[test]
fn legacy_cp437_names_preserve_named_and_lazy_lookup() -> TestResult {
    let dir = Sandbox::new()?;
    let path = dir.0.join("legacy.npz");
    let bytes = cp437_names(archive(&[("x.npy".into(), npy())])?);
    std::fs::write(&path, &bytes)?;
    assert_eq!(Tensor::read_npz(&path)?[0].0, "é");
    assert_eq!(
        Tensor::read_npz_by_name(&path, &["é"])?[0].to_vec1::<f32>()?,
        [1.25, -2.5]
    );
    assert_eq!(
        NpzTensors::new(&path)?
            .get("é")?
            .unwrap()
            .to_vec1::<f32>()?,
        [1.25, -2.5]
    );
    assert_eq!(std::fs::read(&path)?, bytes);
    let entries: Vec<_> = pth_entries()?
        .into_iter()
        .map(|(name, bytes)| {
            let (_, tail) = name.split_once('/').unwrap();
            (format!("x/{tail}"), bytes)
        })
        .collect();
    let path = dir.0.join("legacy.pt");
    let bytes = cp437_names(archive(&entries)?);
    std::fs::write(&path, &bytes)?;
    let tensors = PthTensors::new(&path, None)?;
    assert_eq!(
        tensors.get("test")?.unwrap().to_vec2::<i64>()?,
        [[1, 2, 3, 4], [5, 6, 7, 8]]
    );
    assert_eq!(std::fs::read(&path)?, bytes);
    dir.finish()?;
    Ok(())
}

#[test]
fn malformed_npy_payloads_return_controlled_errors() -> TestResult {
    let dir = Sandbox::new()?;
    let valid = npy();
    for payload in [
        b"bad magic".to_vec(),
        valid[..8].to_vec(),
        valid[..valid.len() - 1].to_vec(),
    ] {
        let path = dir.0.join("bad-payload.npz");
        let bytes = archive(&[("x.npy".into(), payload)])?;
        std::fs::write(&path, &bytes)?;
        assert_npz_payload_error(&path)?;
        assert_eq!(std::fs::read(&path)?, bytes);
    }
    dir.finish()?;
    Ok(())
}

#[test]
fn unsupported_compression_stays_a_controlled_error() -> TestResult {
    let dir = Sandbox::new()?;
    let mut bytes = archive(&[("x.npy".into(), npy())])?;
    for (signature, method_offset) in [(b"PK\x03\x04", 8), (b"PK\x01\x02", 10)] {
        let start = bytes.windows(4).position(|b| b == signature).unwrap();
        bytes[start + method_offset] = 99; // Unsupported encrypted compression.
    }
    let path = dir.0.join("unsupported.npz");
    std::fs::write(&path, &bytes)?;
    assert!(Tensor::read_npz(&path).is_err());
    assert!(Tensor::read_npz_by_name(&path, &["x"]).is_err());
    dir.finish()?;
    Ok(())
}

#[test]
fn pth_repacked_metadata_and_storage_load() -> TestResult {
    let dir = Sandbox::new()?;
    let path = dir.0.join("weights.pt");
    let bytes = archive(&pth_entries()?)?;
    std::fs::write(&path, &bytes)?;
    let tensors = PthTensors::new(&path, None)?;
    assert_eq!(
        tensors.get("test")?.unwrap().to_vec2::<i64>()?,
        [[1, 2, 3, 4], [5, 6, 7, 8]]
    );
    assert!(tensors.get("absent")?.is_none());
    assert_eq!(std::fs::read(&path)?, bytes);
    dir.finish()?;
    Ok(())
}

#[test]
fn malformed_pth_metadata_and_storage_return_errors() -> TestResult {
    let dir = Sandbox::new()?;
    for corrupt_metadata in [true, false] {
        let mut entries = pth_entries()?;
        let (_, payload) = entries
            .iter_mut()
            .find(|(name, _)| {
                if corrupt_metadata {
                    name.ends_with("data.pkl")
                } else {
                    name.ends_with("data/0")
                }
            })
            .unwrap();
        payload.truncate(1);
        let path = dir.0.join("malformed.pt");
        let bytes = archive(&entries)?;
        std::fs::write(&path, &bytes)?;
        if corrupt_metadata {
            assert!(PthTensors::new(&path, None).is_err());
        } else {
            assert!(PthTensors::new(&path, None)?.get("test").is_err());
        }
        assert_eq!(std::fs::read(&path)?, bytes);
    }
    dir.finish()?;
    Ok(())
}

#[test]
fn readers_never_extract_parent_paths_or_overwrite_caller_files() -> TestResult {
    let dir = Sandbox::new()?;
    let archive_dir = dir.0.join("archives");
    std::fs::create_dir(&archive_dir)?;
    let caller = dir.0.join("caller.npy");
    std::fs::write(&caller, b"caller-owned bytes")?;
    let path = archive_dir.join("parent.npz");
    std::fs::write(&path, archive(&[("../caller.npy".into(), npy())])?)?;
    assert_eq!(Tensor::read_npz(&path)?[0].0, "../caller");
    assert_eq!(
        NpzTensors::new(&path)?
            .get("../caller")?
            .unwrap()
            .to_vec1::<f32>()?,
        [1.25, -2.5]
    );
    let mut entries = pth_entries()?;
    entries.push(("../caller.npy".into(), b"archive bytes".to_vec()));
    let path = archive_dir.join("parent.pt");
    std::fs::write(&path, archive(&entries)?)?;
    assert!(PthTensors::new(&path, None)?.get("test")?.is_some());
    assert_eq!(std::fs::read(&caller)?, b"caller-owned bytes");
    dir.finish()?;
    Ok(())
}
