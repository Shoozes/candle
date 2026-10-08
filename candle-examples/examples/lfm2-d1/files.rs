use anyhow::{Context, Result};
use candle::{safetensors, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
pub struct Pin {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Deserialize)]
struct Manifest {
    files: Vec<Pin>,
}

pub struct RetainedArtifacts {
    files: Vec<(File, PathBuf, Pin)>,
    pub identities: Vec<Value>,
}

impl RetainedArtifacts {
    pub fn recheck(&mut self) -> Result<()> {
        for (file, path, pin) in &mut self.files {
            anyhow::ensure!(
                file.metadata()?.len() == pin.bytes && fs::metadata(&*path)?.len() == pin.bytes,
                "artifact changed: {}",
                path.display()
            );
            file.seek(SeekFrom::Start(0))?;
            anyhow::ensure!(
                hash_file(file)? == pin.sha256,
                "retained artifact changed: {}",
                path.display()
            );
            anyhow::ensure!(
                hash_file(&mut File::open(&*path)?)? == pin.sha256,
                "artifact binding changed: {}",
                path.display()
            );
        }
        Ok(())
    }
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_file(file: &mut File) -> Result<String> {
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    anyhow::ensure!(
        size > 0 && size <= maximum,
        "input size outside its bound: {}",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() as u64 == size,
        "input changed during read: {}",
        path.display()
    );
    Ok(bytes)
}

pub fn retain_artifacts(manifest: &Path, paths: &[&PathBuf]) -> Result<RetainedArtifacts> {
    let manifest: Manifest = serde_json::from_slice(&read_bounded(manifest, 1024 * 1024)?)?;
    let mut files = Vec::new();
    let mut identities = Vec::new();
    for path in paths {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .context("artifact filename is not UTF-8")?;
        let mut candidates = manifest.files.iter().filter(|pin| pin.file == name);
        let pin = candidates
            .next()
            .context("artifact is absent from the supplied manifest")?;
        anyhow::ensure!(candidates.next().is_none(), "duplicate artifact pin");
        let metadata = fs::symlink_metadata(path)?;
        anyhow::ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "artifact must be a regular file"
        );
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            anyhow::ensure!(
                metadata.file_attributes() & 0x400 == 0,
                "artifact reparse point rejected"
            );
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.share_mode(1).custom_flags(0x0020_0000);
        }
        let mut file = options.open(path)?;
        anyhow::ensure!(
            file.metadata()?.len() == pin.bytes,
            "artifact size differs from manifest"
        );
        anyhow::ensure!(
            hash_file(&mut file)? == pin.sha256,
            "artifact hash differs: {}",
            path.display()
        );
        identities.push(json!({"path":path,"bytes":pin.bytes,"sha256":pin.sha256}));
        files.push((
            file,
            (*path).clone(),
            Pin {
                file: pin.file.clone(),
                bytes: pin.bytes,
                sha256: pin.sha256.clone(),
            },
        ));
    }
    Ok(RetainedArtifacts { files, identities })
}

pub fn write_json<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

pub fn write_tensor(path: &Path, name: &str, tensor: &Tensor) -> Result<()> {
    write_tensors(path, &[(name, tensor)])
}
pub fn write_tensors(path: &Path, tensors: &[(&str, &Tensor)]) -> Result<()> {
    anyhow::ensure!(!path.exists(), "trace already exists");
    let values = tensors
        .iter()
        .map(|(name, tensor)| Ok(((*name).to_string(), tensor.to_device(&candle::Device::Cpu)?)))
        .collect::<candle::Result<std::collections::HashMap<_, _>>>()?;
    safetensors::save(&values, path)?;
    Ok(())
}

pub fn load_image(path: &Path) -> Result<image::DynamicImage> {
    let bytes = read_bounded(path, 16 * 1024 * 1024)?;
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(64 * 1024 * 1024);
    reader.limits(limits);
    reader.decode().context("decoding bounded d1 image")
}
