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
    pub fn write_verified_report<T: Serialize + ?Sized>(
        &mut self,
        path: &Path,
        value: &T,
    ) -> Result<()> {
        self.recheck()?;
        write_json(path, value)
    }

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
    write_tensors_before_publish(path, tensors, || Ok(()))
}

fn write_tensors_before_publish(
    path: &Path,
    tensors: &[(&str, &Tensor)],
    before_publish: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let values = tensors
        .iter()
        .map(|(name, tensor)| Ok(((*name).to_string(), tensor.to_device(&candle::Device::Cpu)?)))
        .collect::<candle::Result<std::collections::HashMap<_, _>>>()?;
    let stage = TensorStage::new(path)?;
    let result = (|| -> Result<()> {
        safetensors::save(&values, &stage.tensor)?;
        OpenOptions::new()
            .write(true)
            .open(&stage.tensor)?
            .sync_all()?;
        before_publish()?;
        fs::hard_link(&stage.tensor, path).with_context(|| {
            format!(
                "publishing tensor trace without replacement: {}",
                path.display()
            )
        })?;
        Ok(())
    })();
    match (result, stage.cleanup()) {
        (Err(error), Err(cleanup)) => {
            Err(error.context(format!("trace staging cleanup also failed: {cleanup}")))
        }
        (Err(error), _) => Err(error),
        (Ok(()), cleanup) => cleanup,
    }
}

// Stage beside the destination so publication is exclusive without buffering its payload.
struct TensorStage {
    directory: PathBuf,
    tensor: PathBuf,
}

impl TensorStage {
    fn new(destination: &Path) -> Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let parent = parent.canonicalize()?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let directory = parent.join(format!(
            ".candle-d1-tensor-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory)?;
        let tensor = directory.join("tensor.safetensors");
        Ok(Self { directory, tensor })
    }

    fn cleanup(&self) -> Result<()> {
        match fs::remove_file(&self.tensor) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => return Err(error.into()),
        }
        match fs::remove_dir(&self.directory) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
}

impl Drop for TensorStage {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

pub fn load_image(path: &Path) -> Result<image::DynamicImage> {
    let bytes = read_bounded(path, 16 * 1024 * 1024)?;
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(64 * 1024 * 1024);
    reader.limits(limits);
    reader.decode().context("decoding bounded d1 image")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Result<Self> {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "candle-d1-output-{}-{nonce}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }

        fn retained(&self) -> Result<RetainedArtifacts> {
            let path = self.0.join("artifact.bin");
            let bytes = b"admitted";
            fs::write(&path, bytes)?;
            // A mutable test handle exercises recheck failure on Windows too.
            let file = File::open(&path)?;
            Ok(RetainedArtifacts {
                files: vec![(
                    file,
                    path,
                    Pin {
                        file: "artifact.bin".into(),
                        bytes: bytes.len() as u64,
                        sha256: sha256(bytes),
                    },
                )],
                identities: Vec::new(),
            })
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            for name in [
                "trace.safetensors",
                "expected.safetensors",
                "artifact.bin",
                "report.json",
            ] {
                let _ = fs::remove_file(self.0.join(name));
            }
            let _ = fs::remove_dir(&self.0);
        }
    }

    #[test]
    fn tensor_trace_preserves_bytes_layout_and_existing_files() -> Result<()> {
        let root = TestDir::new()?;
        let tensor = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?.t()?;
        let ids = Tensor::new(&[2u32, 0], &Device::Cpu)?;
        let path = root.0.join("trace.safetensors");
        write_tensors(&path, &[("logits", &tensor), ("ids", &ids)])?;
        let expected = std::collections::HashMap::from([
            ("logits".to_owned(), tensor),
            ("ids".to_owned(), ids),
        ]);
        safetensors::save(&expected, root.0.join("expected.safetensors"))?;
        let bytes = fs::read(&path)?;
        assert_eq!(bytes, fs::read(root.0.join("expected.safetensors"))?);
        let loaded = safetensors::load(&path, &Device::Cpu)?;
        assert_eq!(
            loaded["logits"].to_vec2::<f32>()?,
            vec![vec![1., 3.], vec![2., 4.]]
        );
        assert_eq!(loaded["ids"].to_vec1::<u32>()?, vec![2, 0]);
        assert!(write_tensor(&path, "replacement", &Tensor::new(0f32, &Device::Cpu)?).is_err());
        assert_eq!(fs::read(&path)?, bytes);
        Ok(())
    }

    #[test]
    fn tensor_trace_preserves_a_racing_writer() -> Result<()> {
        let root = TestDir::new()?;
        let path = root.0.join("trace.safetensors");
        let tensor = Tensor::new(&[1f32, 2.], &Device::Cpu)?;
        let result = write_tensors_before_publish(&path, &[("logits", &tensor)], || {
            fs::write(&path, b"competing writer")?;
            Ok(())
        });
        assert!(result.is_err(), "racing destination must be rejected");
        assert_eq!(fs::read(&path)?, b"competing writer");
        assert_eq!(
            fs::read_dir(&root.0)?.count(),
            1,
            "staging must be released"
        );
        Ok(())
    }

    #[test]
    fn tensor_trace_failure_leaves_no_output_or_staging() -> Result<()> {
        let root = TestDir::new()?;
        let path = root.0.join("trace.safetensors");
        let tensor = Tensor::new(&[1f32, 2.], &Device::Cpu)?;
        let result = write_tensors_before_publish(&path, &[("logits", &tensor)], || {
            anyhow::bail!("injected failure before publication")
        });
        assert!(result.is_err());
        assert!(!path.exists());
        assert_eq!(fs::read_dir(&root.0)?.count(), 0);
        Ok(())
    }

    #[test]
    fn failed_artifact_recheck_cannot_publish_a_complete_report() -> Result<()> {
        let root = TestDir::new()?;
        let mut retained = root.retained()?;
        fs::write(root.0.join("artifact.bin"), b"modified")?;
        let report = root.0.join("report.json");
        assert!(retained
            .write_verified_report(&report, &json!({"state":"complete"}))
            .is_err());
        assert!(
            !report.exists(),
            "invalid artifacts cannot publish completion"
        );
        Ok(())
    }

    #[test]
    fn verified_report_preserves_json_and_refuses_replacement() -> Result<()> {
        let root = TestDir::new()?;
        let mut retained = root.retained()?;
        let path = root.0.join("report.json");
        let report = json!({"schema":"candle.d1_run.v1","state":"complete","language_forwards":0});
        retained.write_verified_report(&path, &report)?;
        let bytes = fs::read(&path)?;
        assert_eq!(serde_json::from_slice::<Value>(&bytes)?, report);
        assert_eq!(bytes.last(), Some(&b'\n'));
        assert!(retained
            .write_verified_report(&path, &json!({"state":"failed"}))
            .is_err());
        assert_eq!(fs::read(&path)?, bytes);
        Ok(())
    }
}
