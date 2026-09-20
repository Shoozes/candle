use candle::{bail, Result};
use std::path::{Path, PathBuf};

/// The local files required before an experimental GPT-OSS load can begin.
///
/// This is a deliberately small admission boundary.  It never downloads a
/// model and refuses a directory that only contains configuration or
/// tokenizer files, so model-free tests cannot accidentally claim execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GptOssCheckpoint {
    root: PathBuf,
    config: PathBuf,
    tokenizer: PathBuf,
    weights: Vec<PathBuf>,
}

impl GptOssCheckpoint {
    /// Validate a local original-format checkpoint directory.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let metadata = std::fs::metadata(root).map_err(|error| {
            candle::Error::Msg(format!("GPT-OSS checkpoint directory {:?}: {error}", root))
        })?;
        if !metadata.is_dir() {
            bail!("GPT-OSS checkpoint path {:?} is not a directory", root);
        }

        let config = root.join("config.json");
        if !config.is_file() {
            bail!("GPT-OSS checkpoint {:?} is missing config.json", root);
        }
        let tokenizer = root.join("tokenizer.json");
        if !tokenizer.is_file() {
            bail!("GPT-OSS checkpoint {:?} is missing tokenizer.json", root);
        }

        let mut weights = Vec::new();
        let entries = std::fs::read_dir(root).map_err(|error| {
            candle::Error::Msg(format!("GPT-OSS checkpoint directory {:?}: {error}", root))
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                candle::Error::Msg(format!("GPT-OSS checkpoint directory {:?}: {error}", root))
            })?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
                weights.push(path);
            }
        }
        weights.sort();
        if weights.is_empty() {
            bail!(
                "GPT-OSS checkpoint {:?} contains no .safetensors model weights",
                root
            );
        }

        Ok(Self {
            root: root.to_path_buf(),
            config,
            tokenizer,
            weights,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config_path(&self) -> &Path {
        &self.config
    }

    pub fn tokenizer_path(&self) -> &Path {
        &self.tokenizer
    }

    pub fn weight_paths(&self) -> &[PathBuf] {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "candle-gpt-oss-test-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn rejects_model_free_checkpoint_directory() {
        let root = temp_dir();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), b"{}").unwrap();
        std::fs::write(root.join("tokenizer.json"), b"{}").unwrap();

        let error = GptOssCheckpoint::open(&root).expect_err("weights must be required");
        assert!(error.to_string().contains("no .safetensors model weights"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_missing_checkpoint_without_downloading() {
        let root = temp_dir();
        let error = GptOssCheckpoint::open(&root).expect_err("missing checkpoint must fail");
        assert!(error.to_string().contains("checkpoint directory"));
        assert!(!root.exists());
    }
}
