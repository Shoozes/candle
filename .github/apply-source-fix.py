from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old!r}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def remove_suffix(path: str, suffix: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if not text.endswith(suffix):
        raise SystemExit(f"{path}: expected suffix is missing: {suffix!r}")
    file_path.write_text(text[: -len(suffix)] + "\n", encoding="utf-8")


replace_once(
    "candle-transformers/src/models/lfm2/config.rs",
    "\n/// Cache for LFM2 model supporting both attention KV cache and convolution state cache.\n#[derive(Debug, Clone)]\n",
    "\n",
)
replace_once(
    "candle-transformers/src/models/lfm2/cache.rs",
    "pub struct Cache {",
    "/// Cache for LFM2 model supporting both attention KV cache and convolution state cache.\n#[derive(Debug, Clone)]\npub struct Cache {",
)

remove_suffix(
    "candle-transformers/src/models/lfm2_vl/model/types.rs",
    "\n#[derive(Debug)]\n",
)
replace_once(
    "candle-transformers/src/models/lfm2_vl/model/runtime.rs",
    "pub struct Lfm2VlModel {",
    "#[derive(Debug)]\npub struct Lfm2VlModel {",
)
replace_once(
    "candle-transformers/src/models/lfm2_vl/model/encoding.rs",
    "\n/// Merge projected image features into explicit placeholder spans.\n///\n/// This is shared by dense native text and quantized GGUF text. The only\n/// cross-device value is `encoded_images.embeddings`, transferred here to the\n/// text embedding device and dtype immediately before the span replacement.\n",
    "\n",
)
replace_once(
    "candle-transformers/src/models/lfm2_vl/model/merge.rs",
    "pub fn merge_projected_embeddings(",
    "/// Merge projected image features into explicit placeholder spans.\n///\n/// This is shared by dense native text and quantized GGUF text. The only\n/// cross-device value is `encoded_images.embeddings`, transferred here to the\n/// text embedding device and dtype immediately before the span replacement.\npub fn merge_projected_embeddings(",
)

remove_suffix(
    "candle-transformers/src/models/lfm2_vl/weights/manifest.rs",
    "\n#[derive(Debug)]\n",
)
replace_once(
    "candle-transformers/src/models/lfm2_vl/weights/runtime.rs",
    "pub struct Mmproj {",
    "#[derive(Debug)]\npub struct Mmproj {",
)

remove_suffix(
    "candle-transformers/src/models/siglip2/config.rs",
    "\n#[derive(Debug)]\n",
)
replace_once(
    "candle-transformers/src/models/siglip2/embeddings.rs",
    "struct VisionEmbeddings {",
    "#[derive(Debug)]\nstruct VisionEmbeddings {",
)
remove_suffix(
    "candle-transformers/src/models/siglip2/embeddings.rs",
    "\n#[derive(Clone, Debug)]\n",
)
replace_once(
    "candle-transformers/src/models/siglip2/encoder.rs",
    "struct Attention {",
    "#[derive(Clone, Debug)]\nstruct Attention {",
)
remove_suffix(
    "candle-transformers/src/models/siglip2/encoder.rs",
    "\n/// Candle SigLIP2 NaFlex vision encoder for packed patch tensors.\n#[derive(Debug)]\n",
)
replace_once(
    "candle-transformers/src/models/siglip2/model.rs",
    "pub struct Siglip2VisionModel {",
    "/// Candle SigLIP2 NaFlex vision encoder for packed patch tensors.\n#[derive(Debug)]\npub struct Siglip2VisionModel {",
)
