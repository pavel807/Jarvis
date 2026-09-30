#!/usr/bin/env python3
"""
Download IBM Granite Speech 5.0 TurboCTC directly from Hugging Face.
No conversion needed, model is already in safetensors format.
"""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "ibm-granite/granite-speech-5.0-470m-turboctc"
OUT_DIR = Path("models/granite-speech-5")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading {REPO}...")

    files_to_download = [
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "processor_config.json"
    ]

    for fname in files_to_download:
        try:
            path = hf_hub_download(repo_id=REPO, filename=fname, local_dir=OUT_DIR)
            print(f"✅ {fname} -> {path}")
        except Exception as e:
            print(f"⚠️ Failed to download {fname}: {e}")

    # Analyze tokenizer to find vocab size for Rust
    tok_path = OUT_DIR / "tokenizer.json"
    if tok_path.exists():
        tok_data = json.loads(tok_path.read_text(encoding="utf-8"))
        if "model" in tok_data and "vocab" in tok_data["model"]:
            vocab_size = len(tok_data["model"]["vocab"])
            print(f"\n🔍 Tokenizer analysis:")
            print(f"   Vocab size: {vocab_size}")
            print(f"   ⚠️  UPDATE main.rs CONSTANTS:")
            print(f"      const CTC_CLASSES: usize = {vocab_size}; // or vocab_size + 1 depending on blank token")
            print(f"      const CTC_BLANK: u32 = {vocab_size - 1}; // Check config.json for exact blank_id")

    print("\n🎉 Download complete! Files are ready in models/granite-speech-5/")

if __name__ == "__main__":
    main()
