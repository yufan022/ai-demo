use std::fs::File;
use std::path::PathBuf;
use candle_core::quantized::gguf_file;
use tokenizers::{Tokenizer, tokenizer};

struct Args {
    model: String,
}

impl Args {
}

pub fn run() -> anyhow::Result<()> {
    let mut model_path = File::open("./hf_hub/openchat_3.5.Q8_0.gguf")?;
    let model = gguf_file::Content::read(&mut model_path)?;

    let tokenizer_path = PathBuf::from("./hf_hub/openchat_3.5_tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    Ok(())
}
