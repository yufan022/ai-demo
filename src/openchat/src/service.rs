use std::fs::File;
use std::path::PathBuf;
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::{Tokenizer, tokenizer};
use candle_transformers::models::quantized_llama;

struct Args {
    model: String,
}

impl Args {
}

pub fn run() -> anyhow::Result<()> {
    // load model
    println!("load model");
    let mut model_path = File::open("./hf_hub/openchat_3.5.Q8_0.gguf")?;
    let model = gguf_file::Content::read(&mut model_path)?;

    // load tokenizer
    println!("load tokenizer");
    let tokenizer_path = PathBuf::from("./hf_hub/openchat_3.5_tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    // quantinized
    println!("quantinized");
    let mut model = quantized_llama::ModelWeights::from_gguf(model, &mut model_path, &Device::Cpu)?;

    // calculate
    println!("calculate");
    let tokens = tokenizer.encode("User: who are you? <|end_of_turn|> Assistant:", true).map_err(anyhow::Error::msg)?;
    let prompt_tokens = tokens.get_ids();
    println!("logits_processor");
    let mut logits_processor = LogitsProcessor::new(1, Some(0.8), None);
    println!("tensor");
    let input = Tensor::new(prompt_tokens, &Device::Cpu)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?.squeeze(0)?;
    println!("logits_processor sample");
    let next_token = logits_processor.sample(&logits)?;
    println!("decode");
    match tokenizer.decode(&[next_token], true) {
        Ok(str) => println!("decode: {}", str),
        Err(err) => println!("cannot decode: {err}"),
    };
    Ok(())
}
