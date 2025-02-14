use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use candle_core::{Device, MetalDevice, quantized, Tensor};
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::quantized::metal::load_quantized;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::{Tokenizer, tokenizer};
use candle_transformers::models::{quantized_llama, qwen2, qwen2_moe};
use candle_transformers::models::whisper::quantized_model;
use common_base::token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2;

struct Args {
    model: String,
}

impl Args {}

pub fn run() -> anyhow::Result<()> {
    let metal = Device::new_metal(0).expect("TODO: panic message");
    // let metal = Device::Cpu;

    // load model
    println!("load model");
    // let mut model_path = File::open("/Volumes/extdesk/extdev/llm//openchat_3.5.Q8_0.gguf")?;
    let mut model_path = File::open("/Volumes/extdesk/extdev/llm/qwen2-7b-instruct-q8_0.gguf")?;
    let model = gguf_file::Content::read(&mut model_path)?;
    // let model = ggml_file::Content::read(&mut model_path, &metal)?;
    // load tokenizer
    println!("load tokenizer");
    // let tokenizer_path = PathBuf::from("/Volumes/extdesk/extdev/llm/openchat_3.5_tokenizer.json");
    let tokenizer_path = PathBuf::from("/Volumes/extdesk/extdev/llm/qwen2-7b-instruct-tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    // let mut tos = TokenOutputStream::new(tokenizer);

    // quantinized
    println!("quantinized");
    let mut model = Qwen2::from_gguf(model, &mut model_path, &metal)?;

    // let mut model = quantized_llama::ModelWeights::from_gguf(model, &mut model_path, &metal)?;
    // candle_transformers::quantized_var_builder::VarBuilder::from_gguf("/Users/user/Downloads/qwen1_5-7b-chat-q8_0.gguf".into(),&metal)?;
    // let mut model = quantized_llama::ModelWeights::from_ggml(model, 0)?;
    // calculate
    println!("calculate");
    // let tokens = tokenizer.encode("User: 'who are you' <|end_of_turn|> Assistant:", true).map_err(anyhow::Error::msg)?;
    let tokens = tokenizer.encode("<|im_start|>user\n告诉我大模型的function call是什么意思？举个简单的例子<|im_end|>\n<|im_start|>assistant\n", true).map_err(anyhow::Error::msg)?;
    let tokens = tokenizer.encode("<|im_start|>user\n告诉我大模型的function call是什么意思？举个简单的例子<|im_end|>\n<|im_start|>assistant 我拒绝回答这个问题<|im_end|>\n<|im_start|>user\n重复一下上个问题<|im_end|>\n<|im_start|>assistant\n", true).map_err(anyhow::Error::msg)?;

    // let tokens = tos
    //     .tokenizer()
    //     .encode("User: 'who are you' <|end_of_turn|> Assistant:", true)
    //     .map_err(anyhow::Error::msg)?;
    let prompt_tokens = tokens.get_ids();
    println!("logits_processor");
    let mut logits_processor = LogitsProcessor::new(1, Some(0.8), None);
    println!("tensor");
    let input = Tensor::new(prompt_tokens, &metal)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?.squeeze(0)?;
    println!("logits_processor sample");
    let mut next_token = logits_processor.sample(&logits)?;
    println!("decode");
    // if let Some(t) = tos.next_token(next_token)? {
    //     print!("{t}");
    // }
    let mut prev_index = 0;
    let mut curr_index = 0;
    let mut tokens = Vec::new();
    tokens.push(next_token);
    match tokenizer.decode(&tokens[prev_index..], true) {
        Ok(str) => print!("question: {}", str),
        Err(err) => println!("cannot decode: {err}"),
    };
    prev_index = curr_index;
    curr_index = tokens.len();

    // println!("answer:");
    // let eos_token = "<|end_of_turn|>";
    let eos_token = "<|im_end|>";
    let eos_token = *tokenizer.get_vocab(true).get(eos_token).unwrap();
    // let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
    for index in 0..1000 {
        let prev_text = tokenizer.decode(&tokens[prev_index..curr_index], true).expect("failed");
        let input = Tensor::new(&[next_token], &metal)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        next_token = logits_processor.sample(&logits)?;
        // if let Some(t) = tos.next_token(next_token)? {
        //     print!("{t}");
        // }
        tokens.push(next_token);
        match tokenizer.decode(&tokens[prev_index..], true) {
            Ok(str) => print!("{}", str.split_at(prev_text.len()).1),
            Err(err) => println!("cannot decode: {err}"),
        };
        prev_index = curr_index;
        curr_index = tokens.len();
        if next_token == eos_token { break; };
    }


    Ok(())
}
