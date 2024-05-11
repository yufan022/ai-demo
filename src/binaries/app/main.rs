use openchat::service::run;
use qwen::add;

fn main() {
    // qwen
    let result = add(1, 1);
    println!("Hello, world! {}", result);

    // openchat
    match run() {
        Ok(_) => {
            println!("success");
        }
        Err(e) => {
            println!("failed {}", e);
        }
    };
}
