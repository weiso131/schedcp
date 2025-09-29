use std::env;
use tokio::net::UnixStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: autotune <command>");
        std::process::exit(1);
    }
    
    let command = args[1..].join(" ");
    let pwd = env::current_dir()?.to_string_lossy().into_owned();
    
    let mut stream = UnixStream::connect("/tmp/autotune.sock").await?;
    
    let request = format!("{}\n{}", pwd, command);
    stream.write_all(request.as_bytes()).await?;
    stream.shutdown().await?;
    
    let mut response = String::new();
    stream.read_to_string(&mut response).await?;
    
    println!("{}", response);
    Ok(())
}