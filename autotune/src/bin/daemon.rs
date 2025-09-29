use tokio::net::UnixListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::process::Command;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let socket_path = "/tmp/autotune.sock";
    let _ = std::fs::remove_file(socket_path);
    
    let listener = UnixListener::bind(socket_path)?;
    println!("Autotune daemon listening on {}", socket_path);
    
    loop {
        let (mut stream, _) = listener.accept().await?;
        
        tokio::spawn(async move {
            let mut buf = vec![0; 4096];
            let n = stream.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..n]);
            
            let parts: Vec<&str> = request.trim().splitn(2, '\n').collect();
            if parts.len() != 2 {
                let _ = stream.write_all(b"ERROR: Invalid request").await;
                return;
            }
            
            let pwd = parts[0];
            let cmd = parts[1];
            
            println!("Running: {} in {}", cmd, pwd);
            
            // Run the command and measure time
            let start = Instant::now();
            let output = Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .current_dir(pwd)
                .output()
                .expect("Failed to execute command");
            let duration = start.elapsed();
            
            // Prepare prompt for Claude
            let prompt = format!(
                "The command '{}' took {:?} to execute with exit code {}.\n\
                stdout: {}\n\
                stderr: {}\n\
                Please suggest optimizations for this command.",
                cmd, duration, output.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            
            // Call Claude CLI
            let claude_output = Command::new("claude")
                .arg("-q")
                .arg(&prompt)
                .output()
                .expect("Failed to run claude");
            
            let response = if claude_output.status.success() {
                String::from_utf8_lossy(&claude_output.stdout).into_owned()
            } else {
                format!("Claude error: {}", String::from_utf8_lossy(&claude_output.stderr))
            };
            
            let _ = stream.write_all(response.as_bytes()).await;
        });
    }
}