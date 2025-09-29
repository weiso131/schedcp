use tokio::net::UnixListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

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
            
            // Use the daemon library to run the command and get optimization suggestions
            let (_, response) = autotune::daemon::run_command_with_optimization(cmd, pwd);
            
            let _ = stream.write_all(response.as_bytes()).await;
        });
    }
}