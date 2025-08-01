use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command as TokioCommand};
use tokio::sync::mpsc;
use futures::stream::Stream;
use std::pin::Pin;
use crate::types::{ProcessError, ProcessInfo, ProcessStatus};
use uuid::Uuid;

pub type OutputLine = String;
pub type OutputStream = Pin<Box<dyn Stream<Item = OutputLine> + Send>>;

pub struct ProcessRunner {
    child: Child,
    info: ProcessInfo,
    output_rx: Option<mpsc::UnboundedReceiver<OutputLine>>,
}

impl ProcessRunner {
    pub async fn start(
        binary_path: &str,
        name: String,
        args: Vec<String>,
    ) -> Result<Self, ProcessError> {
        log::info!("Starting process: {} with args: {:?}", binary_path, args);
        
        let mut cmd = TokioCommand::new(binary_path);
        cmd.args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        
        let mut child = cmd.spawn()
            .map_err(|e| ProcessError::StartFailed(format!("Failed to spawn process: {}", e)))?;
        
        let pid = child.id();
        
        // Set up output streaming
        let (tx, rx) = mpsc::unbounded_channel();
        
        // Spawn task to read stdout
        if let Some(stdout) = child.stdout.take() {
            let tx_clone = tx.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) => break, // EOF
                        Ok(_) => {
                            if let Err(_) = tx_clone.send(line.trim().to_string()) {
                                break; // Receiver dropped
                            }
                        }
                        Err(e) => {
                            log::warn!("Error reading stdout: {}", e);
                            break;
                        }
                    }
                }
            });
        }
        
        // Spawn task to read stderr
        if let Some(stderr) = child.stderr.take() {
            let tx_clone = tx.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) => break, // EOF
                        Ok(_) => {
                            if let Err(_) = tx_clone.send(format!("[STDERR] {}", line.trim())) {
                                break; // Receiver dropped
                            }
                        }
                        Err(e) => {
                            log::warn!("Error reading stderr: {}", e);
                            break;
                        }
                    }
                }
            });
        }
        
        let info = ProcessInfo {
            id: Uuid::new_v4(),
            name,
            binary_name: std::path::Path::new(binary_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            pid,
            status: ProcessStatus::Running,
            args,
            started_at: Some(chrono::Utc::now()),
            stopped_at: None,
        };
        
        Ok(Self {
            child,
            info,
            output_rx: Some(rx),
        })
    }
    
    pub async fn start_with_sudo(
        binary_path: &str,
        name: String,
        args: Vec<String>,
        sudo_password: &str,
    ) -> Result<Self, ProcessError> {
        log::info!("Starting process with sudo: {} with args: {:?}", binary_path, args);
        
        let mut cmd = TokioCommand::new("sudo");
        cmd.arg("-S")
            .arg(binary_path)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        
        let mut child = cmd.spawn()
            .map_err(|e| ProcessError::StartFailed(format!("Failed to spawn process: {}", e)))?;
        
        // Send sudo password
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(format!("{}\n", sudo_password).as_bytes()).await
                .map_err(|e| ProcessError::StartFailed(format!("Failed to send sudo password: {}", e)))?;
            stdin.flush().await
                .map_err(|e| ProcessError::StartFailed(format!("Failed to flush stdin: {}", e)))?;
            // Don't put stdin back, sudo consumes it
        }
        
        let pid = child.id();
        
        // Set up output streaming
        let (tx, rx) = mpsc::unbounded_channel();
        
        // Spawn task to read stdout
        if let Some(stdout) = child.stdout.take() {
            let tx_clone = tx.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) => break, // EOF
                        Ok(_) => {
                            if let Err(_) = tx_clone.send(line.trim().to_string()) {
                                break; // Receiver dropped
                            }
                        }
                        Err(e) => {
                            log::warn!("Error reading stdout: {}", e);
                            break;
                        }
                    }
                }
            });
        }
        
        // Spawn task to read stderr
        if let Some(stderr) = child.stderr.take() {
            let tx_clone = tx.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) => break, // EOF
                        Ok(_) => {
                            if let Err(_) = tx_clone.send(format!("[STDERR] {}", line.trim())) {
                                break; // Receiver dropped
                            }
                        }
                        Err(e) => {
                            log::warn!("Error reading stderr: {}", e);
                            break;
                        }
                    }
                }
            });
        }
        
        let info = ProcessInfo {
            id: Uuid::new_v4(),
            name,
            binary_name: std::path::Path::new(binary_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            pid,
            status: ProcessStatus::Running,
            args,
            started_at: Some(chrono::Utc::now()),
            stopped_at: None,
        };
        
        Ok(Self {
            child,
            info,
            output_rx: Some(rx),
        })
    }
    
    pub async fn stop(&mut self) -> Result<(), ProcessError> {
        log::info!("Stopping process: {} (pid: {:?})", self.info.name, self.info.pid);
        
        self.child.kill().await
            .map_err(|e| ProcessError::StopFailed(format!("Failed to kill process: {}", e)))?;
        
        self.child.wait().await
            .map_err(|e| ProcessError::StopFailed(format!("Failed to wait for process: {}", e)))?;
        
        self.info.status = ProcessStatus::Stopped;
        self.info.stopped_at = Some(chrono::Utc::now());
        
        Ok(())
    }
    
    pub fn get_output_stream(&mut self) -> Option<OutputStream> {
        self.output_rx.take().map(|rx| {
            let stream = async_stream::stream! {
                let mut rx = rx;
                while let Some(line) = rx.recv().await {
                    yield line;
                }
            };
            Box::pin(stream) as OutputStream
        })
    }
    
    pub fn info(&self) -> &ProcessInfo {
        &self.info
    }
    
    pub fn id(&self) -> Uuid {
        self.info.id
    }
    
    pub fn is_running(&self) -> bool {
        self.info.status == ProcessStatus::Running
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    
    #[tokio::test]
    async fn test_process_runner_echo() {
        let mut runner = ProcessRunner::start(
            "/bin/echo",
            "test_echo".to_string(),
            vec!["Hello, World!".to_string()],
        ).await.unwrap();
        
        assert_eq!(runner.info().name, "test_echo");
        assert_eq!(runner.info().status, ProcessStatus::Running);
        assert!(runner.info().pid.is_some());
        
        // Collect output
        if let Some(mut stream) = runner.get_output_stream() {
            let output: Vec<String> = stream.collect().await;
            assert_eq!(output.len(), 1);
            assert_eq!(output[0], "Hello, World!");
        }
        
        // Process should complete naturally
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    
    #[tokio::test]
    async fn test_process_runner_sleep_and_kill() {
        let mut runner = ProcessRunner::start(
            "/bin/sleep",
            "test_sleep".to_string(),
            vec!["10".to_string()],
        ).await.unwrap();
        
        assert!(runner.is_running());
        
        // Give it a moment to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Stop the process
        runner.stop().await.unwrap();
        
        assert!(!runner.is_running());
        assert_eq!(runner.info().status, ProcessStatus::Stopped);
        assert!(runner.info().stopped_at.is_some());
    }
}