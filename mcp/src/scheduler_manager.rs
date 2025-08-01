use anyhow::{anyhow, Result};
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;
use dashmap::DashMap;
use uuid::Uuid;

#[derive(RustEmbed)]
#[folder = "../scheduler/sche_bin/"]
pub struct SchedulerBinaries;

#[derive(RustEmbed)]
#[folder = "../scheduler/"]
#[include = "schedulers.json"]
pub struct SchedulerConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerInfo {
    pub name: String,
    pub production_ready: bool,
    pub description: String,
    pub use_cases: Vec<String>,
    pub algorithm: String,
    pub characteristics: String,
    pub tuning_parameters: HashMap<String, ParameterInfo>,
    pub limitations: String,
    pub performance_profile: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    #[serde(rename = "type")]
    pub param_type: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulersConfig {
    pub schedulers: Vec<SchedulerInfo>,
}

#[derive(Debug, Clone)]
pub struct SchedulerExecution {
    pub execution_id: String,
    pub scheduler_name: String,
    pub status: Arc<Mutex<String>>,
    pub pid: Arc<Mutex<Option<u32>>>,
    pub start_time: u64,
    pub end_time: Arc<Mutex<Option<u64>>>,
    pub exit_code: Arc<Mutex<Option<i32>>>,
    pub output_buffer: Arc<Mutex<Vec<String>>>,
}

impl SchedulerExecution {
    pub fn new(execution_id: String, scheduler_name: String) -> Self {
        Self {
            execution_id,
            scheduler_name,
            status: Arc::new(Mutex::new("running".to_string())),
            pid: Arc::new(Mutex::new(None)),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            end_time: Arc::new(Mutex::new(None)),
            exit_code: Arc::new(Mutex::new(None)),
            output_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn set_pid(&self, pid: u32) {
        *self.pid.lock().await = Some(pid);
    }

    pub async fn mark_completed(&self, exit_code: i32) {
        *self.status.lock().await = "completed".to_string();
        *self.end_time.lock().await = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
        *self.exit_code.lock().await = Some(exit_code);
    }

    pub async fn mark_failed(&self, reason: &str) {
        *self.status.lock().await = format!("failed: {}", reason);
        *self.end_time.lock().await = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
    }
}

pub struct SchedulerManager {
    config: SchedulersConfig,
    pub temp_dir: Option<PathBuf>,
    pub executions: Arc<DashMap<String, SchedulerExecution>>,
}

impl SchedulerManager {
    pub fn new() -> Result<Self> {
        // Load schedulers.json from embedded resources
        let config_data = SchedulerConfig::get("schedulers.json")
            .ok_or_else(|| anyhow!("schedulers.json not found in embedded resources"))?;
        let config_str = std::str::from_utf8(config_data.data.as_ref())
            .map_err(|e| anyhow!("Failed to decode schedulers.json: {}", e))?;
        let config: SchedulersConfig = serde_json::from_str(config_str)
            .map_err(|e| anyhow!("Failed to parse schedulers.json: {}", e))?;

        Ok(Self {
            config,
            temp_dir: None,
            executions: Arc::new(DashMap::new()),
        })
    }

    pub fn list_schedulers(&self) -> &[SchedulerInfo] {
        &self.config.schedulers
    }

    pub fn get_scheduler(&self, name: &str) -> Option<&SchedulerInfo> {
        self.config.schedulers.iter().find(|s| s.name == name)
    }

    pub async fn extract_schedulers(&mut self) -> Result<PathBuf> {
        // Create temporary directory
        let temp_dir = std::env::temp_dir().join(format!("schedcp_{}", std::process::id()));
        fs::create_dir_all(&temp_dir)?;
        
        // Extract all scheduler binaries
        for file in SchedulerBinaries::iter() {
            let file_name = file.as_ref();
            // Only extract main binaries (skip hash-suffixed versions)
            if file_name.starts_with("scx_") && !file_name.contains('-') {
                if let Some(data) = SchedulerBinaries::get(&file) {
                    let target_path = temp_dir.join(file_name);
                    fs::write(&target_path, data.data)?;
                    
                    // Make executable
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        let mut perms = fs::metadata(&target_path)?.permissions();
                        perms.set_mode(0o755);
                        fs::set_permissions(&target_path, perms)?;
                    }
                }
            }
        }

        self.temp_dir = Some(temp_dir.clone());
        Ok(temp_dir)
    }

    pub async fn run_scheduler(
        &self,
        name: &str,
        args: Vec<String>,
        sudo_password: Option<&str>,
    ) -> Result<()> {
        let _scheduler = self.get_scheduler(name)
            .ok_or_else(|| anyhow!("Scheduler '{}' not found", name))?;

        let binary_path = if let Some(ref temp_dir) = self.temp_dir {
            temp_dir.join(name)
        } else {
            return Err(anyhow!("Schedulers not extracted. Call extract_schedulers() first"));
        };

        if !binary_path.exists() {
            return Err(anyhow!("Scheduler binary '{}' not found", name));
        }

        // Don't print to stdout as it interferes with MCP protocol

        let mut cmd = if sudo_password.is_some() {
            let mut cmd = Command::new("sudo");
            cmd.arg("-S").arg(&binary_path);
            cmd
        } else {
            Command::new(&binary_path)
        };

        // Add user-provided arguments
        for arg in args {
            cmd.arg(arg);
        }

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn()?;

        // Send sudo password if provided
        if let Some(password) = sudo_password {
            if let Some(mut stdin) = child.stdin.take() {
                use tokio::io::AsyncWriteExt;
                stdin.write_all(format!("{}\n", password).as_bytes()).await?;
                stdin.flush().await?;
            }
        }

        // Stream output
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();

        tokio::select! {
            _ = async {
                while let Ok(Some(_line)) = stdout_reader.next_line().await {
                    // Don't print to stdout as it interferes with MCP protocol
                    // Line is consumed but not printed
                }
            } => {},
            _ = async {
                while let Ok(Some(_line)) = stderr_reader.next_line().await {
                    // Don't print to stderr as it interferes with MCP protocol
                    // Line is consumed but not printed
                }
            } => {},
            status = child.wait() => {
                match status {
                    Ok(status) => {
                        if !status.success() {
                            return Err(anyhow!("Scheduler exited with status: {}", status));
                        }
                    }
                    Err(e) => return Err(anyhow!("Failed to wait for scheduler: {}", e)),
                }
            }
        }

        Ok(())
    }

    pub async fn stop_running_schedulers(&self, sudo_password: &str) -> Vec<String> {
        let mut stopped = Vec::new();
        let executions_to_stop: Vec<_> = self.executions.iter()
            .filter(|entry| {
                if let Ok(status) = entry.value().status.try_lock() {
                    *status == "running"
                } else {
                    false
                }
            })
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        for (exec_id, execution) in executions_to_stop {
            if let Ok(pid) = execution.pid.try_lock() {
                if let Some(pid) = *pid {
                    // Try to stop the scheduler
                    let mut cmd = Command::new("sudo");
                    cmd.arg("-S")
                        .arg("kill")
                        .arg("-TERM")
                        .arg(pid.to_string())
                        .stdin(std::process::Stdio::piped());

                    if let Ok(mut child) = cmd.spawn() {
                        // Send sudo password
                        if let Some(mut stdin) = child.stdin.take() {
                            use tokio::io::AsyncWriteExt;
                            let _ = stdin.write_all(format!("{}\n", sudo_password).as_bytes()).await;
                        }
                        let _ = child.wait().await;
                        stopped.push(exec_id);
                    }
                }
            }
        }
        stopped
    }

    pub fn create_execution(&self, name: &str) -> Result<String> {
        // Check if scheduler exists
        if self.get_scheduler(name).is_none() {
            return Err(anyhow!("Scheduler '{}' not found", name));
        }

        // Generate execution ID
        let execution_id = format!("sched_{}", Uuid::new_v4().to_string()[..8].to_string());
        
        // Create execution record
        let execution = SchedulerExecution::new(execution_id.clone(), name.to_string());
        self.executions.insert(execution_id.clone(), execution);
        
        Ok(execution_id)
    }

    pub fn get_execution(&self, execution_id: &str) -> Option<SchedulerExecution> {
        self.executions.get(execution_id).map(|e| e.value().clone())
    }

    pub async fn run_scheduler_background(
        &self,
        execution_id: String,
        args: Vec<String>,
        sudo_password: String,
    ) -> Result<()> {
        let execution = self.executions.get(&execution_id)
            .ok_or_else(|| anyhow!("Execution not found"))?
            .clone();

        let scheduler_name = execution.scheduler_name.clone();
        
        // Get scheduler binary path
        let binary_path = if let Some(ref temp_dir) = self.temp_dir {
            temp_dir.join(&scheduler_name)
        } else {
            execution.mark_failed("Scheduler binaries not extracted").await;
            return Err(anyhow!("Scheduler binaries not extracted"));
        };

        let mut cmd = Command::new("sudo");
        cmd.arg("-S")
            .arg(&binary_path)
            .args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(e) => {
                execution.mark_failed(&format!("Failed to spawn scheduler: {}", e)).await;
                return Err(anyhow!("Failed to spawn scheduler: {}", e));
            }
        };

        // Set PID
        if let Some(pid) = child.id() {
            execution.set_pid(pid).await;
        }

        // Send sudo password
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            let _ = stdin.write_all(format!("{}\n", sudo_password).as_bytes()).await;
            let _ = stdin.flush().await;
        }

        // Capture output streams
        if let Some(stdout) = child.stdout.take() {
            let output_buffer = execution.output_buffer.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let mut buffer = output_buffer.lock().await;
                    buffer.push(format!("[stdout] {}", line));
                    // Keep only last 1000 lines to prevent memory bloat
                    if buffer.len() > 1000 {
                        buffer.drain(0..500);
                    }
                }
            });
        }

        if let Some(stderr) = child.stderr.take() {
            let output_buffer = execution.output_buffer.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let mut buffer = output_buffer.lock().await;
                    buffer.push(format!("[stderr] {}", line));
                    // Keep only last 1000 lines to prevent memory bloat
                    if buffer.len() > 1000 {
                        buffer.drain(0..500);
                    }
                }
            });
        }

        // Wait for process to complete
        match child.wait().await {
            Ok(status) => {
                let exit_code = status.code().unwrap_or(-1);
                execution.mark_completed(exit_code).await;
            }
            Err(e) => {
                execution.mark_failed(&format!("Wait error: {}", e)).await;
            }
        }

        Ok(())
    }

    pub fn print_scheduler_info(&self, scheduler: &SchedulerInfo) {
        println!("================================================================================");
        println!("Scheduler: {}", scheduler.name);
        println!("Production Ready: {}", scheduler.production_ready);
        println!("Algorithm: {}", scheduler.algorithm);
        println!("\nDescription:");
        println!("{}", scheduler.description);
        println!("\nCharacteristics:");
        println!("{}", scheduler.characteristics);
        println!("\nUse Cases:");
        for use_case in &scheduler.use_cases {
            println!("  - {}", use_case);
        }
        println!("\nTuning Parameters:");
        for (param_name, param_info) in &scheduler.tuning_parameters {
            println!("  --{}: {}", param_name.replace('_', "-"), param_info.description);
            if let Some(default) = &param_info.default {
                println!("    Default: {}", default);
            }
            if let Some(range) = &param_info.range {
                println!("    Range: {:?}", range);
            }
            println!("    Type: {}", param_info.param_type);
        }
        if !scheduler.limitations.is_empty() {
            println!("\nLimitations: {}", scheduler.limitations);
        }
        println!("Performance Profile: {}", scheduler.performance_profile);
        println!();
    }
}

impl Drop for SchedulerManager {
    fn drop(&mut self) {
        // Clean up temporary directory
        if let Some(ref temp_dir) = self.temp_dir {
            let _ = fs::remove_dir_all(temp_dir);
        }
    }
}