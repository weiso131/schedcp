use anyhow::{anyhow, Result};
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tokio::io::{AsyncBufReadExt, BufReader};

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

pub struct SchedulerManager {
    config: SchedulersConfig,
    pub temp_dir: Option<PathBuf>,
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
        let scheduler = self.get_scheduler(name)
            .ok_or_else(|| anyhow!("Scheduler '{}' not found", name))?;

        let binary_path = if let Some(ref temp_dir) = self.temp_dir {
            temp_dir.join(name)
        } else {
            return Err(anyhow!("Schedulers not extracted. Call extract_schedulers() first"));
        };

        if !binary_path.exists() {
            return Err(anyhow!("Scheduler binary '{}' not found", name));
        }

        println!("Running scheduler: {}", scheduler.name);
        println!("Description: {}", scheduler.description);
        println!("Production ready: {}", scheduler.production_ready);
        println!();

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
                while let Ok(Some(line)) = stdout_reader.next_line().await {
                    println!("{}", line);
                }
            } => {},
            _ = async {
                while let Ok(Some(line)) = stderr_reader.next_line().await {
                    if !line.starts_with("[sudo] password") {
                        eprintln!("{}", line);
                    }
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

    pub fn print_scheduler_info(&self, name: &str) -> Result<()> {
        let scheduler = self.get_scheduler(name)
            .ok_or_else(|| anyhow!("Scheduler '{}' not found", name))?;

        println!("Scheduler: {}", scheduler.name);
        println!("Production Ready: {}", scheduler.production_ready);
        println!("Description: {}", scheduler.description);
        println!("\nAlgorithm: {}", scheduler.algorithm);
        println!("Characteristics: {}", scheduler.characteristics);
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
        println!("\nLimitations: {}", scheduler.limitations);
        println!("Performance Profile: {}", scheduler.performance_profile);

        Ok(())
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