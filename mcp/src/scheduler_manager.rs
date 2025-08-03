use anyhow::{anyhow, Result};
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use process_manager::{
    BinaryExtractor, ProcessManager, ProcessConfig, ProcessStatus
};
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
    pub process_id: Uuid,
    pub scheduler_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub start_time: u64,
    pub output_buffer: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone)]
pub struct SchedulerManager {
    config: SchedulersConfig,
    process_manager: Arc<ProcessManager>,
    // Map execution_id to process_id
    executions: Arc<Mutex<HashMap<String, SchedulerExecution>>>,
    sudo_password: Option<String>,
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

        // Create binary extractor
        let mut extractor = BinaryExtractor::new()
            .map_err(|e| anyhow!("Failed to create binary extractor: {}", e))?;

        // Extract all scheduler binaries
        for file in SchedulerBinaries::iter() {
            let file_name = file.as_ref();
            // Only extract main binaries (skip hash-suffixed versions)
            if file_name.starts_with("scx_") && !file_name.contains('-') {
                if let Some(data) = SchedulerBinaries::get(&file) {
                    extractor.add_binary(file_name, &data.data)
                        .map_err(|e| anyhow!("Failed to extract {}: {}", file_name, e))?;
                }
            }
        }

        let process_manager = Arc::new(ProcessManager::new(extractor));

        Ok(Self {
            config,
            process_manager,
            executions: Arc::new(Mutex::new(HashMap::new())),
            sudo_password: None,
        })
    }

    pub fn set_sudo_password(&mut self, password: String) {
        self.sudo_password = Some(password);
    }

    pub fn list_schedulers(&self) -> &[SchedulerInfo] {
        &self.config.schedulers
    }

    pub fn get_scheduler(&self, name: &str) -> Option<&SchedulerInfo> {
        self.config.schedulers.iter().find(|s| s.name == name)
    }

    pub async fn stop_running_schedulers(&self) -> Vec<String> {
        let mut stopped = Vec::new();
        let running_processes = self.process_manager.get_running_processes();
        
        for process in running_processes {
            if let Err(e) = self.process_manager.stop_process(process.id).await {
                log::warn!("Failed to stop process {}: {}", process.id, e);
            } else {
                // Find the execution_id for this process
                let executions = self.executions.lock().await;
                for (exec_id, exec) in executions.iter() {
                    if exec.process_id == process.id {
                        stopped.push(exec_id.clone());
                        break;
                    }
                }
            }
        }
        
        stopped
    }

    pub async fn create_execution(&self, name: &str, args: Vec<String>) -> Result<String> {
        // Check if scheduler exists
        if self.get_scheduler(name).is_none() {
            return Err(anyhow!("Scheduler '{}' not found", name));
        }

        // Generate execution ID
        let execution_id = format!("sched_{}", uuid::Uuid::new_v4().to_string()[..8].to_string());
        
        // Create process config
        let config = ProcessConfig {
            name: format!("{}_exec_{}", name, execution_id),
            binary_name: name.to_string(),
            args: args.clone(),
            env: HashMap::new(),
            working_dir: None,
        };

        // Always start schedulers with sudo (empty password means passwordless sudo)
        let sudo_password = self.sudo_password.as_ref()
            .map(|s| s.as_str())
            .unwrap_or("");
        
        let process_id = self.process_manager.start_process_with_sudo(config, sudo_password).await
            .map_err(|e| anyhow!("Failed to start scheduler with sudo: {}", e))?;

        // Create execution record
        let execution = SchedulerExecution {
            execution_id: execution_id.clone(),
            process_id,
            scheduler_name: name.to_string(),
            command: name.to_string(),
            args,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            output_buffer: Arc::new(Mutex::new(Vec::new())),
        };

        // Start output capture task
        let output_buffer = execution.output_buffer.clone();
        let process_manager = self.process_manager.clone();
        let process_id_clone = process_id;
        
        tokio::spawn(async move {
            if let Some(mut stream) = process_manager.get_output_stream(process_id_clone) {
                use futures::StreamExt;
                while let Some(line) = stream.next().await {
                    let mut buffer = output_buffer.lock().await;
                    buffer.push(line);
                    // Keep only last 1000 lines to prevent memory bloat
                    if buffer.len() > 1000 {
                        buffer.drain(0..500);
                    }
                }
            }
        });

        // Store execution
        let mut executions = self.executions.lock().await;
        executions.insert(execution_id.clone(), execution);
        
        Ok(execution_id)
    }

    pub async fn get_execution(&self, execution_id: &str) -> Option<SchedulerExecutionStatus> {
        let executions = self.executions.lock().await;
        if let Some(exec) = executions.get(execution_id) {
            // Get process info from process manager
            if let Some(process_info) = self.process_manager.get_process_info(exec.process_id) {
                let output = exec.output_buffer.lock().await.clone();
                
                Some(SchedulerExecutionStatus {
                    execution_id: exec.execution_id.clone(),
                    scheduler_name: exec.scheduler_name.clone(),
                    command: exec.command.clone(),
                    args: exec.args.clone(),
                    status: match process_info.status {
                        ProcessStatus::Pending => "pending",
                        ProcessStatus::Running => "running",
                        ProcessStatus::Stopped => "stopped",
                        ProcessStatus::Failed => "failed",
                    }.to_string(),
                    pid: process_info.pid,
                    start_time: exec.start_time,
                    end_time: process_info.stopped_at.map(|t| t.timestamp() as u64),
                    output,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    pub async fn stop_scheduler(&self, execution_id: &str) -> Result<()> {
        let executions = self.executions.lock().await;
        if let Some(exec) = executions.get(execution_id) {
            self.process_manager.stop_process(exec.process_id).await
                .map_err(|e| anyhow!("Failed to stop scheduler: {}", e))
        } else {
            Err(anyhow!("Execution not found"))
        }
    }

    pub async fn cleanup_old_executions(&self, max_age_secs: u64) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut executions = self.executions.lock().await;
        let mut to_remove = Vec::new();

        for (exec_id, exec) in executions.iter() {
            if let Some(process_info) = self.process_manager.get_process_info(exec.process_id) {
                if process_info.status != ProcessStatus::Running 
                    && current_time - exec.start_time > max_age_secs {
                    to_remove.push(exec_id.clone());
                }
            }
        }

        for exec_id in to_remove {
            executions.remove(&exec_id);
        }
    }

    pub fn available_schedulers(&self) -> Vec<String> {
        self.process_manager.available_binaries()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    // Compatibility methods for CLI
    pub async fn extract_schedulers(&mut self) -> Result<()> {
        // Already extracted in new(), but we can verify
        Ok(())
    }

    pub async fn run_scheduler(
        &self,
        name: &str,
        args: Vec<String>,
        sudo_password: Option<&str>,
    ) -> Result<()> {
        // Set sudo password if provided
        let mut manager = self.clone();
        if let Some(password) = sudo_password {
            manager.sudo_password = Some(password.to_string());
        }

        // Create and start execution
        let execution_id = manager.create_execution(name, args).await?;
        
        // Wait for the process to complete
        loop {
            if let Some(status) = manager.get_execution(&execution_id).await {
                if status.status != "running" && status.status != "pending" {
                    break;
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
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

#[derive(Debug, Clone, Serialize)]
pub struct SchedulerExecutionStatus {
    pub execution_id: String,
    pub scheduler_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub status: String,
    pub pid: Option<u32>,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub output: Vec<String>,
}