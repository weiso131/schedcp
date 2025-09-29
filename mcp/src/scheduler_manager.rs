use anyhow::{anyhow, Result};
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::path::Path;
use tokio::sync::Mutex;
use process_manager::{
    BinaryExtractor, ProcessManager, ProcessConfig, ProcessStatus
};
use uuid::Uuid;
use super::scheduler_generator::SchedulerGenerator;

// Only embed the JSON configuration, not the binaries
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
    generator: Arc<SchedulerGenerator>,
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

        // Create binary extractor that loads from external directory
        let mut extractor = BinaryExtractor::new()
            .map_err(|e| anyhow!("Failed to create binary extractor: {}", e))?;

        // Get the scheduler binary directory
        let home_dir = std::env::var("HOME")
            .map_err(|_| anyhow!("HOME environment variable not set"))?;
        let scx_bin_dir = Path::new(&home_dir).join(".schedcp").join("scxbin");
        
        // Check if directory exists
        if !scx_bin_dir.exists() {
            return Err(anyhow!(
                "Scheduler binary directory not found at {}. Please run 'make install' in the scheduler directory first.",
                scx_bin_dir.display()
            ));
        }
        
        // Load binaries from external directory
        let entries = std::fs::read_dir(&scx_bin_dir)
            .map_err(|e| anyhow!("Failed to read directory {}: {}", scx_bin_dir.display(), e))?;
        
        let mut binary_count = 0;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();
            
            // Only include files that start with "scx_" and are not hash-suffixed versions
            if file_name.starts_with("scx_") && !file_name.contains('-') && path.is_file() {
                // Read the binary data
                let binary_data = std::fs::read(&path)
                    .map_err(|e| anyhow!("Failed to read {}: {}", path.display(), e))?;
                
                extractor.add_binary(&file_name, &binary_data)
                    .map_err(|e| anyhow!("Failed to add {}: {}", file_name, e))?;
                
                log::info!("Loaded scheduler binary: {} from {}", file_name, path.display());
                binary_count += 1;
            }
        }
        
        if binary_count == 0 {
            return Err(anyhow!(
                "No scheduler binaries found in {}. Please run 'make install' in the scheduler directory.",
                scx_bin_dir.display()
            ));
        }
        
        log::info!("Loaded {} scheduler binaries from {}", binary_count, scx_bin_dir.display());

        let process_manager = Arc::new(ProcessManager::new(extractor));

        // Initialize scheduler generator
        let generator = SchedulerGenerator::new()
            .map_err(|e| anyhow!("Failed to create scheduler generator: {}", e))?;

        Ok(Self {
            config,
            process_manager,
            executions: Arc::new(Mutex::new(HashMap::new())),
            sudo_password: None,
            generator: Arc::new(generator),
        })
    }

    #[allow(dead_code)]
    pub fn set_sudo_password(&mut self, password: String) {
        self.sudo_password = Some(password);
    }

    pub fn list_schedulers(&self) -> &[SchedulerInfo] {
        &self.config.schedulers
    }

    pub fn get_scheduler(&self, name: &str) -> Option<&SchedulerInfo> {
        self.config.schedulers.iter().find(|s| s.name == name)
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub async fn stop_scheduler(&self, execution_id: &str) -> Result<()> {
        let executions = self.executions.lock().await;
        if let Some(exec) = executions.get(execution_id) {
            self.process_manager.stop_process(exec.process_id).await
                .map_err(|e| anyhow!("Failed to stop scheduler: {}", e))
        } else {
            Err(anyhow!("Execution not found"))
        }
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn available_schedulers(&self) -> Vec<String> {
        self.process_manager.available_binaries()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    // Compatibility methods for CLI
    #[allow(dead_code)]
    pub async fn extract_schedulers(&mut self) -> Result<()> {
        // Already extracted in new(), but we can verify
        Ok(())
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    /// Create and verify a custom scheduler
    ///
    /// This method creates a custom BPF scheduler from source code, compiles it,
    /// and verifies it can load and run in the kernel.
    ///
    /// # Arguments
    /// * `info` - Scheduler metadata (name, description, etc.)
    /// * `source_code` - The BPF C source code
    ///
    /// # Returns
    /// Ok(String) with verification details if successful
    pub async fn create_and_verify_scheduler(
        &self,
        info: SchedulerInfo,
        source_code: &str,
    ) -> Result<String> {
        let name = &info.name;

        log::info!("Creating custom scheduler: {}", name);

        // Step 1: Create the scheduler source file
        self.generator.create_scheduler(name, source_code)
            .map_err(|e| anyhow!("Failed to create scheduler source: {}", e))?;

        log::info!("Compiling custom scheduler: {}", name);

        // Step 2: Compile the scheduler
        self.generator.compile_scheduler(name)
            .map_err(|e| anyhow!("Failed to compile scheduler: {}", e))?;

        log::info!("Verifying custom scheduler: {}", name);

        // Step 3: Verify the scheduler can load and run
        let verification_result = self.generator.execute_scheduler(name, None).await
            .map_err(|e| anyhow!("Failed to verify scheduler: {}", e))?;

        log::info!("Successfully created and verified scheduler: {}", name);

        Ok(format!(
            "Custom scheduler '{}' created successfully!\n\nDescription: {}\nAlgorithm: {}\n\n{}",
            name,
            info.description,
            info.algorithm,
            verification_result
        ))
    }

    /// Get the scheduler generator for direct access to custom scheduler management
    pub fn generator(&self) -> &SchedulerGenerator {
        &self.generator
    }

    /// Run a custom generated scheduler persistently
    ///
    /// Unlike execute_scheduler which runs for a fixed duration, this method
    /// starts the scheduler and returns an execution_id for later management.
    ///
    /// # Arguments
    /// * `name` - Name of the custom scheduler
    /// * `args` - Command-line arguments for the scheduler
    ///
    /// # Returns
    /// Ok(execution_id) that can be used with get_execution and stop_scheduler
    pub async fn run_custom_scheduler(&self, name: &str, args: Vec<String>) -> Result<String> {
        // Check if the custom scheduler exists and is compiled
        let object_path = self.generator.get_scheduler_object_path(name)
            .ok_or_else(|| anyhow!("Custom scheduler '{}' not found or not compiled. Use create_and_verify_scheduler first.", name))?;

        // Generate execution ID
        let execution_id = format!("custom_{}", uuid::Uuid::new_v4().to_string()[..8].to_string());

        // Get the loader path from the generator's work directory
        let loader_path = self.generator.work_dir().join("loader");
        if !loader_path.exists() {
            return Err(anyhow!(
                "Loader binary not found at {}. Run 'make' in the new_sched directory first.",
                loader_path.display()
            ));
        }

        // Prepare the command to run the custom scheduler
        // The loader binary takes the .bpf.o file as an argument
        let mut full_args = vec![
            loader_path.to_string_lossy().to_string(),
            object_path.to_string_lossy().to_string(),
        ];
        full_args.extend(args.clone());

        // Create process config for sudo execution
        let config = ProcessConfig {
            name: format!("custom_{}_exec_{}", name, execution_id),
            binary_name: "sudo".to_string(), // We'll run sudo directly
            args: full_args,
            env: HashMap::new(),
            working_dir: Some(self.generator.work_dir().to_string_lossy().to_string()),
        };

        // Start the process with sudo
        let sudo_password = self.sudo_password.as_ref()
            .map(|s| s.as_str())
            .unwrap_or("");

        let process_id = self.process_manager.start_process_with_sudo(config, sudo_password).await
            .map_err(|e| anyhow!("Failed to start custom scheduler with sudo: {}", e))?;

        // Create execution record
        let execution = SchedulerExecution {
            execution_id: execution_id.clone(),
            process_id,
            scheduler_name: format!("custom/{}", name),
            command: loader_path.to_string_lossy().to_string(),
            args: vec![object_path.to_string_lossy().to_string()].into_iter().chain(args).collect(),
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

        log::info!("Started custom scheduler '{}' with execution_id: {}", name, execution_id);

        Ok(execution_id)
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