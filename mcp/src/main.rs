use anyhow::Result;
use dashmap::DashMap;
use rmcp::{
    handler::server::{router::tool::ToolRouter, tool::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
    ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::HashMap,
    future::Future,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    process::Command,
    sync::Mutex,
    time::sleep,
};
use tracing::info;
use uuid::Uuid;

mod scheduler_manager;
use scheduler_manager::{SchedulerManager, ParameterInfo};

type McpError = rmcp::model::ErrorData;

#[derive(Debug, Clone)]
struct SchedulerExecution {
    execution_id: String,
    scheduler_name: String,
    status: Arc<Mutex<String>>,
    pid: Arc<Mutex<Option<u32>>>,
    start_time: u64,
    end_time: Arc<Mutex<Option<u64>>>,
    exit_code: Arc<Mutex<Option<i32>>>,
}

impl SchedulerExecution {
    fn new(execution_id: String, scheduler_name: String) -> Self {
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
        }
    }

    async fn set_pid(&self, pid: u32) {
        *self.pid.lock().await = Some(pid);
    }

    async fn mark_completed(&self, exit_code: i32) {
        *self.status.lock().await = "completed".to_string();
        *self.end_time.lock().await = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
        *self.exit_code.lock().await = Some(exit_code);
    }

    async fn mark_failed(&self, reason: &str) {
        *self.status.lock().await = format!("failed: {}", reason);
        *self.end_time.lock().await = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
    }
}

#[derive(Clone)]
struct SchedMcpServer {
    tool_router: ToolRouter<Self>,
    sudo_password: Arc<String>,
    scheduler_manager: Arc<Mutex<SchedulerManager>>,
    executions: Arc<DashMap<String, SchedulerExecution>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ListSchedulersRequest {
    #[schemars(description = "Filter by scheduler name (partial match)")]
    name: Option<String>,
    #[schemars(description = "Filter by production readiness")]
    production_ready: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ListSchedulersResponse {
    schedulers: Vec<SchedulerSummary>,
}

#[derive(Debug, Serialize)]
struct SchedulerSummary {
    name: String,
    production_ready: bool,
    description: String,
    algorithm: String,
    use_cases: Vec<String>,
    characteristics: String,
    tuning_parameters: HashMap<String, ParameterInfo>,
    limitations: String,
    performance_profile: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RunSchedulerRequest {
    #[schemars(description = "Name of the scheduler to run")]
    name: String,
    #[schemars(description = "Arguments to pass to the scheduler")]
    #[serde(default)]
    args: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RunSchedulerResponse {
    execution_id: String,
    scheduler: String,
    status: String,
    message: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct StopSchedulerRequest {
    #[schemars(description = "Execution ID of the running scheduler")]
    execution_id: String,
}

#[derive(Debug, Serialize)]
struct StopSchedulerResponse {
    execution_id: String,
    status: String,
    message: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GetExecutionStatusRequest {
    #[schemars(description = "Execution ID to query")]
    execution_id: String,
}

#[derive(Debug, Serialize)]
struct GetExecutionStatusResponse {
    execution_id: String,
    scheduler_name: String,
    status: String,
    pid: Option<u32>,
    start_time: u64,
    end_time: Option<u64>,
    duration: Option<u64>,
    exit_code: Option<i32>,
}

impl SchedMcpServer {
    async fn run_scheduler_background(
        execution: SchedulerExecution,
        manager: Arc<Mutex<SchedulerManager>>,
        scheduler_name: String,
        args: Vec<String>,
        sudo_password: String,
    ) {
        // Get scheduler binary path
        let binary_path = {
            let mgr = manager.lock().await;
            if let Some(temp_dir) = mgr.temp_dir.as_ref() {
                temp_dir.join(&scheduler_name)
            } else {
                execution.mark_failed("Scheduler binaries not extracted").await;
                return;
            }
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
                return;
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
    }
}

#[tool_router]
impl SchedMcpServer {
    async fn new(sudo_password: String) -> Result<Self> {
        let mut scheduler_manager = SchedulerManager::new()?;
        
        // Extract schedulers on startup
        scheduler_manager.extract_schedulers().await?;
        
        let server = Self {
            tool_router: Self::tool_router(),
            sudo_password: Arc::new(sudo_password),
            scheduler_manager: Arc::new(Mutex::new(scheduler_manager)),
            executions: Arc::new(DashMap::new()),
        };

        // Start cleanup task
        let executions = server.executions.clone();
        tokio::spawn(async move {
            let cleanup_interval = Duration::from_secs(300); // 5 minutes
            let max_age = 3600; // 1 hour

            loop {
                sleep(cleanup_interval).await;
                let current_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                let mut to_remove = Vec::new();
                for entry in executions.iter() {
                    let is_completed = {
                        let status = entry.value().status.lock().await;
                        status.contains("completed") || status.contains("failed")
                    };
                    
                    if is_completed && current_time - entry.value().start_time > max_age {
                        to_remove.push(entry.key().clone());
                    }
                }

                for key in to_remove {
                    executions.remove(&key);
                }
            }
        });

        Ok(server)
    }

    #[tool(description = "List available sched-ext schedulers with detailed information")]
    async fn list_schedulers(
        &self,
        Parameters(ListSchedulersRequest { name, production_ready }): Parameters<ListSchedulersRequest>,
    ) -> Result<CallToolResult, McpError> {
        let manager = self.scheduler_manager.lock().await;
        let schedulers = manager.list_schedulers();

        let filtered_schedulers: Vec<SchedulerSummary> = schedulers
            .iter()
            .filter(|s| {
                let name_match = name.as_ref().map_or(true, |n| s.name.contains(n));
                let prod_match = production_ready.map_or(true, |pr| s.production_ready == pr);
                name_match && prod_match
            })
            .map(|s| SchedulerSummary {
                name: s.name.clone(),
                production_ready: s.production_ready,
                description: s.description.clone(),
                algorithm: s.algorithm.clone(),
                use_cases: s.use_cases.clone(),
                characteristics: s.characteristics.clone(),
                tuning_parameters: s.tuning_parameters.clone(),
                limitations: s.limitations.clone(),
                performance_profile: s.performance_profile.clone(),
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json!({
                "schedulers": filtered_schedulers
            })).unwrap()
        )]))
    }


    #[tool(description = "Run a sched-ext scheduler")]
    async fn run_scheduler(
        &self,
        Parameters(RunSchedulerRequest { name, args }): Parameters<RunSchedulerRequest>,
    ) -> Result<CallToolResult, McpError> {
        let manager = self.scheduler_manager.lock().await;
        
        // Check if scheduler exists
        if manager.get_scheduler(&name).is_none() {
            return Err(McpError::invalid_params(
                "Scheduler not found",
                Some(json!({"scheduler": name})),
            ));
        }

        // Generate execution ID
        let execution_id = format!("sched_{}", Uuid::new_v4().to_string()[..8].to_string());
        
        // Create execution record
        let execution = SchedulerExecution::new(execution_id.clone(), name.clone());
        self.executions.insert(execution_id.clone(), execution.clone());

        // Start scheduler in background
        let manager_clone = self.scheduler_manager.clone();
        let sudo_password = self.sudo_password.to_string();
        let scheduler_name = name.clone();
        tokio::spawn(async move {
            SchedMcpServer::run_scheduler_background(
                execution,
                manager_clone,
                scheduler_name,
                args,
                sudo_password,
            ).await;
        });

        // Give it a moment to start
        sleep(Duration::from_millis(500)).await;

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "execution_id": execution_id,
                "scheduler": name,
                "status": "started",
                "message": "Scheduler started successfully"
            }).to_string()
        )]))
    }

    #[tool(description = "Stop a running scheduler")]
    async fn stop_scheduler(
        &self,
        Parameters(StopSchedulerRequest { execution_id }): Parameters<StopSchedulerRequest>,
    ) -> Result<CallToolResult, McpError> {
        if let Some(execution) = self.executions.get(&execution_id) {
            let pid = execution.pid.lock().await;
            
            if let Some(pid) = *pid {
                // Kill the process using sudo
                let mut cmd = Command::new("sudo");
                cmd.arg("-S")
                    .arg("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .stdin(std::process::Stdio::piped());

                let mut child = match cmd.spawn() {
                    Ok(child) => child,
                    Err(e) => {
                        return Err(McpError::internal_error(
                            "Failed to stop scheduler",
                            Some(json!({"error": e.to_string()})),
                        ));
                    }
                };

                // Send sudo password
                if let Some(mut stdin) = child.stdin.take() {
                    use tokio::io::AsyncWriteExt;
                    let _ = stdin.write_all(format!("{}\n", self.sudo_password).as_bytes()).await;
                }

                let _ = child.wait().await;

                Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "execution_id": execution_id,
                        "status": "stopped",
                        "message": "Scheduler stop signal sent"
                    }).to_string()
                )]))
            } else {
                Err(McpError::invalid_params(
                    "Scheduler process ID not available",
                    None,
                ))
            }
        } else {
            Err(McpError::invalid_params(
                "Execution ID not found",
                None,
            ))
        }
    }

    #[tool(description = "Get status of a scheduler execution")]
    async fn get_execution_status(
        &self,
        Parameters(GetExecutionStatusRequest { execution_id }): Parameters<GetExecutionStatusRequest>,
    ) -> Result<CallToolResult, McpError> {
        if let Some(execution) = self.executions.get(&execution_id) {
            let status = execution.status.lock().await.clone();
            let pid = *execution.pid.lock().await;
            let end_time = *execution.end_time.lock().await;
            let exit_code = *execution.exit_code.lock().await;
            
            let duration = end_time.map(|end| end - execution.start_time);

            Ok(CallToolResult::success(vec![Content::text(
                json!({
                    "execution_id": execution_id,
                    "scheduler_name": execution.scheduler_name,
                    "status": status,
                    "pid": pid,
                    "start_time": execution.start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "exit_code": exit_code
                }).to_string()
            )]))
        } else {
            Err(McpError::invalid_params(
                "Execution ID not found",
                None,
            ))
        }
    }
}

#[tool_handler]
impl ServerHandler for SchedMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("MCP server for sched-ext - provides Linux kernel scheduler management capabilities".to_string()),
        }
    }
}

fn verify_password(password: &str) -> Result<()> {
    // Test the password with a simple sudo command
    let output = std::process::Command::new("sudo")
        .arg("-S")
        .arg("true")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(format!("{}\n", password).as_bytes())?;
                stdin.flush()?;
            }
            child.wait_with_output()
        })?;
    
    if !output.status.success() {
        std::process::exit(1);
    }
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("schedcp=info".parse()?)
                .add_directive("rmcp=info".parse()?),
        )
        .init();

    // Get password from environment variable
    let sudo_password = match std::env::var("SCHEDCP_SUDO_PASSWORD") {
        Ok(password) => {
            verify_password(&password)?;
            password
        },
        Err(_) => {
            std::process::exit(1);
        }
    };
    
    let server = SchedMcpServer::new(sudo_password).await?;
    
    info!("Starting sched-ext MCP server on stdio");
    
    let service = server.serve(stdio()).await.inspect_err(|e| {
        tracing::error!("serving error: {:?}", e);
    })?;

    service.waiting().await?;
    
    Ok(())
}