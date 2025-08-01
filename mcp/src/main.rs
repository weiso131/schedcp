use anyhow::Result;
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
    time::Duration,
};
use tokio::{
    sync::Mutex,
    time::sleep,
};
use tracing::{info, debug, error, warn};

mod scheduler_manager;
use scheduler_manager::{SchedulerManager, ParameterInfo};

type McpError = rmcp::model::ErrorData;

#[derive(Clone)]
struct SchedMcpServer {
    tool_router: ToolRouter<Self>,
    scheduler_manager: Arc<Mutex<SchedulerManager>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ListSchedulersRequest {
    #[schemars(description = "Filter by scheduler name (partial match). Should leave empty by default.")]
    name: Option<String>,
    #[schemars(description = "Filter by production readiness. Should leave empty by default.")]
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
    #[schemars(description = "Name of the scheduler to run, it should be one of the names returned by list_schedulers")]
    name: String,
    #[schemars(description = "Arguments to pass to the scheduler as a list of strings, as listed in the list_schedulers.")]
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
    output: Vec<String>,
}

impl SchedMcpServer {}

#[tool_router]
impl SchedMcpServer {
    async fn new(sudo_password: String) -> Result<Self> {
        let mut scheduler_manager = SchedulerManager::new()?;
        
        // Set sudo password if provided
        if !sudo_password.is_empty() {
            scheduler_manager.set_sudo_password(sudo_password);
        }
        
        let server = Self {
            tool_router: Self::tool_router(),
            scheduler_manager: Arc::new(Mutex::new(scheduler_manager)),
        };

        // Start cleanup task
        let scheduler_manager_clone = server.scheduler_manager.clone();
        tokio::spawn(async move {
            let cleanup_interval = Duration::from_secs(300); // 5 minutes
            let max_age = 3600; // 1 hour

            loop {
                sleep(cleanup_interval).await;
                
                let manager = scheduler_manager_clone.lock().await;
                manager.cleanup_old_executions(max_age).await;
            }
        });

        Ok(server)
    }

    #[tool(description = "List available sched-ext schedulers with detailed information. Always use this tool to get the info of all schedulers first when you are writing or using a scheduler.")]
    async fn list_schedulers(
        &self,
        Parameters(ListSchedulersRequest { name, production_ready }): Parameters<ListSchedulersRequest>,
    ) -> Result<CallToolResult, McpError> {
        info!("list_schedulers called with filters: name={:?}, production_ready={:?}", name, production_ready);
        
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

        info!("list_schedulers returning {} schedulers", filtered_schedulers.len());
        
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json!({
                "schedulers": filtered_schedulers
            })).unwrap()
        )]))
    }

    #[tool(description = "Run a sched-ext scheduler. After you list_schedulers, you can use this tool to run a scheduler from the list. Run a new scheduler will stop any running scheduler first.")]
    async fn run_scheduler(
        &self,
        Parameters(RunSchedulerRequest { name, args }): Parameters<RunSchedulerRequest>,
    ) -> Result<CallToolResult, McpError> {
        info!("run_scheduler called for scheduler '{}' with args: {:?}", name, args);
        
        let manager = self.scheduler_manager.lock().await;
        
        // Stop any running schedulers first
        let stopped_schedulers = manager.stop_running_schedulers().await;
        if !stopped_schedulers.is_empty() {
            info!("Stopped {} running schedulers: {:?}", stopped_schedulers.len(), stopped_schedulers);
        }

        // Create and start execution
        let execution_id = match manager.create_execution(&name, args).await {
            Ok(id) => {
                info!("Successfully created execution {} for scheduler '{}'", id, name);
                id
            },
            Err(e) => {
                error!("Failed to start scheduler '{}': {}", name, e);
                return Err(McpError::invalid_params(
                    "Failed to start scheduler",
                    Some(json!({"scheduler": name, "error": e.to_string()})),
                ));
            }
        };

        // Give it a moment to start
        sleep(Duration::from_millis(500)).await;

        let message = if stopped_schedulers.is_empty() {
            "Scheduler started successfully".to_string()
        } else {
            format!("Scheduler started successfully (stopped {} previous scheduler(s))", stopped_schedulers.len())
        };

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "execution_id": execution_id,
                "scheduler": name,
                "status": "started",
                "message": message,
                "stopped_schedulers": stopped_schedulers
            }).to_string()
        )]))
    }

    #[tool(description = "Stop a running scheduler. You can use this tool to stop a scheduler that is running.")]
    async fn stop_scheduler(
        &self,
        Parameters(StopSchedulerRequest { execution_id }): Parameters<StopSchedulerRequest>,
    ) -> Result<CallToolResult, McpError> {
        info!("stop_scheduler called for execution_id: {}", execution_id);
        
        let manager = self.scheduler_manager.lock().await;
        
        match manager.stop_scheduler(&execution_id).await {
            Ok(_) => {
                info!("Successfully stopped scheduler with execution_id: {}", execution_id);
                Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "execution_id": execution_id,
                        "status": "stopped",
                        "message": "Scheduler stop signal sent"
                    }).to_string()
                )]))
            }
            Err(e) => {
                error!("Failed to stop scheduler with execution_id {}: {}", execution_id, e);
                Err(McpError::invalid_params(
                    "Failed to stop scheduler",
                    Some(json!({"execution_id": execution_id, "error": e.to_string()})),
                ))
            }
        }
    }

    #[tool(description = "Get status of a scheduler execution, and the output of the scheduler.")]
    async fn get_execution_status(
        &self,
        Parameters(GetExecutionStatusRequest { execution_id }): Parameters<GetExecutionStatusRequest>,
    ) -> Result<CallToolResult, McpError> {
        debug!("get_execution_status called for execution_id: {}", execution_id);
        
        let manager = self.scheduler_manager.lock().await;
        
        if let Some(exec_status) = manager.get_execution(&execution_id).await {
            let duration = exec_status.end_time.map(|end| end - exec_status.start_time);
            
            Ok(CallToolResult::success(vec![Content::text(
                json!({
                    "execution_id": execution_id,
                    "scheduler_name": exec_status.scheduler_name,
                    "status": exec_status.status,
                    "pid": exec_status.pid,
                    "start_time": exec_status.start_time,
                    "end_time": exec_status.end_time,
                    "duration": duration,
                    "exit_code": None::<i32>, // Not provided by process_manager
                    "output": exec_status.output
                }).to_string()
            )]))
        } else {
            warn!("Execution ID not found: {}", execution_id);
            Err(McpError::invalid_params(
                "Execution ID not found",
                Some(json!({"execution_id": execution_id})),
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
    
    // Set up file logging
    let file_appender = tracing_appender::rolling::daily(".", "schedcp.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    
    // Set up combined console and file logging
    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_ansi(false) // Disable ANSI colors in log file
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("schedcp=info".parse()?)
                .add_directive("rmcp=info".parse()?)
                .add_directive("process_manager=info".parse()?),
        )
        .init();
    
    info!("SchedCP MCP server starting up");

    // Get password from environment variable
    let sudo_password = match std::env::var("SCHEDCP_SUDO_PASSWORD") {
        Ok(password) => {
            verify_password(&password)?;
            password
        },
        Err(_) => {
            info!("No sudo password provided, running without sudo");
            "".to_string()
        }
    };
    
    let server = SchedMcpServer::new(sudo_password).await?;
    
    info!("Starting sched-ext MCP server on stdio");
    
    let service = server.serve(stdio()).await.inspect_err(|e| {
        error!("serving error: {:?}", e);
    })?;

    info!("MCP server is now running and waiting for requests");
    service.waiting().await?;
    
    info!("MCP server shutting down");
    Ok(())
}