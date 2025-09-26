use anyhow::Result;
use schedcp::*;
use rmcp::{
    handler::server::{router::tool::ToolRouter, tool::Parameters},
    model::*,
    tool, tool_handler, tool_router,
    transport::stdio,
    ServerHandler, ServiceExt,
};
use serde::Serialize;
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

// Response structs only needed for main.rs
#[derive(Debug, Serialize)]
#[allow(dead_code)]
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

// Wrapper struct for main.rs to enable tool_router macro
#[derive(Clone)]
struct McpServer {
    inner: SchedMcpServer,
    tool_router: ToolRouter<Self>,
}

impl McpServer {
    async fn new_with_sudo(sudo_password: String) -> Result<Self> {
        let mut scheduler_manager = SchedulerManager::new()?;
        
        // Always set sudo password for schedulers (empty string means passwordless sudo)
        scheduler_manager.set_sudo_password(sudo_password);
        
        // Load workload store from persistent storage
        let storage = Arc::new(PersistentStorage::new());
        let workload_store = match storage.load() {
            Ok(store) => store,
            Err(e) => {
                warn!("Failed to load workload store: {}, starting with empty store", e);
                WorkloadStore::new()
            }
        };
        
        let inner = SchedMcpServer {
            tool_router: ToolRouter::new(),
            scheduler_manager: Arc::new(Mutex::new(scheduler_manager)),
            workload_store: Arc::new(Mutex::new(workload_store)),
            storage: storage.clone(),
        };

        // Start cleanup task
        let scheduler_manager_clone = inner.scheduler_manager.clone();
        tokio::spawn(async move {
            let cleanup_interval = Duration::from_secs(300); // 5 minutes
            let max_age = 3600; // 1 hour
            
            loop {
                sleep(cleanup_interval).await;
                let mut manager = scheduler_manager_clone.lock().await;
                manager.cleanup_old_executions(max_age).await;
            }
        });
        
        // Start save task
        let workload_store_clone = inner.workload_store.clone();
        let storage_clone = inner.storage.clone();
        tokio::spawn(async move {
            let save_interval = Duration::from_secs(60); // 1 minute
            
            loop {
                sleep(save_interval).await;
                let store = workload_store_clone.lock().await;
                if let Err(e) = storage_clone.save(&*store) {
                    error!("Failed to save workload store: {}", e);
                }
            }
        });

        Ok(Self { 
            inner,
            tool_router: Self::tool_router(),
        })
    }
}

#[tool_router]
impl McpServer {
    #[tool(description = "List available sched-ext schedulers with detailed information. Always use this tool to get the info of all schedulers first when you are writing or using a scheduler.")]
    async fn list_schedulers(
        &self,
        Parameters(request): Parameters<ListSchedulersRequest>,
    ) -> Result<CallToolResult, McpError> {
        info!("list_schedulers called with filters: name={:?}, production_ready={:?}", 
              request.name, request.production_ready);
        
        let result = self.inner.list_schedulers_impl(request).await?;
        
        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(description = "Run a sched-ext scheduler. After you list_schedulers, you can use this tool to run a scheduler from the list. Run a new scheduler will stop any running scheduler first.")]
    async fn run_scheduler(
        &self,
        Parameters(request): Parameters<RunSchedulerRequest>,
    ) -> Result<CallToolResult, McpError> {
        info!("run_scheduler called for scheduler '{}' with args: {:?}", request.name, request.args);
        
        // Stop any currently running schedulers first
        {
            let manager = self.inner.scheduler_manager.lock().await;
            let stopped = manager.stop_running_schedulers().await;
            if !stopped.is_empty() {
                info!("Stopped {} running scheduler(s) before starting new one", stopped.len());
            }
        }
        
        // Wait a bit to ensure clean shutdown
        sleep(Duration::from_millis(500)).await;
        
        let result = self.inner.run_scheduler_impl(request).await?;
        
        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(description = "Stop a running scheduler. You can use this tool to stop a scheduler that is running.")]
    async fn stop_scheduler(
        &self,
        Parameters(StopSchedulerRequest { execution_id }): Parameters<StopSchedulerRequest>,
    ) -> Result<CallToolResult, McpError> {
        info!("stop_scheduler called for execution_id: {}", execution_id);
        
        let result = self.inner.stop_scheduler_impl(&execution_id).await?;
        
        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(description = "Get status of a scheduler execution, and the output of the scheduler.")]
    async fn get_execution_status(
        &self,
        Parameters(request): Parameters<GetExecutionStatusRequest>,
    ) -> Result<CallToolResult, McpError> {
        debug!("get_execution_status called for execution_id: {}", request.execution_id);
        
        let result = self.inner.get_execution_status_impl(request).await?;
        
        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(description = "Create a new workload profile with a natural language description. When you are asked to optimize scheduler for a specific workload, you should analyze the workload first and then use this tool to create a workload profile, like claude.md.")]
    async fn create_workload_profile(
        &self,
        Parameters(CreateWorkloadProfileRequest { description }): Parameters<CreateWorkloadProfileRequest>,
    ) -> Result<CallToolResult, McpError> {
        info!("create_workload_profile called with description: {}", description);
        
        let result = self.inner.create_workload_profile_impl(&description).await?;
        
        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(description = "Add execution history to a workload profile. After execute the profile and collect the meaningful metrics, you should use this tool to add the history and the metrics (like end to end completion time, latency, etc.) to the workload profile.")]
    async fn add_execution_history(
        &self,
        Parameters(request): Parameters<AddExecutionHistoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        info!("add_execution_history called for workload_id: {}, execution_id: {}", 
              request.workload_id, request.execution_id);
        
        let result = self.inner.add_execution_history_impl(request).await?;
        
        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(description = "List all workload profiles")]
    async fn list_workload_profiles(
        &self,
        Parameters(ListWorkloadProfilesRequest {}): Parameters<ListWorkloadProfilesRequest>,
    ) -> Result<CallToolResult, McpError> {
        info!("list_workload_profiles called");
        
        let result = self.inner.list_workload_profiles_impl().await?;
        
        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(description = "Get workload profile with its execution history. When you need to create or select a new scheduler, you should use this tool to get the history of the workload profile to understand your previous experiments.")]
    async fn get_workload_history(
        &self,
        Parameters(GetWorkloadHistoryRequest { workload_id }): Parameters<GetWorkloadHistoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        info!("get_workload_history called for workload_id: {}", workload_id);
        
        let result = self.inner.get_workload_history_impl(&workload_id).await?;
        
        Ok(CallToolResult::success(vec![Content::text(result)]))
    }
}

#[tool_handler]
impl ServerHandler for McpServer {
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

    if output.status.success() {
        Ok(())
    } else {
        anyhow::bail!("Invalid sudo password")
    }
}

fn setup_logging() -> Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};
    
    let app_name = "schedcp";
    let home_dir = std::env::var("HOME")
        .unwrap_or_else(|_| ".".to_string());
    let log_dir = std::path::Path::new(&home_dir).join(".schedcp").join("logs");
    std::fs::create_dir_all(&log_dir)?;
    
    let file_appender = tracing_appender::rolling::daily(&log_dir, format!("{}.log", app_name));
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    
    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_target(false)
        .with_thread_ids(true)
        .with_line_number(true)
        .with_file(true);
    
    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .compact()
        .with_target(false)
        .with_filter(tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("schedcp=warn".parse()?)
            .add_directive("warn".parse()?));
    
    tracing_subscriber::registry()
        .with(file_layer)
        .with(stderr_layer)
        .init();
    
    info!("Logging initialized. Logs will be written to: {}", log_dir.display());
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging first
    setup_logging()?;
    
    info!("Starting sched-ext MCP server");
    
    // Load environment variables from .env file if it exists
    dotenv::dotenv().ok();
    
    // Get sudo password from environment or prompt
    let sudo_password = if let Ok(pass) = std::env::var("SUDO_PASSWORD") {
        info!("Using sudo password from SUDO_PASSWORD environment variable");
        pass
    } else {
        // Try empty password first (passwordless sudo)
        if verify_password("").is_ok() {
            info!("Passwordless sudo detected");
            "".to_string()
        } else {
            info!("No SUDO_PASSWORD environment variable set and passwordless sudo not available");
            info!("Schedulers will run without sudo - this may cause permission errors");
            "".to_string()
        }
    };
    
    let server = McpServer::new_with_sudo(sudo_password).await?;
    
    info!("Starting sched-ext MCP server on stdio");
    
    let service = server.serve(stdio()).await.inspect_err(|e| {
        error!("serving error: {:?}", e);
    })?;

    info!("MCP server is now running and waiting for requests");
    service.waiting().await?;
    
    info!("MCP server shutting down");
    Ok(())
}