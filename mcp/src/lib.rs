pub use rmcp::{
    handler::server::{router::tool::ToolRouter, tool::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
    ServerHandler, ServiceExt,
};

pub mod scheduler_manager;
pub use scheduler_manager::{SchedulerManager, ParameterInfo};

pub mod workload_profile;
pub use workload_profile::{WorkloadProfile, ExecutionHistory, WorkloadStore};

pub mod storage;
pub use storage::PersistentStorage;

pub mod scheduler_generator;
pub use scheduler_generator::SchedulerGenerator;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    sync::Arc,
};
use tokio::sync::Mutex;

pub type McpError = rmcp::model::ErrorData;

#[derive(Clone)]
pub struct SchedMcpServer {
    pub tool_router: ToolRouter<Self>,
    pub scheduler_manager: Arc<Mutex<SchedulerManager>>,
    pub workload_store: Arc<Mutex<WorkloadStore>>,
    pub storage: Arc<PersistentStorage>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListSchedulersRequest {
    #[schemars(description = "Filter by scheduler name (partial match). Should leave empty by default.")]
    pub name: Option<String>,
    #[schemars(description = "Filter by production readiness. Should leave empty by default.")]
    pub production_ready: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RunSchedulerRequest {
    #[schemars(description = "Name of the scheduler to run, it should be one of the names returned by list_schedulers")]
    pub name: String,
    #[schemars(description = "Arguments to pass to the scheduler as a list of strings, as listed in the list_schedulers.")]
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetExecutionStatusRequest {
    #[schemars(description = "Execution ID to query")]
    pub execution_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StopSchedulerRequest {
    #[schemars(description = "Execution ID of the running scheduler")]
    pub execution_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CreateWorkloadProfileRequest {
    #[schemars(description = "Natural language description of the workload")]
    pub description: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AddExecutionHistoryRequest {
    #[schemars(description = "Workload profile ID")]
    pub workload_id: String,
    #[schemars(description = "Execution ID from run_scheduler")]
    pub execution_id: String,
    #[schemars(description = "Natural language description of the result")]
    pub result_description: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListWorkloadProfilesRequest {
    // Empty for now, can add filters later
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetWorkloadHistoryRequest {
    #[schemars(description = "Workload profile ID")]
    pub workload_id: String,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkloadCommand {
    #[schemars(description = "Command to execute: 'create', 'list', 'get_history', or 'add_history'")]
    pub command: String,
    #[schemars(description = "Description for create command")]
    pub description: Option<String>,
    #[schemars(description = "Workload ID for get_history and add_history commands")]
    pub workload_id: Option<String>,
    #[schemars(description = "Execution ID for add_history command")]
    pub execution_id: Option<String>,
    #[schemars(description = "Result description for add_history command")]
    pub result_description: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct WorkloadRequest {
    #[schemars(description = "Command to execute: 'create', 'list', 'get_history', or 'add_history'")]
    pub command: String,
    #[schemars(description = "Description for create command")]
    pub description: Option<String>,
    #[schemars(description = "Workload ID for get_history and add_history commands")]
    pub workload_id: Option<String>,
    #[schemars(description = "Execution ID for add_history command")]
    pub execution_id: Option<String>,
    #[schemars(description = "Result description for add_history command")]
    pub result_description: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CreateSchedulerRequest {
    #[schemars(description = "Name for the custom scheduler")]
    pub name: String,
    #[schemars(description = "BPF C source code for the scheduler")]
    pub source_code: String,
    #[schemars(description = "Description of what the scheduler does")]
    pub description: String,
    #[schemars(description = "Algorithm type (e.g., 'vruntime-based', 'EDF', 'FIFO', 'Custom')")]
    pub algorithm: String,
    #[schemars(description = "Use cases for this scheduler")]
    pub use_cases: Vec<String>,
    #[schemars(description = "Key characteristics of the scheduler")]
    pub characteristics: String,
    #[schemars(description = "Known limitations of the scheduler")]
    pub limitations: String,
    #[schemars(description = "Performance profile (e.g., 'Low latency', 'High throughput', 'Balanced')")]
    pub performance_profile: String,
}

impl SchedMcpServer {
    /// Create a new server for testing with a custom storage path
    pub async fn new_with_storage(storage_path: String) -> Result<Self> {
        let storage = Arc::new(PersistentStorage::with_path(storage_path));
        let workload_store = Arc::new(Mutex::new(storage.load()?));
        let scheduler_manager = Arc::new(Mutex::new(SchedulerManager::new()?));
        
        let server = Self {
            tool_router: ToolRouter::new(),
            scheduler_manager,
            workload_store: workload_store.clone(),
            storage: storage.clone(),
        };

        Ok(server)
    }

    /// Core implementation for list_schedulers
    pub async fn list_schedulers_impl(&self, request: ListSchedulersRequest) -> Result<String, McpError> {
        let manager = self.scheduler_manager.lock().await;
        let schedulers = manager.list_schedulers();
        
        let filtered: Vec<_> = schedulers
            .iter()
            .filter(|s| {
                if let Some(ref name_filter) = request.name {
                    if !s.name.to_lowercase().contains(&name_filter.to_lowercase()) {
                        return false;
                    }
                }
                if let Some(ready) = request.production_ready {
                    if s.production_ready != ready {
                        return false;
                    }
                }
                true
            })
            .map(|s| {
                json!({
                    "name": s.name,
                    "production_ready": s.production_ready,
                    "description": s.description,
                    "algorithm": s.algorithm,
                    "use_cases": s.use_cases,
                    "characteristics": s.characteristics,
                    "tuning_parameters": s.tuning_parameters,
                    "limitations": s.limitations,
                    "performance_profile": s.performance_profile,
                })
            })
            .collect();

        Ok(json!({
            "schedulers": filtered
        }).to_string())
    }

    /// Core implementation for run_scheduler
    pub async fn run_scheduler_impl(&self, request: RunSchedulerRequest) -> Result<String, McpError> {
        let manager = self.scheduler_manager.lock().await;

        // create_execution will handle both built-in and custom schedulers
        let execution_id = manager.create_execution(&request.name, request.args.clone())
            .await
            .map_err(|e| McpError::invalid_params(
                e.to_string(),
                None,
            ))?;

        Ok(json!({
            "execution_id": execution_id,
            "scheduler": request.name,
            "status": "started",
            "message": format!("Scheduler {} started with execution ID {}", request.name, execution_id)
        }).to_string())
    }

    /// Core implementation for stop_scheduler
    pub async fn stop_scheduler_impl(&self, execution_id: &str) -> Result<String, McpError> {
        let manager = self.scheduler_manager.lock().await;
        manager.stop_scheduler(execution_id).await
            .map_err(|e| McpError::invalid_params(
                e.to_string(),
                None,
            ))?;

        Ok(json!({
            "execution_id": execution_id,
            "status": "stopped",
            "message": format!("Scheduler execution {} stopped", execution_id)
        }).to_string())
    }

    /// Core implementation for get_execution_status
    pub async fn get_execution_status_impl(&self, request: GetExecutionStatusRequest) -> Result<String, McpError> {
        let manager = self.scheduler_manager.lock().await;
        
        if let Some(status) = manager.get_execution(&request.execution_id).await {
            let duration = status.end_time
                .map(|end| end - status.start_time);
            
            let exit_code = if status.status == "stopped" || status.status == "failed" {
                Some(if status.status == "failed" { 1 } else { 0 })
            } else {
                None
            };

            Ok(json!({
                "execution_id": status.execution_id,
                "scheduler_name": status.scheduler_name,
                "command": status.command,
                "args": status.args,
                "status": status.status,
                "pid": status.pid,
                "start_time": status.start_time,
                "end_time": status.end_time,
                "duration": duration,
                "exit_code": exit_code,
                "output": status.output,
            }).to_string())
        } else {
            Err(McpError::invalid_params(
                format!("Execution '{}' not found", request.execution_id),
                Some(json!({"execution_id": request.execution_id})),
            ))
        }
    }

    /// Core implementation for create_workload_profile
    pub async fn create_workload_profile_impl(&self, description: &str) -> Result<String, McpError> {
        let mut store = self.workload_store.lock().await;
        let workload_id = store.create_profile(description.to_string());
        
        // Save to storage
        self.storage.save(&*store)
            .map_err(|e| McpError::invalid_params(
                e.to_string(),
                None,
            ))?;

        Ok(json!({
            "workload_id": workload_id,
            "message": format!("Workload profile '{}' created", workload_id)
        }).to_string())
    }

    /// Core implementation for add_execution_history
    pub async fn add_execution_history_impl(&self, request: AddExecutionHistoryRequest) -> Result<String, McpError> {
        // Get execution details
        let manager = self.scheduler_manager.lock().await;
        let execution = manager.get_execution(&request.execution_id).await
            .ok_or_else(|| McpError::invalid_params(
                format!("Execution '{}' not found", request.execution_id),
                Some(json!({"execution_id": request.execution_id})),
            ))?;

        let mut store = self.workload_store.lock().await;
        
        // Check if workload exists
        if store.get_profile(&request.workload_id).is_none() {
            return Err(McpError::invalid_params(
                format!("Workload profile '{}' not found", request.workload_id),
                Some(json!({"workload_id": request.workload_id})),
            ));
        }

        let history_id = store.add_history(
            request.workload_id.clone(),
            request.execution_id.clone(),
            execution.scheduler_name,
            execution.args,
            request.result_description.clone(),
        );

        // Save to storage
        self.storage.save(&*store)
            .map_err(|e| McpError::invalid_params(
                e.to_string(),
                None,
            ))?;

        Ok(json!({
            "history_id": history_id,
            "message": "Execution history added"
        }).to_string())
    }

    /// Core implementation for list_workload_profiles
    pub async fn list_workload_profiles_impl(&self) -> Result<String, McpError> {
        let store = self.workload_store.lock().await;
        let profiles = store.list_profiles();
        
        let profiles_json: Vec<_> = profiles.iter().map(|p| {
            json!({
                "id": p.id,
                "description": p.description,
                "created_at": p.created_at,
                "updated_at": p.updated_at,
            })
        }).collect();

        Ok(json!({
            "profiles": profiles_json
        }).to_string())
    }

    /// Core implementation for get_workload_history
    pub async fn get_workload_history_impl(&self, workload_id: &str) -> Result<String, McpError> {
        let store = self.workload_store.lock().await;
        
        let profile = store.get_profile(workload_id)
            .ok_or_else(|| McpError::invalid_params(
                format!("Workload profile '{}' not found", workload_id),
                Some(json!({"workload_id": workload_id})),
            ))?;
        
        let history = store.get_history_by_workload(workload_id);
        
        let history_json: Vec<_> = history.iter().map(|h| {
            json!({
                "id": h.id,
                "execution_id": h.execution_id,
                "scheduler_name": h.scheduler_name,
                "args": h.args,
                "result_description": h.result_description,
                "created_at": h.created_at,
            })
        }).collect();

        Ok(json!({
            "profile": {
                "id": profile.id,
                "description": profile.description,
                "created_at": profile.created_at,
                "updated_at": profile.updated_at,
            },
            "history": history_json
        }).to_string())
    }

    /// Core implementation for create_and_verify_scheduler
    pub async fn create_and_verify_scheduler_impl(&self, request: CreateSchedulerRequest) -> Result<String, McpError> {
        let manager = self.scheduler_manager.lock().await;

        // Create scheduler info from request
        let info = scheduler_manager::SchedulerInfo {
            name: request.name.clone(),
            production_ready: false, // Custom schedulers are not production ready by default
            description: request.description,
            use_cases: request.use_cases,
            algorithm: request.algorithm,
            characteristics: request.characteristics,
            tuning_parameters: std::collections::HashMap::new(), // No tuning parameters for now
            limitations: request.limitations,
            performance_profile: request.performance_profile,
        };

        // Create, compile, and verify the scheduler
        let result = manager.create_and_verify_scheduler(info, &request.source_code).await
            .map_err(|e| McpError::invalid_params(
                e.to_string(),
                None,
            ))?;

        Ok(result)
    }

    /// Unified workload command implementation
    pub async fn workload_impl(&self, request: WorkloadRequest) -> Result<String, McpError> {
        match request.command.as_str() {
            "create" => {
                let description = request.description
                    .ok_or_else(|| McpError::invalid_params(
                        "description is required for create command".to_string(),
                        None,
                    ))?;
                self.create_workload_profile_impl(&description).await
            },
            "list" => {
                self.list_workload_profiles_impl().await
            },
            "get_history" => {
                let workload_id = request.workload_id
                    .ok_or_else(|| McpError::invalid_params(
                        "workload_id is required for get_history command".to_string(),
                        None,
                    ))?;
                self.get_workload_history_impl(&workload_id).await
            },
            "add_history" => {
                let workload_id = request.workload_id
                    .ok_or_else(|| McpError::invalid_params(
                        "workload_id is required for add_history command".to_string(),
                        None,
                    ))?;
                let execution_id = request.execution_id
                    .ok_or_else(|| McpError::invalid_params(
                        "execution_id is required for add_history command".to_string(),
                        None,
                    ))?;
                let result_description = request.result_description
                    .ok_or_else(|| McpError::invalid_params(
                        "result_description is required for add_history command".to_string(),
                        None,
                    ))?;
                self.add_execution_history_impl(AddExecutionHistoryRequest {
                    workload_id,
                    execution_id,
                    result_description,
                }).await
            },
            _ => Err(McpError::invalid_params(
                format!("Unknown command: {}", request.command),
                None,
            ))
        }
    }

    // Test helper methods that directly call the impl methods
    pub async fn list_schedulers(&self, request: ListSchedulersRequest) -> Result<String, McpError> {
        self.list_schedulers_impl(request).await
    }

    pub async fn run_scheduler(&self, request: RunSchedulerRequest) -> Result<String, McpError> {
        self.run_scheduler_impl(request).await
    }

    pub async fn get_execution_status(&self, request: GetExecutionStatusRequest) -> Result<String, McpError> {
        self.get_execution_status_impl(request).await
    }

    pub async fn stop_scheduler(&self, request: String) -> Result<String, McpError> {
        let req: serde_json::Value = serde_json::from_str(&request)
            .map_err(|e| McpError::invalid_params(
                e.to_string(),
                None,
            ))?;
        
        let execution_id = req["execution_id"].as_str()
            .ok_or_else(|| McpError::invalid_params(
                "execution_id is required".to_string(),
                None,
            ))?;

        self.stop_scheduler_impl(execution_id).await
    }

    pub async fn create_workload_profile(&self, request: String) -> Result<String, McpError> {
        let req: serde_json::Value = serde_json::from_str(&request)
            .map_err(|e| McpError::invalid_params(
                e.to_string(),
                None,
            ))?;
        
        let description = req["description"].as_str()
            .ok_or_else(|| McpError::invalid_params(
                "description is required".to_string(),
                None,
            ))?;

        self.create_workload_profile_impl(description).await
    }

    pub async fn add_execution_history(&self, request: String) -> Result<String, McpError> {
        let req: serde_json::Value = serde_json::from_str(&request)
            .map_err(|e| McpError::invalid_params(
                e.to_string(),
                None,
            ))?;
        
        let workload_id = req["workload_id"].as_str()
            .ok_or_else(|| McpError::invalid_params(
                "workload_id is required".to_string(),
                None,
            ))?;
        let execution_id = req["execution_id"].as_str()
            .ok_or_else(|| McpError::invalid_params(
                "execution_id is required".to_string(),
                None,
            ))?;
        let result_description = req["result_description"].as_str()
            .ok_or_else(|| McpError::invalid_params(
                "result_description is required".to_string(),
                None,
            ))?;

        self.add_execution_history_impl(AddExecutionHistoryRequest {
            workload_id: workload_id.to_string(),
            execution_id: execution_id.to_string(),
            result_description: result_description.to_string(),
        }).await
    }

    pub async fn list_workload_profiles(&self, _request: String) -> Result<String, McpError> {
        self.list_workload_profiles_impl().await
    }

    pub async fn get_workload_history(&self, request: String) -> Result<String, McpError> {
        let req: serde_json::Value = serde_json::from_str(&request)
            .map_err(|e| McpError::invalid_params(
                e.to_string(),
                None,
            ))?;
        
        let workload_id = req["workload_id"].as_str()
            .ok_or_else(|| McpError::invalid_params(
                "workload_id is required".to_string(),
                None,
            ))?;

        self.get_workload_history_impl(workload_id).await
    }
}

// For testing - create a server with a custom storage path
impl SchedMcpServer {
    pub async fn new(storage_path: String) -> Result<Self> {
        Self::new_with_storage(storage_path).await
    }
}