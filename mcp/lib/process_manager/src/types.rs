use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    pub id: Uuid,
    pub name: String,
    pub binary_name: String,
    pub pid: Option<u32>,
    pub status: ProcessStatus,
    pub args: Vec<String>,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub stopped_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessStatus {
    Pending,
    Running,
    Stopped,
    Failed,
}

#[derive(Debug, Clone)]
pub struct ProcessConfig {
    pub name: String,
    pub binary_name: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub working_dir: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ProcessError {
    #[error("Binary not found: {0}")]
    BinaryNotFound(String),
    
    #[error("Process not found: {0}")]
    ProcessNotFound(Uuid),
    
    #[error("Failed to start process: {0}")]
    StartFailed(String),
    
    #[error("Failed to stop process: {0}")]
    StopFailed(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Binary extraction failed: {0}")]
    ExtractionFailed(String),
}