use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadProfile {
    pub id: String,
    pub description: String,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionHistory {
    pub id: String,
    pub workload_id: String,
    pub execution_id: String,
    pub scheduler_name: String,
    pub args: Vec<String>,
    pub result_description: String,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadStore {
    pub profiles: HashMap<String, WorkloadProfile>,
    pub history: Vec<ExecutionHistory>,
}

impl WorkloadStore {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            history: Vec::new(),
        }
    }

    pub fn create_profile(&mut self, description: String) -> String {
        let id = Uuid::new_v4().to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let profile = WorkloadProfile {
            id: id.clone(),
            description,
            created_at: now,
            updated_at: now,
        };
        
        self.profiles.insert(id.clone(), profile);
        id
    }

    pub fn update_profile(&mut self, id: &str, description: String) -> Result<()> {
        if let Some(profile) = self.profiles.get_mut(id) {
            profile.description = description;
            profile.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            Ok(())
        } else {
            Err(anyhow::anyhow!("Workload profile not found"))
        }
    }

    pub fn get_profile(&self, id: &str) -> Option<&WorkloadProfile> {
        self.profiles.get(id)
    }

    pub fn list_profiles(&self) -> Vec<&WorkloadProfile> {
        self.profiles.values().collect()
    }

    pub fn delete_profile(&mut self, id: &str) -> Result<()> {
        if self.profiles.remove(id).is_some() {
            // Also remove associated history entries
            self.history.retain(|h| h.workload_id != id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Workload profile not found"))
        }
    }

    pub fn add_history(
        &mut self,
        workload_id: String,
        execution_id: String,
        scheduler_name: String,
        args: Vec<String>,
        result_description: String,
    ) -> String {
        let id = Uuid::new_v4().to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let entry = ExecutionHistory {
            id: id.clone(),
            workload_id,
            execution_id,
            scheduler_name,
            args,
            result_description,
            created_at: now,
        };
        
        self.history.push(entry);
        id
    }

    pub fn get_history_by_workload(&self, workload_id: &str) -> Vec<&ExecutionHistory> {
        self.history
            .iter()
            .filter(|h| h.workload_id == workload_id)
            .collect()
    }

    pub fn get_history_by_execution(&self, execution_id: &str) -> Option<&ExecutionHistory> {
        self.history
            .iter()
            .find(|h| h.execution_id == execution_id)
    }

    pub fn get_all_history(&self) -> &Vec<ExecutionHistory> {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_profile() {
        let mut store = WorkloadStore::new();
        let id = store.create_profile("Test workload".to_string());
        
        assert!(!id.is_empty());
        assert_eq!(store.profiles.len(), 1);
        
        let profile = store.get_profile(&id).unwrap();
        assert_eq!(profile.description, "Test workload");
    }

    #[test]
    fn test_update_profile() {
        let mut store = WorkloadStore::new();
        let id = store.create_profile("Initial description".to_string());
        
        store.update_profile(&id, "Updated description".to_string()).unwrap();
        
        let profile = store.get_profile(&id).unwrap();
        assert_eq!(profile.description, "Updated description");
        assert!(profile.updated_at >= profile.created_at);
    }

    #[test]
    fn test_delete_profile() {
        let mut store = WorkloadStore::new();
        let id = store.create_profile("Test workload".to_string());
        
        // Add some history
        store.add_history(
            id.clone(),
            "exec-123".to_string(),
            "scx_bpfland".to_string(),
            vec!["--slice-us".to_string(), "20000".to_string()],
            "Good performance".to_string(),
        );
        
        assert_eq!(store.history.len(), 1);
        
        store.delete_profile(&id).unwrap();
        
        assert_eq!(store.profiles.len(), 0);
        assert_eq!(store.history.len(), 0);
    }

    #[test]
    fn test_add_history() {
        let mut store = WorkloadStore::new();
        let workload_id = store.create_profile("Test workload".to_string());
        
        let history_id = store.add_history(
            workload_id.clone(),
            "exec-123".to_string(),
            "scx_bpfland".to_string(),
            vec!["--slice-us".to_string(), "20000".to_string()],
            "Good performance".to_string(),
        );
        
        assert!(!history_id.is_empty());
        assert_eq!(store.history.len(), 1);
        
        let history = store.get_history_by_workload(&workload_id);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].scheduler_name, "scx_bpfland");
    }
}