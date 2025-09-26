use crate::workload_profile::WorkloadStore;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::fs;
use tracing::{info, error, warn};

const STORAGE_FILE: &str = "schedcp_workloads.json";

pub struct PersistentStorage {
    file_path: PathBuf,
}

impl PersistentStorage {
    pub fn new() -> Self {
        let home_dir = std::env::var("HOME")
            .unwrap_or_else(|_| ".".to_string());
        let schedcp_dir = Path::new(&home_dir).join(".schedcp");
        
        // Ensure the directory exists
        if let Err(e) = fs::create_dir_all(&schedcp_dir) {
            error!("Failed to create ~/.schedcp directory: {}", e);
        }
        
        Self {
            file_path: schedcp_dir.join(STORAGE_FILE),
        }
    }

    #[allow(dead_code)]
    pub fn with_path<P: AsRef<Path>>(path: P) -> Self {
        Self {
            file_path: path.as_ref().to_path_buf(),
        }
    }

    pub fn load(&self) -> Result<WorkloadStore> {
        if !Path::new(&self.file_path).exists() {
            info!("Storage file not found, creating new workload store");
            return Ok(WorkloadStore::new());
        }

        let data = fs::read_to_string(&self.file_path)?;
        match serde_json::from_str(&data) {
            Ok(store) => {
                info!("Loaded workload store from {}", self.file_path.display());
                Ok(store)
            }
            Err(e) => {
                error!("Failed to parse storage file: {}", e);
                warn!("Creating backup and starting with new store");
                
                // Create backup of corrupted file
                let backup_path = self.file_path.with_extension(
                    format!("backup.{}",
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    )
                );
                fs::copy(&self.file_path, &backup_path)?;
                
                Ok(WorkloadStore::new())
            }
        }
    }

    pub fn save(&self, store: &WorkloadStore) -> Result<()> {
        let data = serde_json::to_string_pretty(store)?;
        
        // Ensure parent directory exists
        if let Some(parent) = Path::new(&self.file_path).parent() {
            fs::create_dir_all(parent)?;
        }
        
        // Write to temporary file first
        let temp_path = self.file_path.with_extension("tmp");
        fs::write(&temp_path, data)?;
        
        // Atomic rename
        fs::rename(&temp_path, &self.file_path)?;
        
        info!("Saved workload store to {}", self.file_path.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("test_storage.json");
        let storage = PersistentStorage::with_path(&storage_path);
        
        // Create and populate store
        let mut store = WorkloadStore::new();
        let profile_id = store.create_profile("Test workload".to_string());
        store.add_history(
            profile_id.clone(),
            "exec-123".to_string(),
            "scx_bpfland".to_string(),
            vec!["--arg".to_string()],
            "Test result".to_string(),
        );
        
        // Save
        storage.save(&store).unwrap();
        assert!(storage_path.exists());
        
        // Load
        let loaded_store = storage.load().unwrap();
        assert_eq!(loaded_store.profiles.len(), 1);
        assert_eq!(loaded_store.history.len(), 1);
        
        let loaded_profile = loaded_store.get_profile(&profile_id).unwrap();
        assert_eq!(loaded_profile.description, "Test workload");
    }

    #[test]
    fn test_load_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("nonexistent.json");
        let storage = PersistentStorage::with_path(&storage_path);
        
        let store = storage.load().unwrap();
        assert_eq!(store.profiles.len(), 0);
        assert_eq!(store.history.len(), 0);
    }

    #[test]
    fn test_corrupted_file_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("corrupted.json");
        
        // Write corrupted data
        fs::write(&storage_path, "{ invalid json").unwrap();
        
        let storage = PersistentStorage::with_path(&storage_path);
        let store = storage.load().unwrap();
        
        // Should create new store
        assert_eq!(store.profiles.len(), 0);
        
        // Should have created backup
        let backup_exists = temp_dir.path()
            .read_dir()
            .unwrap()
            .any(|entry| {
                entry.unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains("corrupted.json.backup")
            });
        assert!(backup_exists);
    }
}