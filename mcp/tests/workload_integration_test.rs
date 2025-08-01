use anyhow::Result;
use schedcp::workload_profile::{WorkloadStore, WorkloadProfile, ExecutionHistory};
use schedcp::storage::PersistentStorage;
use tempfile::TempDir;
use std::path::Path;

#[tokio::test]
async fn test_workload_profile_workflow() -> Result<()> {
    // Create temporary directory for storage
    let temp_dir = TempDir::new()?;
    let storage_path = temp_dir.path().join("test_workloads.json");
    let storage = PersistentStorage::with_path(&storage_path);
    
    // Create workload store
    let mut store = WorkloadStore::new();
    
    // Test 1: Create workload profile
    let workload_id = store.create_profile("Heavy computation workload".to_string());
    assert!(!workload_id.is_empty());
    
    // Test 2: Get profile
    let profile = store.get_profile(&workload_id).unwrap();
    assert_eq!(profile.description, "Heavy computation workload");
    
    // Test 3: Add execution history
    let history_id = store.add_history(
        workload_id.clone(),
        "exec-123".to_string(),
        "scx_bpfland".to_string(),
        vec!["--slice-us".to_string(), "20000".to_string()],
        "Good performance with low latency".to_string(),
    );
    assert!(!history_id.is_empty());
    
    // Test 4: Get history
    let history = store.get_history_by_workload(&workload_id);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].scheduler_name, "scx_bpfland");
    assert_eq!(history[0].result_description, "Good performance with low latency");
    
    // Test 5: Save to storage
    storage.save(&store)?;
    assert!(storage_path.exists());
    
    // Test 6: Load from storage
    let loaded_store = storage.load()?;
    assert_eq!(loaded_store.profiles.len(), 1);
    assert_eq!(loaded_store.history.len(), 1);
    
    // Test 7: Update profile
    let mut store2 = loaded_store;
    store2.update_profile(&workload_id, "Updated workload description".to_string())?;
    let updated_profile = store2.get_profile(&workload_id).unwrap();
    assert_eq!(updated_profile.description, "Updated workload description");
    
    Ok(())
}

#[tokio::test]
async fn test_multiple_workload_profiles() -> Result<()> {
    let mut store = WorkloadStore::new();
    
    // Create multiple profiles
    let profile1 = store.create_profile("Web server workload".to_string());
    let profile2 = store.create_profile("Database workload".to_string());
    let profile3 = store.create_profile("ML training workload".to_string());
    
    // Add history to different profiles
    store.add_history(
        profile1.clone(),
        "exec-001".to_string(),
        "scx_lavd".to_string(),
        vec!["--performance".to_string()],
        "Low latency for web requests".to_string(),
    );
    
    store.add_history(
        profile2.clone(),
        "exec-002".to_string(),
        "scx_rusty".to_string(),
        vec!["--fifo".to_string()],
        "Good throughput for database operations".to_string(),
    );
    
    store.add_history(
        profile2.clone(),
        "exec-003".to_string(),
        "scx_layered".to_string(),
        vec!["--config".to_string(), "db.json".to_string()],
        "Even better with custom layer config".to_string(),
    );
    
    // Test listing profiles
    let profiles = store.list_profiles();
    assert_eq!(profiles.len(), 3);
    
    // Test getting history by workload
    let db_history = store.get_history_by_workload(&profile2);
    assert_eq!(db_history.len(), 2);
    
    // Test getting history by execution
    let exec_history = store.get_history_by_execution("exec-002");
    assert!(exec_history.is_some());
    assert_eq!(exec_history.unwrap().scheduler_name, "scx_rusty");
    
    Ok(())
}

#[tokio::test]
async fn test_workload_profile_deletion() -> Result<()> {
    let mut store = WorkloadStore::new();
    
    // Create profile and add history
    let profile_id = store.create_profile("Test workload".to_string());
    store.add_history(
        profile_id.clone(),
        "exec-123".to_string(),
        "scx_simple".to_string(),
        vec![],
        "Basic test".to_string(),
    );
    
    assert_eq!(store.profiles.len(), 1);
    assert_eq!(store.history.len(), 1);
    
    // Delete profile
    store.delete_profile(&profile_id)?;
    
    // Verify profile and history are deleted
    assert_eq!(store.profiles.len(), 0);
    assert_eq!(store.history.len(), 0);
    
    Ok(())
}

#[tokio::test]
async fn test_storage_persistence() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let storage_path = temp_dir.path().join("persistence_test.json");
    
    // Create and save data
    {
        let storage = PersistentStorage::with_path(&storage_path);
        let mut store = WorkloadStore::new();
        
        let profile_id = store.create_profile("Persistent workload".to_string());
        store.add_history(
            profile_id,
            "exec-999".to_string(),
            "scx_flash".to_string(),
            vec!["--tickless".to_string()],
            "Testing persistence".to_string(),
        );
        
        storage.save(&store)?;
    }
    
    // Load data in new scope
    {
        let storage = PersistentStorage::with_path(&storage_path);
        let loaded_store = storage.load()?;
        
        assert_eq!(loaded_store.profiles.len(), 1);
        assert_eq!(loaded_store.history.len(), 1);
        
        let profile = loaded_store.list_profiles()[0];
        assert_eq!(profile.description, "Persistent workload");
        
        let history = &loaded_store.history[0];
        assert_eq!(history.scheduler_name, "scx_flash");
        assert_eq!(history.result_description, "Testing persistence");
    }
    
    Ok(())
}