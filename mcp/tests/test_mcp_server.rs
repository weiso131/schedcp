use schedcp::{SchedMcpServer, ListSchedulersRequest, RunSchedulerRequest, GetExecutionStatusRequest};
use tokio;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_server() -> SchedMcpServer {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("test_storage.json");
        
        SchedMcpServer::new(storage_path.to_str().unwrap().to_string())
            .await
            .expect("Failed to create test server")
    }

    #[tokio::test]
    async fn test_server_creation() {
        let server = create_test_server().await;
        // Just ensure server is created successfully
        assert!(server.scheduler_manager.lock().await.list_schedulers().len() > 0);
    }

    #[tokio::test]
    async fn test_list_schedulers() {
        let server = create_test_server().await;
        
        // Test listing all schedulers
        let request = ListSchedulersRequest {
            name: None,
            production_ready: None,
        };
        
        let result = server.list_schedulers(request).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        let schedulers: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(schedulers["schedulers"].is_array());
    }

    #[tokio::test]
    async fn test_list_schedulers_with_filter() {
        let server = create_test_server().await;
        
        // Test with name filter
        let request = ListSchedulersRequest {
            name: Some("rusty".to_string()),
            production_ready: None,
        };
        
        let result = server.list_schedulers(request).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        let schedulers: serde_json::Value = serde_json::from_str(&response).unwrap();
        let sched_array = schedulers["schedulers"].as_array().unwrap();
        
        for scheduler in sched_array {
            assert!(scheduler["name"].as_str().unwrap().contains("rusty"));
        }
    }

    #[tokio::test]
    async fn test_list_schedulers_production_ready() {
        let server = create_test_server().await;
        
        // Test with production_ready filter
        let request = ListSchedulersRequest {
            name: None,
            production_ready: Some(true),
        };
        
        let result = server.list_schedulers(request).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        let schedulers: serde_json::Value = serde_json::from_str(&response).unwrap();
        let sched_array = schedulers["schedulers"].as_array().unwrap();
        
        for scheduler in sched_array {
            assert_eq!(scheduler["production_ready"].as_bool().unwrap(), true);
        }
    }

    #[tokio::test]
    async fn test_run_scheduler_invalid() {
        let server = create_test_server().await;
        
        // Test running non-existent scheduler
        let request = RunSchedulerRequest {
            name: "non_existent_scheduler".to_string(),
            args: vec![],
        };
        
        let result = server.run_scheduler(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_get_execution_status_invalid() {
        let server = create_test_server().await;
        
        // Test getting status of non-existent execution
        let request = GetExecutionStatusRequest {
            execution_id: "invalid_id".to_string(),
        };
        
        let result = server.get_execution_status(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_create_workload_profile() {
        let server = create_test_server().await;
        
        // Test creating a workload profile
        let request = serde_json::json!({
            "description": "Test workload for CPU-intensive tasks"
        });
        
        let result = server.create_workload_profile(
            serde_json::to_string(&request).unwrap()
        ).await;
        
        assert!(result.is_ok());
        let response: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(response["workload_id"].is_string());
        assert!(response["message"].as_str().unwrap().contains("created"));
    }

    #[tokio::test]
    async fn test_list_workload_profiles() {
        let server = create_test_server().await;
        
        // Create a profile first
        let create_request = serde_json::json!({
            "description": "Test workload"
        });
        server.create_workload_profile(
            serde_json::to_string(&create_request).unwrap()
        ).await.unwrap();
        
        // List profiles
        let result = server.list_workload_profiles("{}".to_string()).await;
        assert!(result.is_ok());
        
        let response: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(response["profiles"].is_array());
        assert!(response["profiles"].as_array().unwrap().len() > 0);
    }

    #[tokio::test]
    async fn test_add_execution_history_invalid_workload() {
        let server = create_test_server().await;
        
        // Test adding history to non-existent workload
        let request = serde_json::json!({
            "workload_id": "invalid_id",
            "execution_id": "exec_123",
            "result_description": "Test result"
        });
        
        let result = server.add_execution_history(
            serde_json::to_string(&request).unwrap()
        ).await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_get_workload_history_invalid() {
        let server = create_test_server().await;
        
        // Test getting history of non-existent workload
        let request = serde_json::json!({
            "workload_id": "invalid_id"
        });
        
        let result = server.get_workload_history(
            serde_json::to_string(&request).unwrap()
        ).await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_stop_scheduler_invalid() {
        let server = create_test_server().await;
        
        // Test stopping non-existent scheduler
        let request = serde_json::json!({
            "execution_id": "invalid_id"
        });
        
        let result = server.stop_scheduler(
            serde_json::to_string(&request).unwrap()
        ).await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_workload_profile_lifecycle() {
        let server = create_test_server().await;
        
        // Create a workload profile
        let create_request = serde_json::json!({
            "description": "Comprehensive test workload"
        });
        
        let create_result = server.create_workload_profile(
            serde_json::to_string(&create_request).unwrap()
        ).await.unwrap();
        
        let create_response: serde_json::Value = serde_json::from_str(&create_result).unwrap();
        let workload_id = create_response["workload_id"].as_str().unwrap();
        
        // Get workload history (should be empty)
        let history_request = serde_json::json!({
            "workload_id": workload_id
        });
        
        let history_result = server.get_workload_history(
            serde_json::to_string(&history_request).unwrap()
        ).await.unwrap();
        
        let history_response: serde_json::Value = serde_json::from_str(&history_result).unwrap();
        assert_eq!(history_response["history"].as_array().unwrap().len(), 0);
        assert_eq!(history_response["profile"]["description"].as_str().unwrap(), "Comprehensive test workload");
    }

    #[tokio::test]
    async fn test_scheduler_info_details() {
        let server = create_test_server().await;
        
        // Test that scheduler info contains all expected fields
        let request = ListSchedulersRequest {
            name: None,
            production_ready: None,
        };
        
        let result = server.list_schedulers(request).await.unwrap();
        let response: serde_json::Value = serde_json::from_str(&result).unwrap();
        let schedulers = response["schedulers"].as_array().unwrap();
        
        assert!(schedulers.len() > 0);
        
        // Check first scheduler has all required fields
        let first = &schedulers[0];
        assert!(first["name"].is_string());
        assert!(first["production_ready"].is_boolean());
        assert!(first["description"].is_string());
        assert!(first["algorithm"].is_string());
        assert!(first["use_cases"].is_array());
        assert!(first["characteristics"].is_string());
        assert!(first["tuning_parameters"].is_object());
        assert!(first["limitations"].is_string());
        assert!(first["performance_profile"].is_string());
    }
}