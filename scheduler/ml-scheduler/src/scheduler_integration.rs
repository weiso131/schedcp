use crate::bpf_interface::{TaskContext, MigrationDecision};
use crate::{MLScheduler, MigrationFeatures};
use anyhow::{Result, anyhow};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct SchedulerIntegration {
    ml_scheduler: Arc<Mutex<MLScheduler>>,
}

impl SchedulerIntegration {
    pub fn new(ml_scheduler: Arc<Mutex<MLScheduler>>) -> Self {
        Self { ml_scheduler }
    }

    pub async fn process_migration_request(&self, task: &TaskContext) -> Result<MigrationDecision> {
        let features = MigrationFeatures {
            cpu: task.cpu,
            cpu_idle: task.cpu_idle,
            cpu_not_idle: task.cpu_not_idle,
            src_dom_load: task.src_dom_load,
            dst_dom_load: task.dst_dom_load,
        };

        let scheduler = self.ml_scheduler.lock().await;
        let should_migrate = scheduler.should_migrate(&features)
            .map_err(|e| anyhow!("ML prediction failed: {}", e))?;

        Ok(MigrationDecision {
            should_migrate,
            target_dom: if should_migrate { 
                // Simple logic: migrate to next domain
                (task.dom_id + 1) % 4 
            } else { 
                task.dom_id 
            },
            confidence: if should_migrate { 0.8 } else { 0.2 },
        })
    }

    pub fn update_model_metrics(&self, decision: &MigrationDecision, actual_improvement: f64) {
        // In a real implementation, this would collect metrics for model retraining
        log::debug!(
            "Migration decision - Should migrate: {}, Confidence: {:.2}, Actual improvement: {:.2}%",
            decision.should_migrate,
            decision.confidence,
            actual_improvement * 100.0
        );
    }
}