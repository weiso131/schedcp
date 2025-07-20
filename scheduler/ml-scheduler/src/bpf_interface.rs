use std::mem::size_of;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TaskContext {
    pub pid: u32,
    pub cpu: i32,
    pub dom_id: u32,
    pub src_dom_load: f64,
    pub dst_dom_load: f64,
    pub cpu_idle: i32,
    pub cpu_not_idle: i32,
}

impl Default for TaskContext {
    fn default() -> Self {
        Self {
            pid: 0,
            cpu: 0,
            dom_id: 0,
            src_dom_load: 0.0,
            dst_dom_load: 0.0,
            cpu_idle: 0,
            cpu_not_idle: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MigrationDecision {
    pub should_migrate: bool,
    pub target_dom: u32,
    pub confidence: f64,
}

impl Default for MigrationDecision {
    fn default() -> Self {
        Self {
            should_migrate: false,
            target_dom: 0,
            confidence: 0.0,
        }
    }
}

pub const TASK_CONTEXT_SIZE: usize = size_of::<TaskContext>();
pub const MIGRATION_DECISION_SIZE: usize = size_of::<MigrationDecision>();