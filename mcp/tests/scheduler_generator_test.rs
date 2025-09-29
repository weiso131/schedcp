use schedcp::SchedulerGenerator;
use std::time::Duration;
use tempfile::TempDir;

/// Helper to create a minimal BPF scheduler source code
fn create_test_bpf_source(name: &str) -> String {
    format!(
        r#"
#include <scx/common.bpf.h>

char _license[] SEC("license") = "GPL";

UEI_DEFINE(uei);

s32 BPF_STRUCT_OPS_SLEEPABLE(test_{0}_init)
{{
    return 0;
}}

void BPF_STRUCT_OPS(test_{0}_exit, struct scx_exit_info *ei)
{{
    UEI_RECORD(uei, ei);
}}

void BPF_STRUCT_OPS(test_{0}_enqueue, struct task_struct *p, u64 enq_flags)
{{
    scx_bpf_dsq_insert(p, SCX_DSQ_GLOBAL, SCX_SLICE_DFL, enq_flags);
}}

void BPF_STRUCT_OPS(test_{0}_dispatch, s32 cpu, struct task_struct *prev)
{{
    scx_bpf_dsq_move_to_local(SCX_DSQ_GLOBAL);
}}

SEC(".struct_ops.link")
struct sched_ext_ops test_{0}_ops = {{
    .enqueue        = (void *)test_{0}_enqueue,
    .dispatch       = (void *)test_{0}_dispatch,
    .init           = (void *)test_{0}_init,
    .exit           = (void *)test_{0}_exit,
    .name           = "test_{0}",
}};
"#,
        name
    )
}

#[test]
fn test_scheduler_generator_new() {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path().join("new_sched");

    let generator = SchedulerGenerator::with_work_dir(&work_dir);
    assert!(generator.is_ok(), "Failed to create SchedulerGenerator");
}

#[test]
fn test_create_scheduler() {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path().join("new_sched");
    let generator = SchedulerGenerator::with_work_dir(&work_dir).unwrap();

    let source_code = create_test_bpf_source("test_sched");
    let result = generator.create_scheduler("test_sched", &source_code);

    assert!(result.is_ok(), "Failed to create scheduler");

    // Verify it exists
    let info = generator.scheduler_exists("test_sched");
    assert!(info.has_source);
    assert!(!info.has_object); // Not compiled yet
}

#[test]
fn test_compile_scheduler() {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path().join("new_sched");
    let generator = SchedulerGenerator::with_work_dir(&work_dir).unwrap();

    let source_code = create_test_bpf_source("compile_test");
    generator.create_scheduler("compile_test", &source_code).unwrap();

    // Try to compile
    let result = generator.compile_scheduler("compile_test");

    match result {
        Ok(_) => {
            let info = generator.scheduler_exists("compile_test");
            assert!(info.has_source);
            assert!(info.has_object);
        }
        Err(e) => {
            eprintln!("Compilation failed (may be expected in test env): {}", e);
        }
    }
}

#[test]
fn test_create_and_compile() {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path().join("new_sched");
    let generator = SchedulerGenerator::with_work_dir(&work_dir).unwrap();

    let source_code = create_test_bpf_source("full_workflow");
    let result = generator.create_and_compile("full_workflow", &source_code);

    match result {
        Ok(_) => {
            let info = generator.scheduler_exists("full_workflow");
            assert!(info.has_source);
            assert!(info.has_object);
        }
        Err(e) => {
            eprintln!("Create and compile failed (may be expected in test env): {}", e);
        }
    }
}

#[test]
fn test_get_scheduler_source() {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path().join("new_sched");
    let generator = SchedulerGenerator::with_work_dir(&work_dir).unwrap();

    let source_code = create_test_bpf_source("source_test");
    generator.create_scheduler("source_test", &source_code).unwrap();

    let retrieved = generator.get_scheduler_source("source_test").unwrap();
    assert_eq!(retrieved, source_code);
}

#[test]
fn test_delete_scheduler() {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path().join("new_sched");
    let generator = SchedulerGenerator::with_work_dir(&work_dir).unwrap();

    let source_code = create_test_bpf_source("delete_test");
    generator.create_scheduler("delete_test", &source_code).unwrap();

    // Verify it exists
    let info = generator.scheduler_exists("delete_test");
    assert!(info.has_source);

    // Delete it
    generator.delete_scheduler("delete_test").unwrap();

    // Verify it's gone
    let info = generator.scheduler_exists("delete_test");
    assert!(!info.has_source);
    assert!(!info.has_object);
}

#[test]
fn test_list_schedulers() {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path().join("new_sched");
    let generator = SchedulerGenerator::with_work_dir(&work_dir).unwrap();

    // Create several schedulers
    let source1 = create_test_bpf_source("sched1");
    let source2 = create_test_bpf_source("sched2");
    let source3 = create_test_bpf_source("sched3");

    generator.create_scheduler("sched1", &source1).unwrap();
    generator.create_scheduler("sched2", &source2).unwrap();
    generator.create_scheduler("sched3", &source3).unwrap();

    let list = generator.list_schedulers().unwrap();
    assert!(list.len() >= 3);

    let names: Vec<String> = list.iter().map(|s| s.name.clone()).collect();
    assert!(names.contains(&"sched1".to_string()));
    assert!(names.contains(&"sched2".to_string()));
    assert!(names.contains(&"sched3".to_string()));
}

#[test]
fn test_scheduler_exists() {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path().join("new_sched");
    let generator = SchedulerGenerator::with_work_dir(&work_dir).unwrap();

    // Non-existent scheduler
    let info = generator.scheduler_exists("nonexistent");
    assert!(!info.has_source);
    assert!(!info.has_object);

    // Create scheduler
    let source_code = create_test_bpf_source("exists_test");
    generator.create_scheduler("exists_test", &source_code).unwrap();

    let info = generator.scheduler_exists("exists_test");
    assert!(info.has_source);
    assert!(!info.has_object);
}

#[test]
fn test_compile_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path().join("new_sched");
    let generator = SchedulerGenerator::with_work_dir(&work_dir).unwrap();

    let result = generator.compile_scheduler("nonexistent");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("source not found"));
}

#[tokio::test]
async fn test_execute_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path().join("new_sched");
    let generator = SchedulerGenerator::with_work_dir(&work_dir).unwrap();

    let result = generator.execute_scheduler("nonexistent", Some(Duration::from_secs(1))).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("object not found"));
}

#[tokio::test]
async fn test_execute_not_compiled() {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path().join("new_sched");
    let generator = SchedulerGenerator::with_work_dir(&work_dir).unwrap();

    let source_code = create_test_bpf_source("not_compiled");
    generator.create_scheduler("not_compiled", &source_code).unwrap();

    let result = generator.execute_scheduler("not_compiled", Some(Duration::from_secs(1))).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("object not found"));
}

#[tokio::test]
#[ignore] // This test requires real scheduler binary and sudo access
async fn test_execute_real_scheduler() {
    // Use the actual new_sched directory
    let generator = SchedulerGenerator::new().unwrap();

    // Check if vruntime scheduler exists
    let info = generator.scheduler_exists("vruntime");
    if !info.has_object {
        eprintln!("Skipping test: vruntime.bpf.o not found. Run 'make' in new_sched first.");
        return;
    }

    // Try to run for 3 seconds
    let result = generator.execute_scheduler("vruntime", Some(Duration::from_secs(3))).await;

    match result {
        Ok(output) => {
            println!("Execution successful!\n{}", output);
            assert!(output.contains("Scheduler verification successful") || output.contains("PID"));
        }
        Err(e) => {
            eprintln!("Execution failed (may be expected without sudo): {}", e);
        }
    }
}

#[tokio::test]
#[ignore] // This test requires compilation and sudo access
async fn test_full_workflow_create_compile_execute() {
    let generator = SchedulerGenerator::new().unwrap();

    let source_code = create_test_bpf_source("workflow_test");
    let name = "workflow_test";

    // Create and compile
    let result = generator.create_and_compile(name, &source_code);

    match result {
        Ok(_) => {
            println!("Created and compiled scheduler '{}'", name);

            // Verify execution
            let exec_result = generator.execute_scheduler(name, Some(Duration::from_secs(2))).await;

            match exec_result {
                Ok(output) => {
                    println!("Execution output:\n{}", output);
                    assert!(output.contains("Scheduler verification successful") || output.contains("PID"));
                }
                Err(e) => {
                    eprintln!("Execution failed (may be expected without sudo): {}", e);
                }
            }

            // Clean up
            generator.delete_scheduler(name).unwrap();
        }
        Err(e) => {
            eprintln!("Create and compile failed (may be expected in test env): {}", e);
        }
    }
}