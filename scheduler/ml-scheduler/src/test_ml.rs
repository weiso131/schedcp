use ml_scheduler::ml_scheduler_exact::{MLLoadBalancer, TaskCtx};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing ML Scheduler (Exact Implementation)");
    
    // Initialize the ML scheduler with the model path
    let model_path = "src/model_dir/model_path";
    println!("Loading model from: {}", model_path);
    
    let ml_balancer = MLLoadBalancer::new(model_path)?;
    println!("Model loaded successfully!");
    
    // Test case 1: High source load, low destination load (should migrate)
    let task1 = TaskCtx {
        pid: 1234,
        dom_id: 0,
        cpu: 32,
        cpu_idle: 16,
        cpu_not_idle: 16,
        src_dom_load: 80,  // 80% load
        dst_dom_load: 20,  // 20% load
    };
    
    let should_migrate1 = ml_balancer.should_migrate_task(&task1);
    println!("\nTest 1 - High load imbalance:");
    println!("  Source domain load: {}%", task1.src_dom_load);
    println!("  Destination domain load: {}%", task1.dst_dom_load);
    println!("  Migration decision: {}", should_migrate1);
    
    // Test case 2: Balanced load (should not migrate)
    let task2 = TaskCtx {
        pid: 5678,
        dom_id: 1,
        cpu: 32,
        cpu_idle: 16,
        cpu_not_idle: 16,
        src_dom_load: 50,  // 50% load
        dst_dom_load: 48,  // 48% load
    };
    
    let should_migrate2 = ml_balancer.should_migrate_task(&task2);
    println!("\nTest 2 - Balanced load:");
    println!("  Source domain load: {}%", task2.src_dom_load);
    println!("  Destination domain load: {}%", task2.dst_dom_load);
    println!("  Migration decision: {}", should_migrate2);
    
    // Test the raw migrate_inference function
    println!("\nTest 3 - Raw inference function:");
    let cpu = 64;
    let cpu_idle = 32;
    let cpu_not_idle = 32;
    let src_load = 0.75;
    let dst_load = 0.25;
    
    let migrate = ml_balancer.migrate_inference(&cpu, &cpu_idle, &cpu_not_idle, &src_load, &dst_load);
    println!("  CPU: {}, Idle: {}, Not Idle: {}", cpu, cpu_idle, cpu_not_idle);
    println!("  Source load: {:.2}, Destination load: {:.2}", src_load, dst_load);
    println!("  Migration decision: {}", migrate);
    
    Ok(())
}