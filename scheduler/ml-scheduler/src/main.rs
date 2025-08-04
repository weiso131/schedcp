use anyhow::Result;
use clap::Parser;
use log::info;
use ml_scheduler::{MLScheduler, MigrationFeatures};
use simplelog::{Config, LevelFilter, SimpleLogger};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value = "src/model_dir/model_path")]
    model_path: String,
    
    #[clap(short, long, action = clap::ArgAction::SetTrue)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    let log_level = if args.verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };
    
    SimpleLogger::init(log_level, Config::default())?;
    
    info!("Initializing ML Scheduler with model: {}", args.model_path);
    
    let scheduler = Arc::new(Mutex::new(MLScheduler::new(&args.model_path)?));
    
    info!("ML Scheduler started successfully");
    
    // Example usage - in production this would be integrated with the BPF scheduler
    let features = MigrationFeatures {
        cpu: 172,
        cpu_idle: 1,
        cpu_not_idle: 171,
        src_dom_load: 0.5,
        dst_dom_load: 0.5,
    };
    
    let scheduler_guard = scheduler.lock().await;
    match scheduler_guard.should_migrate(&features) {
        Ok(should_migrate) => {
            info!("Migration decision for {:?}: {}", features, should_migrate);
        }
        Err(e) => {
            log::error!("Error making migration decision: {}", e);
        }
    }
    
    // Keep the scheduler running
    let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(1);
    
    ctrlc::set_handler(move || {
        let _ = tx.try_send(());
    })?;
    
    info!("ML Scheduler running. Press Ctrl+C to exit.");
    rx.recv().await;
    
    info!("Shutting down ML Scheduler");
    Ok(())
}