use anyhow::Result;
use clap::{Parser, Subcommand};
use std::env;

mod scheduler_manager;
use scheduler_manager::SchedulerManager;

#[derive(Parser)]
#[command(name = "schedcp-cli")]
#[command(about = "SchedCP CLI - Run and manage sched-ext schedulers", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List schedulers with detailed information
    List {
        /// Filter by scheduler name
        #[arg(short, long)]
        name: Option<String>,
        /// Show only production-ready schedulers
        #[arg(short, long)]
        production: bool,
    },
    /// Run a scheduler
    Run {
        /// Name of the scheduler to run
        scheduler: String,
        /// Additional arguments to pass to the scheduler
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
        /// Use sudo to run the scheduler (will prompt for password via SCHEDCP_SUDO_PASSWORD env var)
        #[arg(short, long)]
        sudo: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut manager = SchedulerManager::new()?;

    match cli.command {
        Commands::List { name, production } => {
            let schedulers = manager.list_schedulers();
            
            for scheduler in schedulers {
                // Apply filters
                if production && !scheduler.production_ready {
                    continue;
                }
                if let Some(ref filter_name) = name {
                    if !scheduler.name.contains(filter_name) {
                        continue;
                    }
                }
                
                // Print detailed information for each scheduler
                manager.print_scheduler_info(scheduler);
            }
        }
        
        Commands::Run { scheduler, args, sudo } => {
            // Extract schedulers to temporary directory
            manager.extract_schedulers().await?;
            
            // Get sudo password from environment
            // If --sudo flag is set or no flag (default), use sudo
            // Empty string means passwordless sudo
            let sudo_password = if sudo {
                env::var("SCHEDCP_SUDO_PASSWORD").ok()
            } else {
                // Even without --sudo flag, still use passwordless sudo for schedulers
                Some("".to_string())
            };
            
            // Run the scheduler
            manager.run_scheduler(&scheduler, args, sudo_password.as_deref()).await?;
        }
    }

    Ok(())
}