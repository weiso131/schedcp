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
    /// List all available schedulers
    List {
        /// Show only production-ready schedulers
        #[arg(short, long, default_value = "false")]
        production: bool,
    },
    /// Show detailed information about a scheduler
    Info {
        /// Name of the scheduler
        scheduler: String,
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
        Commands::List { production } => {
            let schedulers = manager.list_schedulers();
            
            println!("Available schedulers:");
            println!("{:<20} {:<20} {}", "Name", "Production Ready", "Description");
            println!("{}", "-".repeat(80));
            
            for scheduler in schedulers {
                if production && !scheduler.production_ready {
                    continue;
                }
                
                let desc_truncated = if scheduler.description.len() > 40 {
                    format!("{}...", &scheduler.description[..37])
                } else {
                    scheduler.description.clone()
                };
                
                println!(
                    "{:<20} {:<20} {}",
                    scheduler.name,
                    if scheduler.production_ready { "Yes" } else { "No" },
                    desc_truncated
                );
            }
        }
        
        Commands::Info { scheduler } => {
            manager.print_scheduler_info(&scheduler)?;
        }
        
        Commands::Run { scheduler, args, sudo } => {
            // Extract schedulers to temporary directory
            manager.extract_schedulers().await?;
            
            // Get sudo password from environment if needed
            let sudo_password = if sudo {
                env::var("SCHEDCP_SUDO_PASSWORD").ok()
            } else {
                None
            };
            
            // Run the scheduler
            manager.run_scheduler(&scheduler, args, sudo_password.as_deref()).await?;
        }
    }

    Ok(())
}