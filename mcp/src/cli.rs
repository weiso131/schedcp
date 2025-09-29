use anyhow::Result;
use clap::{Parser, Subcommand};
use std::env;
use std::collections::HashMap;

mod scheduler_manager;
mod scheduler_generator;
use scheduler_manager::{SchedulerManager, SchedulerInfo};

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
    /// Create, compile, and run a custom scheduler from source code
    CreateAndRun {
        /// Path to the BPF C source file (.bpf.c)
        source: String,
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

        Commands::CreateAndRun { source } => {
            // Set sudo password from environment
            let sudo_password = env::var("SCHEDCP_SUDO_PASSWORD").ok();
            if let Some(ref password) = sudo_password {
                manager.set_sudo_password(password.clone());
            }

            // Infer scheduler name from source file (e.g., "my_sched.bpf.c" -> "my_sched")
            let source_path = std::path::Path::new(&source);
            let name = source_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.trim_end_matches(".bpf"))
                .ok_or_else(|| anyhow::anyhow!("Invalid source file name"))?
                .to_string();

            // Read source code from file
            println!("Reading source code from: {}", source);
            let source_code = std::fs::read_to_string(&source)?;

            // Create minimal scheduler info
            let info = SchedulerInfo {
                name: name.clone(),
                production_ready: false,
                description: String::new(),
                use_cases: Vec::new(),
                algorithm: String::new(),
                characteristics: String::new(),
                tuning_parameters: HashMap::new(),
                limitations: String::new(),
                performance_profile: String::new(),
            };

            println!("Creating and compiling custom scheduler: {}", name);

            // Create and compile the scheduler (skip verification, we'll run it directly)
            manager.generator().create_and_compile(&name, &source_code)?;
            println!("Scheduler compiled successfully!");

            println!("\nStarting custom scheduler...");
            let mut child = manager.generator().run_scheduler_process(&name, vec![]).await?;
            println!("Custom scheduler started with PID: {:?}", child.id());

            // Keep the process running until interrupted
            println!("\nPress Ctrl+C to stop the scheduler...");
            tokio::signal::ctrl_c().await?;

            println!("\nStopping scheduler...");
            child.kill().await?;
            println!("Scheduler stopped");
        }
    }

    Ok(())
}