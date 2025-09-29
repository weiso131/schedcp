use std::env;
use std::path::PathBuf;
use tokio::net::UnixStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use clap::{Parser, Subcommand};
use tokio::net::UnixListener;
use autotune::prompt;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a command and get optimization suggestions (former test-daemon)
    Run {
        /// Run in interactive mode with Claude
        #[arg(short, long)]
        interactive: bool,
        
        /// The command to run
        #[arg(required = true, num_args = 1.., allow_hyphen_values = true)]
        command: Vec<String>,
    },
    
    /// Start an interactive Claude session for command optimization
    Cc {
        /// The command to run
        #[arg(required = true, num_args = 1.., allow_hyphen_values = true)]
        command: Vec<String>,
    },
    
    /// Submit a command to the daemon for optimization
    Submit {
        /// Run in interactive mode with Claude
        #[arg(short, long)]
        interactive: bool,
        
        /// The command to run
        #[arg(required = true, num_args = 1.., allow_hyphen_values = true)]
        command: Vec<String>,
    },
    
    /// Start the autotune daemon
    Daemon {
        /// Socket path for the daemon
        #[arg(short, long, default_value = "/tmp/autotune.sock")]
        socket: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Run { interactive, command } => {
            run_command(command.join(" ").as_str(), *interactive).await?
        },
        Commands::Cc { command } => {
            run_cc_command(command.join(" ").as_str()).await?
        },
        Commands::Submit { interactive, command } => {
            submit_to_daemon(command.join(" ").as_str(), *interactive).await?
        },
        Commands::Daemon { socket } => {
            start_daemon(socket).await?
        },
    }
    
    Ok(())
}

/// Run a command locally and get optimization suggestions
async fn run_command(command: &str, interactive: bool) -> Result<(), Box<dyn std::error::Error>> {
    let pwd = env::current_dir()?.to_string_lossy().into_owned();
    
    println!("Running command: {} in {}", command, pwd);
    
    // First run the command
    let result = match autotune::daemon::run_command(command, &pwd) {
        Ok(res) => {
            println!("\nCommand executed successfully:");
            println!("Exit code: {}", res.exit_code);
            println!("Duration: {:?}", res.duration);
            println!("Stdout:\n{}", res.stdout);
            if !res.stderr.is_empty() {
                println!("Stderr:\n{}", res.stderr);
            }
            res
        },
        Err(e) => {
            eprintln!("Error running command: {}", e);
            std::process::exit(1);
        }
    };
    
    if interactive {
        // Start interactive mode with Claude
        println!("\nStarting interactive session with Claude...");
        match autotune::daemon::start_interactive_claude(&result) {
            Ok(_) => {
                println!("\nInteractive session ended.");
            },
            Err(e) => {
                eprintln!("Error starting interactive session: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Automatically get optimization suggestions
        println!("\nGetting optimization suggestions...");
        
        // Get optimization suggestions
        match autotune::daemon::get_optimization_suggestions(&result) {
            Ok(suggestions) => {
                println!("\nOptimization suggestions:\n{}", suggestions);
            },
            Err(e) => {
                eprintln!("Error getting optimization suggestions: {}", e);
            }
        }
    }
    
    Ok(())
}

/// Submit a command to the daemon for optimization
async fn submit_to_daemon(command: &str, interactive: bool) -> Result<(), Box<dyn std::error::Error>> {
    let pwd = env::current_dir()?.to_string_lossy().into_owned();
    
    let mut stream = match UnixStream::connect("/tmp/autotune.sock").await {
        Ok(stream) => stream,
        Err(_) => {
            eprintln!("Error: Daemon not running. Start it with 'autotune daemon'");
            std::process::exit(1);
        }
    };
    
    // Add interactive flag to request if needed
    let request = if interactive {
        format!("{}\n{}\ninteractive", pwd, command)
    } else {
        format!("{}\n{}", pwd, command)
    };
    
    stream.write_all(request.as_bytes()).await?;
    stream.shutdown().await?;
    
    let mut response = String::new();
    stream.read_to_string(&mut response).await?;
    
    println!("{}", response);
    Ok(())
}

/// Run a command with the cc subcommand (special prompt for scheduler optimization)
async fn run_cc_command(command: &str) -> Result<(), Box<dyn std::error::Error>> {
    let pwd = env::current_dir()?.to_string_lossy().into_owned();
    
    println!("Command: {} in {}", command, pwd);
    println!("Starting interactive session with Claude for scheduler optimization...");
    
    // Create the special prompt for Claude
    let cc_prompt = prompt::create_cc_prompt(command);
    
    // Call Claude directly with the custom prompt using the daemon's function
    match autotune::daemon::call_claude_with_prompt(&cc_prompt, true) {
        Ok(_) => {
            println!("\nInteractive session ended.");
        },
        Err(e) => {
            eprintln!("Error starting interactive session: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

/// Start the autotune daemon
async fn start_daemon(socket_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let socket_path_str = socket_path.to_string_lossy();
    let _ = std::fs::remove_file(&socket_path);
    
    let listener = UnixListener::bind(&socket_path)?;
    println!("Autotune daemon listening on {}", socket_path_str);
    
    loop {
        let (mut stream, _) = listener.accept().await?;
        
        tokio::spawn(async move {
            let mut buf = vec![0; 4096];
            let n = stream.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..n]);
            
            // Split request into parts (pwd, cmd, optional flags)
            let lines: Vec<&str> = request.trim().split('\n').collect();
            if lines.len() < 2 {
                let _ = stream.write_all(b"ERROR: Invalid request").await;
                return;
            }
            
            let pwd = lines[0];
            let cmd = lines[1];
            
            // Check if interactive mode is requested
            let interactive = lines.len() > 2 && lines[2] == "interactive";
            
            println!("Running: {} in {} (interactive: {})", cmd, pwd, interactive);
            
            // First run the command
            let result = match autotune::daemon::run_command(cmd, pwd) {
                Ok(res) => res,
                Err(e) => {
                    let _ = stream.write_all(e.as_bytes()).await;
                    return;
                }
            };
            
            // Get response based on mode
            let response = if interactive {
                // Start interactive Claude session
                match autotune::daemon::start_interactive_claude(&result) {
                    Ok(_) => "Interactive session completed".to_string(),
                    Err(e) => format!("Error in interactive session: {}", e),
                }
            } else {
                // Get optimization suggestions
                match autotune::daemon::get_optimization_suggestions(&result) {
                    Ok(suggestions) => suggestions,
                    Err(e) => e,
                }
            };
            
            let _ = stream.write_all(response.as_bytes()).await;
        });
    }
}