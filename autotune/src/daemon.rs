use std::process::{Command, Output};
use std::time::Instant;

/// Structure to hold command execution results
#[derive(Debug)]
pub struct CommandResult {
    pub command: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration: std::time::Duration,
}

/// Run a command in the specified directory
pub fn run_command(cmd: &str, pwd: &str) -> Result<CommandResult, String> {
    let start = Instant::now();
    
    let output = match Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(pwd)
        .output() {
            Ok(out) => out,
            Err(e) => {
                return Err(format!("Error executing command: {}", e));
            }
        };
    
    let duration = start.elapsed();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    
    Ok(CommandResult {
        command: cmd.to_string(),
        exit_code: output.status.code().unwrap_or(-1),
        stdout,
        stderr,
        duration,
    })
}

/// Get optimization suggestions from Claude for a command result
pub fn get_optimization_suggestions(result: &CommandResult) -> Result<String, String> {
    // Prepare prompt for Claude
    let prompt = crate::prompt::create_optimization_prompt(
        &result.command,
        result.duration,
        result.exit_code,
        &result.stdout,
        &result.stderr,
    );
    
    // Check if Claude CLI is available
    let claude_check = Command::new("which")
        .arg("claude")
        .output();
    
    match claude_check {
        Ok(check_output) => {
            if !check_output.status.success() {
                return Err(String::from("Error: Claude CLI not found in PATH. Please install Claude CLI or check your PATH."));
            }
        },
        Err(e) => {
            return Err(format!("Error checking for Claude CLI: {}", e));
        }
    }
    
    // Call Claude CLI with detailed error handling
    let claude_output = match call_claude_with_prompt(&prompt, false) {
        Ok(out) => out,
        Err(e) => {
            return Err(format!("Error running Claude: {}", e));
        }
    };
    
    if claude_output.status.success() {
        Ok(String::from_utf8_lossy(&claude_output.stdout).into_owned())
    } else {
        let error_text = String::from_utf8_lossy(&claude_output.stderr).into_owned();
        Err(format!("Claude error: {}", error_text))
    }
}

/// Start an interactive session with Claude based on command result
pub fn start_interactive_claude(result: &CommandResult) -> Result<(), String> {
    // Prepare prompt for Claude
    let prompt = crate::prompt::create_optimization_prompt(
        &result.command,
        result.duration,
        result.exit_code,
        &result.stdout,
        &result.stderr,
    );
    
    // Check if Claude CLI is available
    let claude_check = Command::new("which")
        .arg("claude")
        .output();
    
    match claude_check {
        Ok(check_output) => {
            if !check_output.status.success() {
                return Err(String::from("Error: Claude CLI not found in PATH. Please install Claude CLI or check your PATH."));
            }
        },
        Err(e) => {
            return Err(format!("Error checking for Claude CLI: {}", e));
        }
    }
    
    // Call Claude CLI in interactive mode
    match call_claude_with_prompt(&prompt, true) {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Error running interactive Claude: {}", e))
    }
}

/// Run a command and get optimization suggestions (for backward compatibility)
pub fn run_command_with_optimization(cmd: &str, pwd: &str) -> (String, String) {
    // Run the command
    let result = match run_command(cmd, pwd) {
        Ok(res) => res,
        Err(e) => {
            return (String::new(), e);
        }
    };
    
    // Create prompt
    let prompt = crate::prompt::create_optimization_prompt(
        &result.command,
        result.duration,
        result.exit_code,
        &result.stdout,
        &result.stderr,
    );
    
    // Get optimization suggestions
    match get_optimization_suggestions(&result) {
        Ok(suggestions) => (prompt, suggestions),
        Err(e) => (prompt, e),
    }
}

/// Helper function to call Claude CLI with a prompt
/// If interactive is true, will connect stdin/stdout/stderr for direct interaction
fn call_claude_with_prompt(prompt: &str, interactive: bool) -> Result<Output, std::io::Error> {
    // Use a direct path to Claude CLI that we know works
    let claude_path = "/usr/local/bin/claude";
    
    if interactive {
        println!("Starting interactive session with Claude...");
        println!("Initial prompt sent to Claude:\n{}", prompt);
        println!("----------------------------------------");
        
        // Create an interactive process with stdio inheritance
        let mut command = Command::new(claude_path);
        command.arg(prompt);  // In interactive mode, don't use --print
        
        // Inherit stdio for interactive use
        command.stdin(std::process::Stdio::inherit())
               .stdout(std::process::Stdio::inherit())
               .stderr(std::process::Stdio::inherit());
        
        // Execute and wait for completion
        let status = command.status()?;
        
        // Create a dummy output for the return value
        return Ok(Output {
            status,
            stdout: Vec::new(),
            stderr: Vec::new(),
        });
    } else {
        // Non-interactive mode (original behavior)
        println!("Executing: {} --print \"<prompt>\"", claude_path);
        println!("Prompt sent to Claude:\n{}", prompt);
        
        // Call Claude directly with the prompt as an argument
        let output = Command::new(claude_path)
            .arg("--print")  // Non-interactive mode
            .arg(prompt)
            .output();
        
        // Return the output or error
        match output {
            Ok(out) => {
                println!("Claude command completed with status: {}", out.status);
                if !out.status.success() {
                    println!("Claude stderr: {}", String::from_utf8_lossy(&out.stderr));
                } else {
                    println!("Claude stdout length: {}", out.stdout.len());
                }
                Ok(out)
            },
            Err(e) => {
                println!("Failed to execute Claude command: {}", e);
                Err(e)
            }
        }
    }
}
