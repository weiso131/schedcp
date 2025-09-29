use std::process::{Command, Output};
use std::time::Instant;

/// Run a command in the specified directory and get optimization suggestions from Claude
pub fn run_command_with_optimization(cmd: &str, pwd: &str) -> (String, String) {
    println!("[DEBUG] Starting command execution: '{}' in directory: '{}'", cmd, pwd);
    
    // Run the command and measure time
    let start = Instant::now();
    let output = match Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(pwd)
        .output() {
            Ok(out) => {
                println!("[DEBUG] Command executed successfully");
                out
            },
            Err(e) => {
                println!("[ERROR] Failed to execute command: {}", e);
                return (String::new(), format!("Error executing command: {}", e));
            }
        };
    let duration = start.elapsed();
    println!("[DEBUG] Command execution time: {:?}", duration);
    
    // Prepare prompt for Claude
    let prompt = crate::prompt::create_optimization_prompt(
        cmd,
        duration,
        output.status.code().unwrap_or(-1),
        &String::from_utf8_lossy(&output.stdout),
        &String::from_utf8_lossy(&output.stderr),
    );
    println!("[DEBUG] Created prompt for Claude");
    
    // Check if Claude CLI is available
    let claude_check = Command::new("which")
        .arg("claude")
        .output();
    
    match claude_check {
        Ok(check_output) => {
            if !check_output.status.success() {
                println!("[ERROR] Claude CLI not found in PATH");
                return (prompt, String::from("Error: Claude CLI not found in PATH. Please install Claude CLI or check your PATH."));
            }
            println!("[DEBUG] Claude CLI found at: {}", String::from_utf8_lossy(&check_output.stdout).trim());
        },
        Err(e) => {
            println!("[ERROR] Failed to check for Claude CLI: {}", e);
            return (prompt, format!("Error checking for Claude CLI: {}", e));
        }
    }
    
    // Call Claude CLI with detailed error handling
    println!("[DEBUG] Calling Claude CLI");
    let claude_output = match call_claude_with_prompt(&prompt) {
        Ok(out) => out,
        Err(e) => {
            println!("[ERROR] Failed to run Claude: {}", e);
            return (prompt, format!("Error running Claude: {}", e));
        }
    };
    
    let response = if claude_output.status.success() {
        let response_text = String::from_utf8_lossy(&claude_output.stdout).into_owned();
        println!("[DEBUG] Claude responded successfully (length: {} bytes)", response_text.len());
        response_text
    } else {
        let error_text = String::from_utf8_lossy(&claude_output.stderr).into_owned();
        println!("[ERROR] Claude returned error: {}", error_text);
        format!("Claude error: {}", error_text)
    };
    
    (prompt, response)
}

/// Helper function to call Claude CLI with a prompt
fn call_claude_with_prompt(prompt: &str) -> Result<Output, std::io::Error> {
    println!("[DEBUG] Setting up Claude command");
    
    // Try to find claude in PATH or common locations
    let claude_paths = vec!["claude", "/usr/bin/claude", "/usr/local/bin/claude", 
                          "~/.local/bin/claude", "~/.cargo/bin/claude"];
    
    let mut claude_cmd = None;
    for path in claude_paths {
        println!("[DEBUG] Trying Claude at: {}", path);
        if let Ok(status) = Command::new(path).arg("--version").status() {
            if status.success() {
                claude_cmd = Some(path);
                println!("[DEBUG] Found working Claude at: {}", path);
                break;
            }
        }
    }
    
    let claude_path = claude_cmd.unwrap_or("claude");
    println!("[DEBUG] Using Claude at: {}", claude_path);
    
    let output = Command::new(claude_path)
        .arg("--print")  // Non-interactive mode
        .arg(prompt)
        .output()?;
    
    println!("[DEBUG] Claude command completed with status: {}", output.status);
    Ok(output)
}
