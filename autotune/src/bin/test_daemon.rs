use std::env;
use std::process::Command;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: test-daemon <command>");
        std::process::exit(1);
    }
    
    let command = args[1..].join(" ");
    let pwd = env::current_dir().unwrap().to_string_lossy().into_owned();
    
    println!("Would run: {} in {}", command, pwd);
    
    // Run the command and measure time
    let start = Instant::now();
    let output = Command::new("sh")
        .arg("-c")
        .arg(&command)
        .current_dir(&pwd)
        .output()
        .expect("Failed to execute command");
    let duration = start.elapsed();
    
    // Prepare prompt for Claude
    let prompt = format!(
        "The command '{}' took {:?} to execute with exit code {}.\n\
        stdout: {}\n\
        stderr: {}\n\
        Please suggest optimizations for this command.",
        command, duration, output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    
    println!("\n--- Would send to Claude ---\n{}\n--- End prompt ---\n", prompt);
    
    // Call Claude CLI
    let claude_output = Command::new("claude")
        .arg("-q")
        .arg(&prompt)
        .output()
        .expect("Failed to run claude");
    
    if claude_output.status.success() {
        println!("Claude response:\n{}", String::from_utf8_lossy(&claude_output.stdout));
    } else {
        println!("Claude error: {}", String::from_utf8_lossy(&claude_output.stderr));
    }
}