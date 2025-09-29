use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: test-daemon <command>");
        std::process::exit(1);
    }
    
    let command = args[1..].join(" ");
    let pwd = env::current_dir().unwrap().to_string_lossy().into_owned();
    
    println!("Would run: {} in {}", command, pwd);
    
    // Use the daemon library to run the command and get optimization suggestions
    let (prompt, response) = autotune::daemon::run_command_with_optimization(&command, &pwd);
    
    println!("\n--- Would send to Claude ---\n{}\n--- End prompt ---\n", prompt);
    
    // Check if response contains "Error" and handle appropriately
    if response.contains("Error") {
        eprintln!("Error: {}", response);
    } else {
        println!("{}", response);
    }
}