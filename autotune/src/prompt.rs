use std::time::Duration;

pub fn create_optimization_prompt(
    command: &str,
    duration: Duration,
    exit_code: i32,
    stdout: &str,
    stderr: &str,
) -> String {
    format!(
        "The command '{}' took {:?} to execute with exit code {}.\n\
        stdout: {}\n\
        stderr: {}\n\
        Please suggest optimizations for this command.",
        command, duration, exit_code, stdout, stderr
    )
}

/// Creates a prompt specifically for the cc subcommand
pub fn create_cc_prompt(command: &str) -> String {
    format!("For the command '{}', optimize the scheduler with the schedcp mcp tools.", command)
}