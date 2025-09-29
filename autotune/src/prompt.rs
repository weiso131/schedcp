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