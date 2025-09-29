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
    format!("For the command '{}', optimize the scheduler with the schedcp mcp tools. When optimize, make sure run the exact command user wants, do not change or break it down. You should analyze the workload, create a profile include the exact full command and workload user wants, and the analysis of the workload, list the schedulers, and test the default and different other schedulers available, update history after each test and summary for the best one.", command)
}