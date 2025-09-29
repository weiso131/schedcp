use std::time::Duration;
use crate::system_info::get_system_spec_prompt;

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
    let system_spec = get_system_spec_prompt();
    format!(
        "{}\n\n\
        For the command '{}', optimize the scheduler with the schedcp mcp tools. When optimize, make sure run the exact command user wants, do not change or break it down. You should analyze the workload, check the detail like reading code, create a profile include the exact full command and workload user wants, and the analysis of the workload, list the schedulers, and test the default and max 3 different other schedulers (that may be best) available with best configuration, update history for all the available metrics correctly after each test, and summary for the best one. If you know the command generate too many outputs, consider using like 1>/dev/null to make sure you capture the metrics. otherwise check the output of the command. The exact command is '{}'.",
        system_spec, command, command
    )
}