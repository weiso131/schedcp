use anyhow::Result;
use std::fs;
use std::process::Command;

#[derive(Debug, Clone)]
pub struct SystemSpec {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub memory_total_gb: f64,
    pub os_name: String,
    pub kernel_version: String,
    pub architecture: String,
}

impl SystemSpec {
    /// Collect system hardware and OS information
    pub fn collect() -> Result<Self> {
        Ok(SystemSpec {
            cpu_model: get_cpu_model()?,
            cpu_cores: get_cpu_cores()?,
            cpu_threads: get_cpu_threads()?,
            memory_total_gb: get_memory_total_gb()?,
            os_name: get_os_name()?,
            kernel_version: get_kernel_version()?,
            architecture: get_architecture()?,
        })
    }

    /// Generate a natural language description of the system
    pub fn to_prompt(&self) -> String {
        format!(
            "System Specifications:\n\
            - CPU: {} ({} cores, {} threads)\n\
            - Memory: {:.1} GB\n\
            - OS: {}\n\
            - Kernel: {}\n\
            - Architecture: {}",
            self.cpu_model,
            self.cpu_cores,
            self.cpu_threads,
            self.memory_total_gb,
            self.os_name,
            self.kernel_version,
            self.architecture
        )
    }
}

fn get_cpu_model() -> Result<String> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
    for line in cpuinfo.lines() {
        if line.starts_with("model name") {
            if let Some(model) = line.split(':').nth(1) {
                return Ok(model.trim().to_string());
            }
        }
    }
    Ok("Unknown CPU".to_string())
}

fn get_cpu_cores() -> Result<usize> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
    let mut cores = 0;
    for line in cpuinfo.lines() {
        if line.starts_with("cpu cores") {
            if let Some(count) = line.split(':').nth(1) {
                cores = count.trim().parse().unwrap_or(0);
                break;
            }
        }
    }
    if cores == 0 {
        // Fallback: count processor entries
        cores = cpuinfo.lines().filter(|l| l.starts_with("processor")).count();
    }
    Ok(cores)
}

fn get_cpu_threads() -> Result<usize> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
    let threads = cpuinfo.lines().filter(|l| l.starts_with("processor")).count();
    Ok(threads)
}

fn get_memory_total_gb() -> Result<f64> {
    let meminfo = fs::read_to_string("/proc/meminfo")?;
    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: u64 = parts[1].parse().unwrap_or(0);
                return Ok(kb as f64 / 1024.0 / 1024.0);
            }
        }
    }
    Ok(0.0)
}

fn get_os_name() -> Result<String> {
    if let Ok(content) = fs::read_to_string("/etc/os-release") {
        for line in content.lines() {
            if line.starts_with("PRETTY_NAME=") {
                let name = line
                    .trim_start_matches("PRETTY_NAME=")
                    .trim_matches('"')
                    .to_string();
                return Ok(name);
            }
        }
    }
    Ok("Linux".to_string())
}

fn get_kernel_version() -> Result<String> {
    let output = Command::new("uname").arg("-r").output()?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn get_architecture() -> Result<String> {
    let output = Command::new("uname").arg("-m").output()?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Public function to get system specifications as a prompt string
pub fn get_system_spec_prompt() -> String {
    match SystemSpec::collect() {
        Ok(spec) => spec.to_prompt(),
        Err(e) => format!("Unable to collect system information: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_spec_collect() {
        let spec = SystemSpec::collect();
        assert!(spec.is_ok(), "Should be able to collect system specs");

        let spec = spec.unwrap();
        assert!(!spec.cpu_model.is_empty(), "CPU model should not be empty");
        assert!(spec.cpu_cores > 0, "CPU cores should be greater than 0");
        assert!(spec.cpu_threads > 0, "CPU threads should be greater than 0");
        assert!(spec.memory_total_gb > 0.0, "Memory should be greater than 0");
        assert!(!spec.os_name.is_empty(), "OS name should not be empty");
        assert!(!spec.kernel_version.is_empty(), "Kernel version should not be empty");
        assert!(!spec.architecture.is_empty(), "Architecture should not be empty");
    }

    #[test]
    fn test_system_spec_to_prompt() {
        let spec = SystemSpec {
            cpu_model: "Test CPU".to_string(),
            cpu_cores: 8,
            cpu_threads: 16,
            memory_total_gb: 32.0,
            os_name: "Test OS".to_string(),
            kernel_version: "6.12.0".to_string(),
            architecture: "x86_64".to_string(),
        };

        let prompt = spec.to_prompt();
        assert!(prompt.contains("Test CPU"));
        assert!(prompt.contains("8 cores"));
        assert!(prompt.contains("16 threads"));
        assert!(prompt.contains("32.0 GB"));
        assert!(prompt.contains("Test OS"));
        assert!(prompt.contains("6.12.0"));
        assert!(prompt.contains("x86_64"));
    }

    #[test]
    fn test_get_system_spec_prompt() {
        let prompt = get_system_spec_prompt();
        assert!(!prompt.is_empty(), "Prompt should not be empty");
        assert!(
            prompt.contains("System Specifications:") || prompt.contains("Unable to collect"),
            "Prompt should contain expected content"
        );
    }

    #[test]
    fn test_get_cpu_model() {
        let model = get_cpu_model();
        assert!(model.is_ok(), "Should be able to get CPU model");
        assert!(!model.unwrap().is_empty());
    }

    #[test]
    fn test_get_cpu_cores() {
        let cores = get_cpu_cores();
        assert!(cores.is_ok(), "Should be able to get CPU cores");
        assert!(cores.unwrap() > 0);
    }

    #[test]
    fn test_get_cpu_threads() {
        let threads = get_cpu_threads();
        assert!(threads.is_ok(), "Should be able to get CPU threads");
        assert!(threads.unwrap() > 0);
    }

    #[test]
    fn test_get_memory_total_gb() {
        let memory = get_memory_total_gb();
        assert!(memory.is_ok(), "Should be able to get memory");
        assert!(memory.unwrap() > 0.0);
    }

    #[test]
    fn test_get_kernel_version() {
        let version = get_kernel_version();
        assert!(version.is_ok(), "Should be able to get kernel version");
        assert!(!version.unwrap().is_empty());
    }

    #[test]
    fn test_get_architecture() {
        let arch = get_architecture();
        assert!(arch.is_ok(), "Should be able to get architecture");
        assert!(!arch.unwrap().is_empty());
    }
}