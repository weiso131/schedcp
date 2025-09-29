use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::timeout;

/// Information about a managed scheduler
#[derive(Debug, Clone)]
pub struct SchedulerInfo {
    pub name: String,
    pub has_source: bool,
    pub has_object: bool,
}

/// Manages the generation and compilation of new BPF schedulers
///
/// Provides a name-based API to hide path management complexity.
/// All operations use scheduler names instead of file paths.
pub struct SchedulerGenerator {
    /// Project root directory
    project_root: PathBuf,
    /// Working directory for new schedulers (typically mcp/new_sched)
    work_dir: PathBuf,
}

impl SchedulerGenerator {
    /// Create a new SchedulerGenerator with default work directory
    pub fn new() -> Result<Self> {
        // Get project root using git
        let output = Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .output()
            .context("Failed to get git project root")?;

        let project_root = PathBuf::from(
            String::from_utf8(output.stdout)
                .context("Invalid UTF-8 in project root path")?
                .trim()
        );

        let work_dir = project_root.join("mcp/new_sched");

        // Create work directory if it doesn't exist
        fs::create_dir_all(&work_dir)
            .context("Failed to create work directory")?;

        Ok(Self {
            project_root,
            work_dir,
        })
    }

    /// Create a new SchedulerGenerator with custom work directory
    pub fn with_work_dir(work_dir: impl AsRef<Path>) -> Result<Self> {
        let work_dir = work_dir.as_ref().to_path_buf();

        // Get project root using git
        let output = Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .output()
            .context("Failed to get git project root")?;

        let project_root = PathBuf::from(
            String::from_utf8(output.stdout)
                .context("Invalid UTF-8 in project root path")?
                .trim()
        );

        // Create work directory if it doesn't exist
        fs::create_dir_all(&work_dir)
            .context("Failed to create work directory")?;

        Ok(Self {
            project_root,
            work_dir,
        })
    }

    /// Get the path for a scheduler's source file
    fn get_source_path(&self, name: &str) -> PathBuf {
        self.work_dir.join(format!("{}.bpf.c", name))
    }

    /// Get the path for a scheduler's object file
    fn get_object_path(&self, name: &str) -> PathBuf {
        self.work_dir.join(format!("{}.bpf.o", name))
    }

    // ============================================================================
    // Public Name-Based API
    // ============================================================================

    /// Create a new scheduler from source code
    ///
    /// # Arguments
    /// * `name` - Name of the scheduler (without .bpf.c extension)
    /// * `source_code` - The BPF C source code
    ///
    /// # Returns
    /// Ok(()) if the scheduler was created successfully
    pub fn create_scheduler(&self, name: &str, source_code: &str) -> Result<()> {
        let source_path = self.get_source_path(name);

        // Write source code to file
        fs::write(&source_path, source_code)
            .context(format!("Failed to write source file for scheduler '{}'", name))?;

        log::info!("Created scheduler '{}' at: {}", name, source_path.display());
        Ok(())
    }

    /// Compile a scheduler by name
    ///
    /// # Arguments
    /// * `name` - Name of the scheduler (without extension)
    ///
    /// # Returns
    /// Ok(()) if compilation succeeded
    pub fn compile_scheduler(&self, name: &str) -> Result<()> {
        let source_path = self.get_source_path(name);

        if !source_path.exists() {
            anyhow::bail!("Scheduler '{}' source not found. Create it first with create_scheduler()", name);
        }

        self.compile_bpf_scheduler(&source_path)?;
        log::info!("Successfully compiled scheduler '{}'", name);
        Ok(())
    }

    /// Create and compile a scheduler in one step
    ///
    /// # Arguments
    /// * `name` - Name of the scheduler
    /// * `source_code` - The BPF C source code
    ///
    /// # Returns
    /// Ok(()) if creation and compilation succeeded
    pub fn create_and_compile(&self, name: &str, source_code: &str) -> Result<()> {
        self.create_scheduler(name, source_code)?;
        self.compile_scheduler(name)?;
        Ok(())
    }

    /// Execute (verify) a scheduler by name
    ///
    /// Loads and runs the scheduler in the kernel for the specified duration,
    /// then stops it. This verifies the scheduler can load and run.
    ///
    /// # Arguments
    /// * `name` - Name of the scheduler
    /// * `duration` - How long to run (default: 10 seconds)
    ///
    /// # Returns
    /// Ok(String) with execution details if successful
    pub async fn execute_scheduler(&self, name: &str, duration: Option<Duration>) -> Result<String> {
        let object_path = self.get_object_path(name);

        if !object_path.exists() {
            anyhow::bail!("Scheduler '{}' object not found. Compile it first with compile_scheduler()", name);
        }

        self.execution_verify(&object_path, duration).await
    }

    /// Get the source code of a scheduler by name
    ///
    /// # Arguments
    /// * `name` - Name of the scheduler
    ///
    /// # Returns
    /// Ok(String) containing the source code
    pub fn get_scheduler_source(&self, name: &str) -> Result<String> {
        let source_path = self.get_source_path(name);

        if !source_path.exists() {
            anyhow::bail!("Scheduler '{}' source not found", name);
        }

        fs::read_to_string(&source_path)
            .context(format!("Failed to read source for scheduler '{}'", name))
    }

    /// Delete a scheduler by name (removes both source and object files)
    ///
    /// # Arguments
    /// * `name` - Name of the scheduler
    ///
    /// # Returns
    /// Ok(()) if deletion succeeded (or files didn't exist)
    pub fn delete_scheduler(&self, name: &str) -> Result<()> {
        let source_path = self.get_source_path(name);
        let object_path = self.get_object_path(name);

        let mut deleted = false;

        if source_path.exists() {
            fs::remove_file(&source_path)
                .context(format!("Failed to delete source file for scheduler '{}'", name))?;
            log::info!("Deleted source: {}", source_path.display());
            deleted = true;
        }

        if object_path.exists() {
            fs::remove_file(&object_path)
                .context(format!("Failed to delete object file for scheduler '{}'", name))?;
            log::info!("Deleted object: {}", object_path.display());
            deleted = true;
        }

        if !deleted {
            log::warn!("Scheduler '{}' not found", name);
        }

        Ok(())
    }

    /// Check if a scheduler exists
    ///
    /// # Arguments
    /// * `name` - Name of the scheduler
    ///
    /// # Returns
    /// SchedulerInfo with existence details
    pub fn scheduler_exists(&self, name: &str) -> SchedulerInfo {
        let source_path = self.get_source_path(name);
        let object_path = self.get_object_path(name);

        SchedulerInfo {
            name: name.to_string(),
            has_source: source_path.exists(),
            has_object: object_path.exists(),
        }
    }

    /// List all managed schedulers
    ///
    /// # Returns
    /// Vec of SchedulerInfo for all schedulers in the work directory
    pub fn list_schedulers(&self) -> Result<Vec<SchedulerInfo>> {
        let mut schedulers = HashMap::new();

        // Scan for .bpf.c and .bpf.o files
        let entries = fs::read_dir(&self.work_dir)
            .context("Failed to read work directory")?;

        for entry in entries {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if file_name.ends_with(".bpf.c") {
                    let name = file_name.trim_end_matches(".bpf.c");
                    schedulers.entry(name.to_string())
                        .or_insert_with(|| SchedulerInfo {
                            name: name.to_string(),
                            has_source: false,
                            has_object: false,
                        })
                        .has_source = true;
                } else if file_name.ends_with(".bpf.o") {
                    let name = file_name.trim_end_matches(".bpf.o");
                    schedulers.entry(name.to_string())
                        .or_insert_with(|| SchedulerInfo {
                            name: name.to_string(),
                            has_source: false,
                            has_object: false,
                        })
                        .has_object = true;
                }
            }
        }

        let mut result: Vec<_> = schedulers.into_values().collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(result)
    }

    // ============================================================================
    // Internal Implementation Methods
    // ============================================================================

    /// Compile a BPF scheduler from a .bpf.c file
    ///
    /// # Arguments
    /// * `bpf_source` - Path to the .bpf.c source file
    ///
    /// # Returns
    /// Path to the compiled .bpf.o object file
    pub fn compile_bpf_scheduler(&self, bpf_source: impl AsRef<Path>) -> Result<PathBuf> {
        let bpf_source = bpf_source.as_ref();

        // Validate input file exists and has .bpf.c extension
        if !bpf_source.exists() {
            anyhow::bail!("BPF source file does not exist: {}", bpf_source.display());
        }

        let file_name = bpf_source
            .file_name()
            .context("Invalid BPF source filename")?
            .to_str()
            .context("Non-UTF8 filename")?;

        if !file_name.ends_with(".bpf.c") {
            anyhow::bail!("BPF source must have .bpf.c extension, got: {}", file_name);
        }

        // Determine output path
        let output_name = file_name.replace(".bpf.c", ".bpf.o");
        let output_path = self.work_dir.join(&output_name);

        // Build include paths
        let scx_includes = self.build_scx_includes()?;
        let sys_includes = Self::build_sys_includes();

        // Build BPF compilation flags
        let mut clang_args = vec![
            "-g".to_string(),
            "-O2".to_string(),
            "-Wall".to_string(),
            "-Wno-compare-distinct-pointer-types".to_string(),
            "-D__TARGET_ARCH_x86".to_string(),
            "-mcpu=v3".to_string(),
            "-mlittle-endian".to_string(),
            "-target".to_string(),
            "bpf".to_string(),
            "-c".to_string(),
        ];

        // Add system includes
        clang_args.extend(sys_includes);

        // Add scx includes
        clang_args.extend(scx_includes);

        // Add input and output files
        clang_args.push(bpf_source.to_string_lossy().to_string());
        clang_args.push("-o".to_string());
        clang_args.push(output_path.to_string_lossy().to_string());

        // Execute clang compilation
        log::info!("Compiling BPF scheduler: {}", bpf_source.display());
        log::debug!("Clang args: {:?}", clang_args);

        let output = Command::new("clang")
            .args(&clang_args)
            .current_dir(&self.work_dir)
            .output()
            .context("Failed to execute clang")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "BPF compilation failed:\n{}",
                stderr
            );
        }

        // Verify output file was created
        if !output_path.exists() {
            anyhow::bail!("Compilation succeeded but output file not found: {}", output_path.display());
        }

        log::info!("Successfully compiled BPF scheduler: {}", output_path.display());
        Ok(output_path)
    }

    /// Build scx-specific include paths
    fn build_scx_includes(&self) -> Result<Vec<String>> {
        let scx_base = self.project_root.join("scheduler/scx/scheds/include");

        let includes = vec![
            scx_base.clone(),
            scx_base.join("scx"),
            scx_base.join("arch/x86"),
            scx_base.join("bpf-compat"),
            scx_base.join("lib"),
        ];

        // Verify paths exist
        for path in &includes {
            if !path.exists() {
                log::warn!("Include path does not exist: {}", path.display());
            }
        }

        Ok(includes
            .into_iter()
            .map(|p| format!("-I{}", p.display()))
            .collect())
    }

    /// Build system include paths
    fn build_sys_includes() -> Vec<String> {
        vec![
            "-idirafter", "/usr/lib/llvm-19/lib/clang/19/include",
            "-idirafter", "/usr/local/include",
            "-idirafter", "/usr/include/x86_64-linux-gnu",
            "-idirafter", "/usr/include",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect()
    }


    /// Verify scheduler execution in kernel
    ///
    /// Loads the compiled BPF scheduler into the kernel, runs it for a specified
    /// duration, then stops it. This verifies that the scheduler can be loaded
    /// and run successfully.
    ///
    /// # Arguments
    /// * `bpf_object` - Path to the compiled .bpf.o file
    /// * `duration` - Duration to run the scheduler (default: 10 seconds)
    ///
    /// # Returns
    /// Result with execution output
    pub async fn execution_verify(
        &self,
        bpf_object: impl AsRef<Path>,
        duration: Option<Duration>,
    ) -> Result<String> {
        let bpf_object = bpf_object.as_ref();
        let duration = duration.unwrap_or(Duration::from_secs(10));

        // Validate input
        if !bpf_object.exists() {
            anyhow::bail!("BPF object file does not exist: {}", bpf_object.display());
        }

        if !bpf_object.to_string_lossy().ends_with(".bpf.o") {
            anyhow::bail!("File must be a .bpf.o object file");
        }

        // Use the loader binary from new_sched directory
        let loader_path = self.work_dir.join("loader");
        if !loader_path.exists() {
            anyhow::bail!(
                "Loader binary not found at {}. Run 'make' in new_sched directory first.",
                loader_path.display()
            );
        }

        log::info!("Starting scheduler verification for: {}", bpf_object.display());
        log::info!("Test duration: {} seconds", duration.as_secs());

        // Check if we need sudo
        let sudo_password = std::env::var("SCHEDCP_SUDO_PASSWORD").ok();

        // Start the scheduler process
        let mut child = if sudo_password.is_some() {
            // With sudo
            let mut cmd = tokio::process::Command::new("sudo");
            cmd.arg("-S") // Read password from stdin
                .arg(&loader_path)
                .arg(bpf_object)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .current_dir(&self.work_dir);

            let mut child = cmd.spawn()
                .context("Failed to spawn scheduler process with sudo")?;

            // Write password to stdin if provided
            if let Some(ref password) = sudo_password {
                if let Some(mut stdin) = child.stdin.take() {
                    use tokio::io::AsyncWriteExt;
                    stdin.write_all(format!("{}\n", password).as_bytes()).await
                        .context("Failed to write sudo password")?;
                }
            }

            child
        } else {
            // Without sudo (for testing or if passwordless sudo is configured)
            tokio::process::Command::new("sudo")
                .arg(&loader_path)
                .arg(bpf_object)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .current_dir(&self.work_dir)
                .spawn()
                .context("Failed to spawn scheduler process")?
        };

        let pid = child.id().context("Failed to get child process ID")?;
        log::info!("Scheduler started with PID: {}", pid);

        // Wait for the specified duration
        log::info!("Running scheduler for {} seconds...", duration.as_secs());
        tokio::time::sleep(duration).await;

        // Stop the scheduler
        log::info!("Stopping scheduler...");

        // Kill the child process
        child.start_kill().context("Failed to kill scheduler process")?;

        // Wait for process to finish with timeout
        let output = timeout(Duration::from_secs(5), child.wait_with_output())
            .await
            .context("Timeout waiting for scheduler to stop")?
            .context("Failed to get scheduler output")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        log::info!("Scheduler stopped");
        log::debug!("Exit status: {:?}", output.status);
        log::debug!("Stdout: {}", stdout);
        log::debug!("Stderr: {}", stderr);

        // Check if execution was successful
        // Note: Schedulers often exit with non-zero when killed, so we check stderr for errors
        if stderr.contains("error") || stderr.contains("Error") || stderr.contains("failed") {
            anyhow::bail!(
                "Scheduler execution failed:\nStdout: {}\nStderr: {}",
                stdout,
                stderr
            );
        }

        Ok(format!(
            "Scheduler verification successful!\nRan for {} seconds\nPID: {}\nOutput:\n{}\n{}",
            duration.as_secs(),
            pid,
            stdout,
            stderr
        ))
    }
}