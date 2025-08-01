# Process Manager

A Rust library for managing external processes with support for binary extraction, output streaming, and lifecycle management.

## Features

- **Binary Extraction**: Extract embedded binaries to temporary directories with automatic cleanup
- **Process Management**: Start, stop, and monitor multiple processes
- **Output Streaming**: Real-time streaming of stdout/stderr from running processes
- **Sudo Support**: Optional sudo execution with password authentication
- **Async/Await**: Built on Tokio for efficient async operations
- **Type Safety**: Strong typing with comprehensive error handling

## Architecture

The library consists of four main components:

### 1. BinaryExtractor
Manages extraction of embedded binaries to temporary directories:
- Creates temporary directories that are automatically cleaned up
- Sets appropriate executable permissions
- Provides path resolution for extracted binaries

### 2. ProcessRunner
Handles individual process execution:
- Spawns processes with configurable arguments
- Captures stdout/stderr streams
- Manages process lifecycle (start, stop, status)
- Supports both regular and sudo execution

### 3. ProcessManager
High-level API for managing multiple processes:
- Coordinates binary extraction and process execution
- Maintains registry of running processes
- Provides batch operations (start multiple, stop all)
- Offers process querying and monitoring

### 4. Types
Common types and error definitions:
- `ProcessInfo`: Process metadata and status
- `ProcessConfig`: Configuration for starting processes
- `ProcessStatus`: Running state enumeration
- `ProcessError`: Comprehensive error types

## Usage

### Basic Example

```rust
use process_manager::{BinaryExtractor, ProcessManager, ProcessConfig};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create binary extractor and add binaries
    let mut extractor = BinaryExtractor::new()?;
    extractor.add_binary("my_program", include_bytes!("../my_program"))?;
    
    // Create process manager
    let manager = ProcessManager::new(extractor);
    
    // Configure and start a process
    let config = ProcessConfig {
        name: "my_process".to_string(),
        binary_name: "my_program".to_string(),
        args: vec!["--arg1".to_string(), "value".to_string()],
        env: HashMap::new(),
        working_dir: None,
    };
    
    let process_id = manager.start_process(config).await?;
    
    // Get process info
    if let Some(info) = manager.get_process_info(process_id) {
        println!("Process {} is {:?}", info.name, info.status);
    }
    
    // Stop the process
    manager.stop_process(process_id).await?;
    
    Ok(())
}
```

### Streaming Output

```rust
// Get output stream from a running process
if let Some(mut stream) = manager.get_output_stream(process_id) {
    use futures::StreamExt;
    
    while let Some(line) = stream.next().await {
        println!("Process output: {}", line);
    }
}
```

### Sudo Execution

```rust
use process_manager::ProcessRunner;

// Start a process with sudo
let runner = ProcessRunner::start_with_sudo(
    "/path/to/binary",
    "process_name".to_string(),
    vec!["arg1".to_string()],
    "sudo_password",
).await?;
```

## API Reference

### ProcessManager

- `new(extractor: BinaryExtractor) -> Self`
- `start_process(config: ProcessConfig) -> Result<Uuid, ProcessError>`
- `start_process_with_sudo(config: ProcessConfig, sudo_password: &str) -> Result<Uuid, ProcessError>`
- `stop_process(id: Uuid) -> Result<(), ProcessError>`
- `stop_all_processes() -> Vec<(Uuid, Result<(), ProcessError>)>`
- `get_process_info(id: Uuid) -> Option<ProcessInfo>`
- `get_output_stream(id: Uuid) -> Option<OutputStream>`
- `list_processes() -> Vec<ProcessInfo>`
- `get_running_processes() -> Vec<ProcessInfo>`

### BinaryExtractor

- `new() -> Result<Self, ProcessError>`
- `add_binary(name: &str, data: &[u8]) -> Result<(), ProcessError>`
- `add_multiple_binaries(binaries: &[(&str, &[u8])]) -> Result<(), ProcessError>`
- `get_binary_path(name: &str) -> Option<&Path>`
- `list_binaries() -> Vec<&str>`

### ProcessRunner

- `start(binary_path: &str, name: String, args: Vec<String>) -> Result<Self, ProcessError>`
- `start_with_sudo(binary_path: &str, name: String, args: Vec<String>, sudo_password: &str) -> Result<Self, ProcessError>`
- `stop() -> Result<(), ProcessError>`
- `get_output_stream() -> Option<OutputStream>`
- `info() -> &ProcessInfo`
- `is_running() -> bool`

## Error Handling

The library provides detailed error types through `ProcessError`:

- `BinaryNotFound`: Requested binary doesn't exist
- `ProcessNotFound`: Process ID not found
- `StartFailed`: Process failed to start
- `StopFailed`: Process failed to stop
- `ExtractionFailed`: Binary extraction failed
- `Io`: Underlying I/O error

## Dependencies

- `tokio`: Async runtime
- `futures`: Stream utilities
- `uuid`: Process identification
- `chrono`: Timestamp management
- `tempfile`: Temporary directory management
- `dashmap`: Concurrent hash map
- `async-stream`: Stream creation utilities
- `thiserror`: Error derivation

## Safety and Security

- Binaries are extracted with restricted permissions (755)
- Temporary directories are automatically cleaned up
- Sudo passwords are handled securely (not logged)
- Processes are killed on drop by default
- Output buffers are bounded to prevent memory exhaustion

## Testing

The library includes comprehensive tests for all major components. Run tests with:

```bash
cargo test
```

## License

This library is part of the schedcp project and follows the same license terms.