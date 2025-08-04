# ML-Based Scheduler

This is a standalone implementation of an ML-based scheduler using TensorFlow for intelligent task migration decisions in the Linux kernel scheduler. It integrates with the sched-ext (scx) framework to optimize CPU task placement across domains.

## Overview

This scheduler uses a neural network model to make task migration decisions based on system metrics. It is based on the research paper "Can ML Techniques Improve Linux Kernel Scheduler Decisions?" and achieves:
- 10% improvement in kernel compilation time
- 77% reduction in task migrations
- Dynamic adaptation to workload patterns

## Architecture

### Core Components

1. **TensorFlow Model (`TensorFlowModel`)**
   - Loads a pre-trained SavedModel from disk
   - Performs inference using TensorFlow C++ API
   - Expects input shape: `[batch_size, 10]` with float32 values
   - Returns binary classification (migrate: true/false)

2. **ML Scheduler (`MLScheduler`)**
   - High-level interface for the scheduling logic
   - Manages the TensorFlow model instance
   - Converts scheduler features to model inputs

3. **Migration Features (`MigrationFeatures`)**
   - Represents the current state for migration decision:
     - `cpu`: Current CPU ID
     - `cpu_idle`: Number of idle CPUs in the system
     - `cpu_not_idle`: Number of busy CPUs
     - `src_dom_load`: Source domain load (0.0-1.0)
     - `dst_dom_load`: Destination domain load (0.0-1.0)

4. **BPF Interface**
   - Data structures for communicating with kernel-side BPF programs
   - Fields added to task context for ML inference:
     ```c
     unsigned long src_dom_load;
     unsigned long dst_dom_load;
     int cpu_idle;
     int cpu_not_idle;
     int cpu;
     ```

### File Structure

- **lib.rs**: Core ML scheduler implementation with TensorFlow model loading and prediction
- **main.rs**: Standalone executable for testing the ML scheduler
- **bpf_interface.rs**: Data structures for BPF communication
- **scheduler_integration.rs**: Integration layer for processing migration requests
- **model_export.py**: Python script for training and exporting ML models
- **model_dir/**: Contains the TensorFlow SavedModel and training scripts

## Migration Decision Algorithm

### Core Algorithm

The ML Scheduler implements an intelligent task migration algorithm based on domain load balancing:

```
Algorithm: ML-based Task Migration Decision
Input: cpu_id, idle_cpus, busy_cpus, src_load, dst_load
Output: boolean (migrate or not)

1. Feature Extraction:
   - Normalize CPU metrics to [0,1] range
   - Calculate load imbalance: |src_load - dst_load|
   - Determine system utilization: busy_cpus / total_cpus

2. Feature Vector Construction:
   F = [cpu_id, idle_cpus, busy_cpus, src_load, dst_load, 0, 0, 0, 0, 0]
   
3. Neural Network Forward Pass:
   H1 = ReLU(W1 * F + b1)     // First hidden layer
   H2 = ReLU(W2 * H1 + b2)    // Second hidden layer
   O = Sigmoid(W3 * H2 + b3)   // Output layer
   
4. Decision:
   if O > 0.5:
      return MIGRATE
   else:
      return STAY
```

### Decision Factors

The algorithm considers multiple factors:

1. **Load Imbalance**
   - High source domain load (>0.7) with low destination load (<0.3) favors migration
   - Balanced loads (both ~0.5) discourage migration

2. **System Utilization**
   - More idle CPUs increase migration likelihood
   - Fully loaded system discourages migration

3. **CPU Locality**
   - The specific CPU ID can influence decisions based on topology

### Example Scenarios

**Scenario 1: High Load Imbalance**
```
Input: cpu=4, idle=2, busy=6, src_load=0.9, dst_load=0.2
Expected: MIGRATE (true)
Reasoning: Large load imbalance justifies migration cost
```

**Scenario 2: Balanced System**
```
Input: cpu=2, idle=4, busy=4, src_load=0.5, dst_load=0.5
Expected: STAY (false)
Reasoning: Already balanced, migration would add overhead
```

## Model Details

### Architecture
- Input layer: 10 features
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 1 neuron with Sigmoid activation

### Input/Output Format
- **Input**: Shape `[batch_size, 10]`, float32, 5 features padded with zeros
- **Output**: Shape `[batch_size, 1]`, float32 (0-1), >0.5 indicates migration

## Prerequisites

Install the TensorFlow C library:

```bash
# Download TensorFlow C library
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz

# Extract to /usr/local (requires sudo)
sudo tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz

# Update library cache
sudo ldconfig

# Clean up
rm libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz
```

## Building and Running

### Quick Start

```bash
# Install Python dependencies
make install-deps

# Generate a sample model (creates dummy model in src/model_dir/model_path/)
make model

# Build the scheduler
make build

# Run the scheduler
make run
```

### Detailed Instructions

1. **First-time setup:**
   ```bash
   # Install TensorFlow C library (see Prerequisites above)
   
   # Install Python dependencies
   make install-deps
   ```

2. **Generate the ML model:**
   ```bash
   # This creates the model in src/model_dir/model_path/
   make model
   
   # If the model generation fails, create a dummy model manually:
   cd src/model_dir && python3 create_dummy_model.py && cd ../..
   ```

3. **Build and run:**
   ```bash
   # Build the scheduler binary
   cargo build --release
   
   # Run tests
   cargo run --bin test-ml
   
   # Run the scheduler (press Ctrl+C to stop)
   ./target/release/ml-scheduler --model-path src/model_dir/model_path
   ```

### Command Line Options
- `--model-path`: Path to TensorFlow SavedModel directory (default: `src/model_dir/model_path`)
- `--verbose`: Enable debug logging

## How It Works

### 1. Initialization
```rust
let scheduler = MLScheduler::new("path/to/model")?;
```
- Loads the TensorFlow SavedModel from the specified directory
- Initializes the graph and session for inference

### 2. Feature Collection
The scheduler collects runtime features from the kernel:
- CPU utilization metrics
- Domain load balancing information
- Task migration candidates

### 3. Inference Process
```rust
let features = MigrationFeatures {
    cpu: 4,
    cpu_idle: 2,
    cpu_not_idle: 2,
    src_dom_load: 0.75,
    dst_dom_load: 0.25,
};
let should_migrate = scheduler.should_migrate(&features)?;
```

The inference pipeline:
1. Converts features to float32 vector
2. Pads input to 10 features (model requirement)
3. Creates 2D tensor with shape [1, 10]
4. Runs TensorFlow inference
5. Applies threshold (>0.5) to get binary decision

### 4. Decision Making
- **True**: Task should be migrated to balance load
- **False**: Task should remain on current CPU

## Integration with sched-ext

In production, the ML Scheduler integrates with the kernel BPF scheduler:

### Communication Protocol

1. **Shared Memory Interface**
   ```c
   struct migration_request {
       u32 cpu_id;
       u32 task_pid;
       u64 timestamp;
       struct migration_features features;
   };
   ```

2. **Request Flow**
   ```
   BPF Scheduler → Ring Buffer → ML Scheduler
                                     ↓
   Decision Cache ← Response ← Neural Network
   ```

3. **Batching Strategy**
   - Collect requests for up to 1ms or 32 requests
   - Process batch through neural network
   - Cache decisions for frequently seen patterns

### Performance Optimizations

1. **Decision Caching**: LRU cache for recent decisions
2. **Feature Quantization**: Quantize loads to 0.1 increments
3. **Inference Optimization**: Pre-allocate tensors, reuse session

## Performance Considerations

1. **Inference Latency**: Single inference ~1-5ms on CPU
2. **Batching**: Can process multiple migration decisions together
3. **Caching**: Model loaded once at startup
4. **Threading**: Async-ready with Tokio runtime
5. **Algorithm Complexity**:
   - Time: O(1) for inference (fixed network size)
   - Space: O(1) for decision (model pre-loaded)

## Model Training

For production model training:

1. **Data Collection**
   - Gather migration decisions and performance outcomes
   - Track metrics: latency, throughput, cache misses

2. **Labels**
   - Positive: Migrations that improved performance
   - Negative: Migrations that degraded performance

3. **Loss Function**
   - Binary cross-entropy for classification
   - Custom penalties for migration overhead

4. **Model Conversion**
   - Original model: `model_stress_model.h5`
   - Convert to SavedModel using `transfer.py`

## Troubleshooting

### Common Issues

1. **"libtensorflow_framework.so.2: cannot open shared object file"**
   - Install the TensorFlow C library as described in Prerequisites

2. **Model Loading Failed**
   - Verify model path exists
   - Check SavedModel format compatibility

3. **Type Mismatch Error**
   - Ensure model expects float32 inputs
   - Check tensor dimensions match [batch, 10]

4. **Inference Errors**
   - Validate input features are within expected ranges
   - Check TensorFlow version compatibility

## Future Improvements

1. **Online Learning**: Update model based on migration outcomes
2. **Feature Engineering**: Add more sophisticated scheduling features
3. **Model Optimization**: Use TensorFlow Lite or quantization
4. **Hardware Acceleration**: GPU/TPU support for inference
5. **Real BPF Integration**: Full connection to kernel scheduler

## Original Research

This implementation is based on:
- Paper: "Can ML Techniques Improve Linux Kernel Scheduler Decisions?" (https://arxiv.org/abs/2407.10077)
- Fork: scx_rusty commit 7eb71797d35d7d2f45daba7418159c5ab902d1ab

## License

This project follows the same licensing as the sched-ext framework.