# ML Scheduler Documentation

## Overview

The ML Scheduler is a machine learning-based scheduler component that uses TensorFlow to make intelligent task migration decisions in the Linux kernel scheduler. It integrates with the sched-ext (scx) framework to optimize CPU task placement across domains.

## Architecture

### Components

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

## Model Details

### Input Format
- **Shape**: `[batch_size, 10]`
- **Type**: float32
- **Features**: Currently uses 5 features, padded with zeros to 10

### Output Format
- **Shape**: `[batch_size, 1]`
- **Type**: float32 (sigmoid activation, range 0-1)
- **Interpretation**: Values > 0.5 indicate migration

### Model Architecture
The dummy model includes:
- Input layer: 10 features
- Hidden layer 1: 64 neurons with ReLU
- Hidden layer 2: 64 neurons with ReLU
- Output layer: 1 neuron with Sigmoid

## Integration with BPF Scheduler

In production, the ML Scheduler would:
1. Receive migration requests from BPF scheduler via shared memory
2. Batch multiple requests for efficient inference
3. Return decisions back to kernel space
4. Update model based on performance feedback

## Building and Running

### Prerequisites
- Rust 1.82+
- TensorFlow C library
- Pre-trained model in SavedModel format

### Build
```bash
make build
# or
cargo build --release
```

### Run
```bash
make run
# or
./target/release/ml-scheduler --model-path path/to/model
```

### Command Line Options
- `--model-path`: Path to TensorFlow SavedModel directory (default: `src/model_dir/model_path`)
- `--verbose`: Enable debug logging

## Performance Considerations

1. **Inference Latency**: Single inference ~1-5ms on CPU
2. **Batching**: Can process multiple migration decisions together
3. **Caching**: Model loaded once at startup
4. **Threading**: Async-ready with Tokio runtime

## Future Improvements

1. **Online Learning**: Update model based on migration outcomes
2. **Feature Engineering**: Add more sophisticated scheduling features
3. **Model Optimization**: Use TensorFlow Lite or quantization
4. **Hardware Acceleration**: GPU/TPU support for inference
5. **Real BPF Integration**: Connect to actual kernel scheduler

## Troubleshooting

### Common Issues

1. **Type Mismatch Error**
   - Ensure model expects float32 inputs
   - Check tensor dimensions match [batch, 10]

2. **Model Loading Failed**
   - Verify model path exists
   - Check SavedModel format compatibility

3. **Inference Errors**
   - Validate input features are within expected ranges
   - Check TensorFlow version compatibility