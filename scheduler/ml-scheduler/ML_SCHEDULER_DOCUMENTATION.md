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

The algorithm considers multiple factors for migration decisions:

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

**Scenario 3: Light Load**
```
Input: cpu=1, idle=7, busy=1, src_load=0.1, dst_load=0.1
Expected: STAY (false)
Reasoning: Low overall load, no benefit from migration
```

### Mathematical Formulation

The neural network implements the function:
```
f(x) = σ(W₃ · ReLU(W₂ · ReLU(W₁ · x + b₁) + b₂) + b₃)
```

Where:
- σ = Sigmoid activation: `1 / (1 + e^(-x))`
- ReLU = Rectified Linear Unit: `max(0, x)`
- W₁, W₂, W₃ = Weight matrices
- b₁, b₂, b₃ = Bias vectors

### Theoretical Foundation

The ML scheduler is based on several key principles:

1. **Load Balancing Theory**
   - Minimizes makespan: `max(load_i)` across all domains
   - Follows the principle: migrate when `benefit > cost`
   - Benefit = `(src_load - dst_load) × task_weight`
   - Cost = `migration_overhead + cache_penalty`

2. **Online Learning**
   - Adapts to workload patterns over time
   - Uses reinforcement learning signals from performance metrics
   - Reward function: `R = -latency_delta + throughput_delta`

3. **Decision Boundaries**
   - The neural network learns non-linear decision boundaries
   - Can capture complex interactions between features
   - Example: High CPU count with low load variance → no migration

### Algorithm Complexity

- **Time Complexity**: O(1) for inference (fixed network size)
- **Space Complexity**: O(1) for decision (model pre-loaded)
- **Cache Lookup**: O(1) average case with hash table

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
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 1 neuron with Sigmoid activation

### Training Considerations

For a production model, training would involve:

1. **Data Collection**
   - Gather migration decisions and performance outcomes
   - Track metrics: latency, throughput, cache misses

2. **Labels**
   - Positive: Migrations that improved performance
   - Negative: Migrations that degraded performance

3. **Loss Function**
   - Binary cross-entropy for classification
   - Can add custom penalties for migration overhead

4. **Optimization**
   - Online learning with gradient descent
   - Periodic retraining with collected data

## Integration with BPF Scheduler

### Communication Protocol

In production, the ML Scheduler integrates with the kernel BPF scheduler:

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

1. **Decision Caching**
   ```rust
   // LRU cache for recent decisions
   if let Some(cached) = decision_cache.get(&features_hash) {
       return cached.decision;
   }
   ```

2. **Feature Quantization**
   - Quantize loads to 0.1 increments
   - Reduces unique feature combinations
   - Improves cache hit rate

3. **Inference Optimization**
   - Pre-allocate tensors
   - Reuse session across requests
   - Consider TensorRT for GPU acceleration

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