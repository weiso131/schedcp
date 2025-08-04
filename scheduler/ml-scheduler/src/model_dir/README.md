# Model Directory

This directory contains the TensorFlow SavedModel used by the ML Scheduler for making task migration decisions.

## Directory Structure

```
model_dir/
├── README.md                  # This file
├── create_dummy_model.py      # Script to create a dummy model for testing
├── model_export.py           # Script to train and export production models
├── transfer.py               # Script to convert Keras .h5 models to SavedModel format
└── model_path/               # TensorFlow SavedModel directory (created after running scripts)
    ├── saved_model.pb        # Model graph definition
    ├── variables/            # Model weights and variables
    │   ├── variables.data-00000-of-00001
    │   └── variables.index
    └── assets/               # Additional model assets (if any)
```

## How It Works

### Model Architecture

The neural network model implements a binary classifier for migration decisions:

1. **Input Layer**: 10 features (float32)
   - Features 0-4: Active scheduling metrics
     - `cpu`: Current CPU ID
     - `cpu_idle`: Number of idle CPUs
     - `cpu_not_idle`: Number of busy CPUs  
     - `src_dom_load`: Source domain load (0.0-1.0)
     - `dst_dom_load`: Destination domain load (0.0-1.0)
   - Features 5-9: Padding zeros (reserved for future features)

2. **Hidden Layers**:
   - Layer 1: 64 neurons with ReLU activation
   - Layer 2: 64 neurons with ReLU activation

3. **Output Layer**: 1 neuron with Sigmoid activation
   - Output > 0.5: Migrate task
   - Output ≤ 0.5: Keep task on current CPU

### Model Creation

#### For Testing (Dummy Model)

The `create_dummy_model.py` script creates a simple model with random weights:

```bash
python3 create_dummy_model.py
```

This creates a model that:
- Has the correct architecture for the scheduler
- Makes random migration decisions (for testing only)
- Validates the TensorFlow integration

#### For Production (Trained Model)

The `model_export.py` script can be used to train a real model:

```bash
python3 model_export.py
```

This script:
1. Generates synthetic training data based on load patterns
2. Trains a neural network to predict optimal migrations
3. Exports the model in SavedModel format

### Model Conversion

If you have an existing Keras model in .h5 format (e.g., from the original research), use `transfer.py`:

```bash
python3 transfer.py --input model_stress_model.h5 --output model_path
```

This converts:
- Keras H5 format → TensorFlow SavedModel format
- Preserves model weights and architecture
- Makes it compatible with TensorFlow C API

## Training a Production Model

For a production-ready model, you would:

1. **Collect Real Data**:
   ```python
   # Collect migration decisions and outcomes
   data = {
       'features': [...],  # CPU and load metrics
       'migrated': [...],  # Whether migration occurred
       'performance': [...]  # Performance impact
   }
   ```

2. **Label Data**:
   - Positive examples: Migrations that improved performance
   - Negative examples: Migrations that degraded performance

3. **Train Model**:
   ```python
   model.fit(
       X_train, y_train,
       validation_data=(X_val, y_val),
       epochs=100,
       callbacks=[early_stopping, model_checkpoint]
   )
   ```

4. **Evaluate Performance**:
   - Migration reduction rate
   - Performance improvement metrics
   - False positive/negative rates

## Model Files

### saved_model.pb
- Protocol buffer file containing the model graph
- Defines the network architecture and operations
- Platform-independent representation

### variables/
- Contains the trained weights and biases
- Binary format optimized for fast loading
- Checkpoint-compatible format

### assets/
- Additional files needed by the model (if any)
- Could include vocabulary files, normalization parameters, etc.

## Integration with Scheduler

The scheduler loads the model at startup:

```rust
// In lib.rs
let model = TensorFlowModel::new("src/model_dir/model_path")?;

// For each migration decision
let input = prepare_features(&migration_request);
let should_migrate = model.predict(&input)?;
```

The model is loaded once and reused for all predictions, minimizing overhead.

## Performance Considerations

1. **Model Size**: ~500KB for the default architecture
2. **Load Time**: ~100-200ms at startup
3. **Inference Time**: ~1-5ms per decision
4. **Memory Usage**: ~10-20MB including TensorFlow runtime

## Updating the Model

To update the model with new training:

1. Train new model with updated data
2. Validate performance on test set
3. Export to `model_path_new/`
4. Test with scheduler in development environment
5. Replace `model_path/` with new model
6. Restart scheduler to load new model

## Troubleshooting

### Model Not Found
```
Error: Model not found at path
```
Solution: Run `python3 create_dummy_model.py` to create model

### Incompatible Model Format
```
Error: Invalid SavedModel format
```
Solution: Ensure model was exported with TensorFlow 2.x SavedModel API

### Wrong Input Shape
```
Error: Expected shape [?, 10], got [?, 5]
```
Solution: Model expects 10 features, pad with zeros if needed

## References

- TensorFlow SavedModel Guide: https://www.tensorflow.org/guide/saved_model
- Original Paper: "Can ML Techniques Improve Linux Kernel Scheduler Decisions?"
- Model Training Notebook: See `model_export.py` for example code