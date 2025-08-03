# Emotion Classifier Mass Testing Guide

This repository contains several scripts for mass testing and hyperparameter optimization of emotion classification models.

## Scripts Available

1. **emotion-classifier-mass-testing.py** - Comprehensive mass testing with multiple architectures and configurations
2. **emotion-classifier-quick-test.py** - Quick testing with 8 promising configurations
3. **emotion-classifier-hyperopt.py** - Advanced hyperparameter optimization using random search and grid search

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Test (Recommended to start with)
```bash
python emotion-classifier-quick-test.py
```

This will test 8 different promising model configurations and should complete in 15-30 minutes.

### Comprehensive Mass Testing
```bash
python emotion-classifier-mass-testing.py
```

This will test dozens of combinations and may take several hours to complete.

### Advanced Hyperparameter Optimization
```bash
python emotion-classifier-hyperopt.py
```

This runs both random search and focused grid search, and may take 2-4 hours.

## Output Files

Each script generates:
- **JSON results file** - Detailed results with all configurations tested
- **CSV summary file** - Tabular summary for easy analysis
- **Best model .pth file** - PyTorch model weights for the best performing model
- **Best model .onnx file** - ONNX format for deployment

## Model Architectures Tested

### Neural Network Variations:
- Simple shallow networks (1-2 layers)
- Deep networks (3-4 layers)
- Wide networks (larger hidden layers)
- Batch normalization variants
- Different activation functions (ReLU, LeakyReLU, ELU, GELU)

### Hyperparameters:
- Learning rates: 0.0001 to 0.1
- Batch sizes: 16, 32, 64, 128
- Dropout rates: 0.0 to 0.5
- Weight decay: 0.0 to 0.01
- Optimizers: Adam, SGD, RMSprop
- Learning rate schedulers

## Expected Results

The scripts will automatically:
1. Train each model configuration
2. Evaluate on test data
3. Save the best performing model
4. Generate comprehensive reports
5. Export the best model to ONNX format for Unity integration

## Tips for Best Results

1. **Start with quick-test** to get a baseline
2. **Use mass-testing** for comprehensive exploration
3. **Use hyperopt** for fine-tuning the best architectures
4. **Monitor GPU usage** if available (models will automatically use CUDA)
5. **Check generated CSV files** for detailed analysis

## Integration with Unity

The best model will be exported as an ONNX file that can be directly used in Unity with the Barracuda inference engine.

## File Structure After Running

```
Facial Recognition Model/
├── emotion-classifier-*.py           # Testing scripts
├── data/
│   └── unity_collection.csv         # Your training data
├── best_model_*.pth                  # Best PyTorch models
├── best_model_*.onnx                 # Best ONNX models
├── *_results_*.json                  # Detailed results
├── *_summary_*.csv                   # Summary tables
└── requirements.txt                  # Dependencies
```
