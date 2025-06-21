# 🚀 nomopoly Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Quick Setup

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv nomopoly_env

# Activate virtual environment
# On macOS/Linux:
source nomopoly_env/bin/activate
# On Windows:
# nomopoly_env\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install nomopoly in development mode
pip install -e .
```

### 3. Verify Installation

```bash
# Run the test suite
python3 test_nomopoly.py
```

You should see output like:
```
🔐 NOMOPOLY - Package Test Suite
==================================================
Testing imports...
✅ All imports successful

Testing network creation...
✅ Networks created successfully
  Prover params: 57,537
  Verifier params: 181,121
  Adversary params: 174,369

Testing forward passes...
✅ Prover forward pass: torch.Size([4, 2]) -> torch.Size([4, 1]), torch.Size([4, 32])
✅ Verifier forward pass: verification shape torch.Size([4, 1])
✅ Adversary forward pass: torch.Size([4, 2]) -> torch.Size([4, 1]), torch.Size([4, 32])

Testing ONNX graph creation...
Simple sum ONNX model saved to temp_test/test_sum.onnx
✅ ONNX graph created successfully

Testing training setup...
✅ Training setup successful
  Device: cpu

==================================================
📊 Test Results: 5/5 tests passed
🎉 All tests passed! nomopoly is ready to use.
==================================================
```

### 4. Run the Complete Demo

```bash
# Run the full demonstration
python3 demo.py
```

This will:
1. Create a simple ONNX sum graph
2. Initialize and train the ZK networks for 50 epochs
3. Benchmark the performance
4. Export models to ONNX format
5. Generate training visualizations
6. Show live system demonstration

## What You'll Get

After running the demo, you'll have:

- `models/`: Trained PyTorch model checkpoints
- `exported_models/`: ONNX models ready for deployment
- `benchmark_results/`: Comprehensive evaluation results
- `plots/`: Training progress visualizations
- `logs/`: TensorBoard training logs

## Viewing Training Progress

To view training progress in real-time with TensorBoard:

```bash
# Install tensorboard if not already installed
pip install tensorboard

# Start tensorboard (after training has started)
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser.

## Hardware Requirements

- **Minimum**: CPU with 4GB RAM
- **Recommended**: GPU with CUDA support for faster training
- **Training time**: 
  - CPU: ~10-15 minutes for demo (50 epochs)
  - GPU: ~2-3 minutes for demo (50 epochs)

## Troubleshooting

### Import Errors
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt
pip install -e .
```

### CUDA Issues
If you have GPU but encounter CUDA issues:
```bash
# Install PyTorch with your specific CUDA version
# Check https://pytorch.org/get-started/locally/
```

### Memory Issues
If you encounter out-of-memory errors:
- Reduce batch size in training
- Reduce model hidden dimensions
- Use CPU instead of GPU

### Permission Issues
```bash
# Make scripts executable
chmod +x demo.py
chmod +x test_nomopoly.py
```

## Next Steps

1. **Experiment**: Modify network architectures in `nomopoly/networks.py`
2. **Extend**: Add support for more complex ONNX graphs
3. **Optimize**: Tune hyperparameters for better performance
4. **Deploy**: Use exported ONNX models in production

## Support

If you encounter issues:
1. Check this setup guide
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Try running in a fresh virtual environment

Happy zero-knowledge machine learning! 🔐 