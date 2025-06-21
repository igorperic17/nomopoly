# 🔐 nomopoly - No More Polynomial Commitments!

**nomopoly** is a neural network compiler for converting ONNX compute graphs into Zero Knowledge ML (ZKML) circuits using adversarial training. Instead of traditional polynomial commitments, nomopoly trains three neural networks jointly to create a zero-knowledge proof system for machine learning inference.

## ✨ Key Features

- **🧠 Three-Network Architecture**: Prover, Verifier, and Adversarial networks trained together
- **🎯 GAN-like Training**: Adversarial training approach for robust ZK proofs
- **📊 ONNX Integration**: Seamless integration with ONNX computation graphs
- **⚡ PyTorch Backend**: Built on PyTorch for flexibility and performance
- **📈 Comprehensive Benchmarking**: Built-in evaluation and comparison tools
- **🔄 Model Export**: Export trained models to ONNX for deployment

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/nomopoly.git
cd nomopoly

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Run the Demo

```bash
# Run the complete demo
python demo.py
```

This will:
1. Create a simple ONNX sum graph
2. Initialize and train the ZK networks
3. Benchmark performance
4. Export models to ONNX
5. Generate training visualizations

## 🏗️ Architecture

### Core Components

#### 1. ZKProverNet
- Augments original computation with proof generation
- Learns to create valid proofs for correct computations
- Architecture: Original computation + Proof generation network

#### 2. ZKVerifierNet  
- Binary classifier for proof verification
- Distinguishes valid proofs from invalid ones
- Takes input, output, and proof as input

#### 3. ZKAdversarialNet
- Generates fake proofs to test verifier robustness
- Learns to exploit verifier weaknesses
- Creates adversarial examples for training

### Training Process

The three networks are trained jointly using an adversarial approach:

1. **Prover** learns to generate correct computations and convincing proofs
2. **Verifier** learns to distinguish real proofs from fake ones
3. **Adversary** learns to generate fake proofs that fool the verifier

This creates a robust system where the verifier becomes increasingly good at detecting fake proofs, while the prover learns to generate high-quality proofs.

## 📊 Usage Example

```python
from nomopoly import (
    ZKProverNet, ZKVerifierNet, ZKAdversarialNet,
    AutoZKTraining, create_simple_onnx_graph
)

# Create networks
prover = ZKProverNet(input_dim=2, output_dim=1, proof_dim=64)
verifier = ZKVerifierNet(input_dim=2, output_dim=1, proof_dim=64)
adversary = ZKAdversarialNet(input_dim=2, output_dim=1, proof_dim=64)

# Initialize training
trainer = AutoZKTraining(prover, verifier, adversary)

# Train the system
stats = trainer.train(num_epochs=100, num_samples=10000)

# Export trained models
trainer.save_models_as_onnx("./exported_models")
```

## 🔬 Proof of Concept

The current implementation focuses on the simplest possible computation: **sum of two numbers**. This serves as a proof of concept to demonstrate:

- Joint training of three networks
- Adversarial proof generation and verification
- ONNX integration and export
- Comprehensive benchmarking

Future versions will support more complex ONNX graphs and computations.

## 📈 Benchmarking

nomopoly includes comprehensive benchmarking tools:

```python
from nomopoly import ZKMLBenchmark

benchmark = ZKMLBenchmark(prover, verifier, adversary)
results = benchmark.run_comprehensive_benchmark(test_data)
```

Metrics include:
- **Prover accuracy**: How well the prover computes the original function
- **Verifier performance**: Precision/recall for proof verification
- **Adversary effectiveness**: Success rate of fooling the verifier
- **Inference speed**: Timing benchmarks for all networks
- **Model sizes**: Memory footprint analysis

## 🗂️ Project Structure

```
nomopoly/
├── nomopoly/                 # Main package
│   ├── __init__.py          # Package initialization
│   ├── networks.py          # Core network architectures
│   ├── training.py          # Training pipeline
│   ├── utils.py             # ONNX utilities
│   └── benchmarks.py        # Evaluation tools
├── demo.py                  # Complete demo script
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔮 Future Roadmap

- [ ] Support for complex ONNX graphs (CNN, RNN, Transformers)
- [ ] Optimized proof sizes and verification times
- [ ] Integration with existing ZK proof systems
- [ ] Distributed training support
- [ ] Production deployment tools
- [ ] Formal security analysis

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the need for practical ZKML solutions
- Built on the shoulders of PyTorch and ONNX communities
- Thanks to the zero-knowledge cryptography research community