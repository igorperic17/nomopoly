#!/usr/bin/env python3
"""
Nomopoly Demo: Zero Knowledge Machine Learning Proof of Concept

This script demonstrates the complete workflow of nomopoly:
1. Create a simple ONNX computation graph (sum of two numbers)
2. Initialize ZK networks (Prover, Verifier, Adversary)
3. Train the networks using adversarial training
4. Evaluate and benchmark the results
5. Export models to ONNX format

Run this script to see nomopoly in action!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Import nomopoly modules
from nomopoly import (
    ZKProverNet, 
    ZKVerifierNet, 
    ZKAdversarialNet,
    AutoZKTraining,
    create_simple_onnx_graph,
    OnnxHandler,
    ZKMLBenchmark
)


def main():
    """Main demo function."""
    print("=" * 60)
    print("🔐 NOMOPOLY - Zero Knowledge Machine Learning Demo")
    print("No more polynomial commitments!")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = {
        "input_dim": 2,
        "output_dim": 1,
        "proof_dim": 64,
        "num_epochs": 50,
        "num_samples": 5000,
        "test_samples": 1000,
        "device": "auto"
    }
    
    print(f"Configuration: {config}")
    print()
    
    # Step 1: Create simple ONNX graph
    print("📊 Step 1: Creating simple ONNX computation graph...")
    onnx_path = create_simple_onnx_graph("models/simple_sum.onnx")
    
    # Verify the ONNX model works (skip verification to avoid compatibility issues)
    print("✅ ONNX model created (skipping verification due to version compatibility)")
    print()
    
    # Step 2: Initialize ZK Networks
    print("🧠 Step 2: Initializing ZK Networks...")
    
    # Create networks
    prover = ZKProverNet(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        proof_dim=config["proof_dim"],
        hidden_dims=(128, 256, 128)
    )
    
    verifier = ZKVerifierNet(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        proof_dim=config["proof_dim"],
        hidden_dims=(256, 512, 256, 128)
    )
    
    adversary = ZKAdversarialNet(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        proof_dim=config["proof_dim"],
        hidden_dims=(128, 256, 512, 256, 128)
    )
    
    print(f"✅ Prover parameters: {sum(p.numel() for p in prover.parameters()):,}")
    print(f"✅ Verifier parameters: {sum(p.numel() for p in verifier.parameters()):,}")
    print(f"✅ Adversary parameters: {sum(p.numel() for p in adversary.parameters()):,}")
    print()
    
    # Step 3: Train the networks
    print("🏋️ Step 3: Training ZK Networks...")
    
    trainer = AutoZKTraining(
        prover=prover,
        verifier=verifier,
        adversary=adversary,
        device=config["device"],
        learning_rates={
            "prover": 1e-4,
            "verifier": 2e-4,
            "adversary": 1e-4
        }
    )
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train the networks
    training_stats = trainer.train(
        num_epochs=config["num_epochs"],
        num_samples=config["num_samples"],
        save_interval=10,
        save_dir="models"
    )
    
    print("✅ Training completed!")
    print()
    
    # Step 4: Evaluate and benchmark
    print("📈 Step 4: Evaluating and benchmarking...")
    
    # Create test data
    test_data = torch.FloatTensor(config["test_samples"], config["input_dim"]).uniform_(-10.0, 10.0)
    
    # Initialize benchmark
    benchmark = ZKMLBenchmark(
        prover=prover,
        verifier=verifier,
        adversary=adversary,
        device=config["device"]
    )
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        test_data=test_data,
        save_results=True,
        results_dir="benchmark_results"
    )
    
    # Print key results
    print("\n🎯 Key Results:")
    print(f"Prover MAE: {results['prover_accuracy']['mae']:.6f}")
    print(f"Prover Accuracy: {results['prover_accuracy']['accuracy']:.4f}")
    print()
    
    # Step 5: Export models to ONNX
    print("💾 Step 5: Exporting models to ONNX...")
    
    os.makedirs("exported_models", exist_ok=True)
    
    trainer.save_models_as_onnx(
        save_dir="exported_models",
        input_shape=(1, config["input_dim"])
    )
    
    print("✅ Models exported to ONNX format!")
    print()
    
    # Step 6: Plot training progress
    print("📊 Step 6: Generating training plots...")
    
    os.makedirs("plots", exist_ok=True)
    trainer.plot_training_progress(save_path="plots/training_progress.png")
    
    print("✅ Training plots saved!")
    print()
    
    # Step 7: Demonstrate the system
    print("🎬 Step 7: Live demonstration...")
    demonstrate_system(prover, verifier, adversary, trainer.device)
    
    print("=" * 60)
    print("🎉 Demo completed successfully!")
    print("Files generated:")
    print("  - models/: Trained PyTorch models")
    print("  - exported_models/: ONNX models for deployment")
    print("  - benchmark_results/: Comprehensive benchmark results")
    print("  - plots/: Training progress visualization")
    print("  - logs/: TensorBoard logs")
    print("=" * 60)


def demonstrate_system(prover, verifier, adversary, device):
    """Demonstrate the ZK system with live examples."""
    print("\n🔍 Live System Demonstration:")
    
    # Set networks to evaluation mode
    prover.eval()
    verifier.eval()
    adversary.eval()
    
    # Test cases
    test_cases = [
        [2.0, 3.0],   # Expected: 5.0
        [7.0, -2.0],  # Expected: 5.0
        [0.0, 10.0],  # Expected: 10.0
        [-5.0, -3.0]  # Expected: -8.0
    ]
    
    with torch.no_grad():
        for i, inputs in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {inputs}")
            
            # Convert to tensor
            x = torch.FloatTensor([inputs]).to(device)
            
            # Ground truth
            ground_truth = sum(inputs)
            
            # Prover computation and proof
            prover_output, proof = prover(x)
            prover_result = prover_output.item()
            
            # Verifier check
            verification_score = verifier(x, prover_output, proof).item()
            
            # Adversary attempt
            fake_output, fake_proof = adversary(x)
            fake_result = fake_output.item()
            fake_verification = verifier(x, fake_output, fake_proof).item()
            
            print(f"  Ground Truth: {ground_truth:.4f}")
            print(f"  Prover Result: {prover_result:.4f} (Error: {abs(prover_result - ground_truth):.6f})")
            print(f"  Proof Verification: {verification_score:.4f} {'✅' if verification_score > 0.5 else '❌'}")
            print(f"  Adversary Fake: {fake_result:.4f}")
            print(f"  Fake Verification: {fake_verification:.4f} {'🚨 FOOLED!' if fake_verification > 0.5 else '✅ DETECTED'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc() 