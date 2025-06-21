#!/usr/bin/env python3
"""
Demo script showcasing Zero Knowledge Machine Learning with MNIST classification.
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from nomopoly.networks import ZKProverNet, ZKVerifierNet, ZKAdversarialNet
from nomopoly.training import AutoZKTraining
from nomopoly.benchmarks import ZKBenchmark


def get_real_mnist_sample(digit: int = None) -> torch.Tensor:
    """Get a real MNIST sample for demonstration."""
    
    # Transform to match our network input
    transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    # Load MNIST test set
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    if digit is not None:
        # Find samples of the specific digit
        indices = [i for i, (_, label) in enumerate(test_dataset) if label == digit]
        if indices:
            idx = np.random.choice(indices)
            sample, label = test_dataset[idx]
            return sample.unsqueeze(0), label
    
    # Random sample
    idx = np.random.randint(len(test_dataset))
    sample, label = test_dataset[idx]
    return sample.unsqueeze(0), label


def demonstrate_zk_classification():
    """Demonstrate the ZK classification system with computational proofs."""
    print("🔐 Zero Knowledge MNIST Classification Demo")
    print("=" * 60)
    
    # Initialize networks
    input_dim = 196  # 14x14 flattened
    output_dim = 10  # MNIST digits 0-9
    hidden_dims = (128, 256)
    proof_dim = 64
    
    print(f"Creating ZK networks (input: {input_dim}, output: {output_dim}, proof: {proof_dim})")
    
    prover = ZKProverNet(input_dim, output_dim, proof_dim, hidden_dims)
    verifier = ZKVerifierNet(input_dim, output_dim, proof_dim, (256, 128))
    adversary = ZKAdversarialNet(input_dim, output_dim, proof_dim, (128, 256, 128))
    
    print(f"Networks created with {sum(p.numel() for p in prover.parameters()):,} prover parameters")
    
    # Initialize training
    trainer = AutoZKTraining(prover, verifier, adversary)
    
    # Train the system
    print("\n🏋️ Training ZK system with computational proofs on real MNIST...")
    training_stats = trainer.train(num_epochs=50, num_samples=5000)
    
    # Plot training progress
    trainer.plot_training_progress(training_stats, "plots/training_progress.png")
    
    # Run comprehensive benchmarking
    print("\n📊 Running comprehensive benchmarks...")
    benchmark = ZKBenchmark(prover, verifier, adversary)
    benchmark_results = benchmark.run_comprehensive_benchmark(num_test_samples=1000)
    benchmark.print_benchmark_summary()
    
    # Interactive demonstration
    print("\n🎮 Interactive Classification Demo with Real MNIST")
    print("-" * 50)
    
    # Set networks to evaluation mode
    prover.eval()
    verifier.eval()
    adversary.eval()
    
    with torch.no_grad():
        for demo_round in range(5):
            print(f"\n--- Round {demo_round + 1} ---")
            
            # Get a real MNIST sample
            test_input, actual_digit = get_real_mnist_sample()
            
            print(f"🎯 Ground Truth: Digit {actual_digit}")
            
            # Prover generates classification and proof
            log_probs, _ = prover(test_input)
            probs = torch.exp(log_probs)
            predicted_digit = torch.argmax(log_probs, dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
            
            # Generate computational proof (deterministic from computation)
            computational_proof = trainer.generate_computational_proof(test_input, log_probs)
            
            # Verify computational proof
            real_verification = verifier(test_input, log_probs, computational_proof)
            real_score = real_verification.item()
            
            # Generate fake proof from adversary
            fake_log_probs, fake_proof = adversary(test_input)
            fake_verification = verifier(test_input, fake_log_probs, fake_proof)
            fake_score = fake_verification.item()
            
            print(f"🤖 Prover Prediction: Digit {predicted_digit} (confidence: {confidence:.1%})")
            print(f"✅ Computational Proof Verification: {real_score:.3f} ({'ACCEPTED' if real_score > 0.5 else 'REJECTED'})")
            print(f"❌ Fake Proof Verification: {fake_score:.3f} ({'ACCEPTED' if fake_score > 0.5 else 'REJECTED'})")
            
            # Status
            classification_correct = predicted_digit == actual_digit
            proof_system_working = real_score > 0.5 and fake_score < 0.5
            
            status = "✅" if classification_correct and proof_system_working else "⚠️"
            print(f"{status} System Status: Classification {'✓' if classification_correct else '✗'}, "
                  f"Proof System {'✓' if proof_system_working else '✗'}")
    
    # Final system summary
    print("\n" + "=" * 60)
    print("🎉 ZK Classification Demo Complete!")
    
    # Get final training statistics
    if training_stats["verifier_accuracy_real"] and training_stats["verifier_accuracy_fake"]:
        final_real_acc = training_stats["verifier_accuracy_real"][-1]
        final_fake_acc = training_stats["verifier_accuracy_fake"][-1]
        final_prover_acc = training_stats["prover_classification_accuracy"][-1]
        
        print(f"📈 Final Training Results:")
        print(f"   Prover Classification Accuracy: {final_prover_acc:.1%}")
        print(f"   Verifier Real Proof Accuracy: {final_real_acc:.1%}")
        print(f"   Verifier Fake Rejection Rate: {(1-final_fake_acc):.1%}")
        print(f"   Overall Verifier Performance: {(final_real_acc + (1-final_fake_acc))/2:.1%}")
    
    print(f"📊 Benchmark results saved to: benchmark_results/benchmark_results.json")
    print(f"📈 Training plots saved to: plots/training_progress.png")
    print("\n💡 Key Insight: Proofs are generated deterministically from the computation itself")
    print("   - No secret keys or trusted setup required")
    print("   - Verifier learns to distinguish computational consistency")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    demonstrate_zk_classification() 