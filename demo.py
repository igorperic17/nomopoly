#!/usr/bin/env python3
"""
Demo script showcasing Zero Knowledge Machine Learning with any pretrained classifier.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from nomopoly.networks import ZKVerifierNet, ZKAdversarialNet
from nomopoly.training import GeneralZKTraining, train_pretrained_classifier
from nomopoly.benchmarks import ZKBenchmark


def analyze_proof_distributions(trainer, test_loader):
    """Analyze the distributions of real vs fake proofs to understand the problem."""
    print("\n🔍 ANALYZING PROOF DISTRIBUTIONS")
    print("=" * 50)
    
    trainer.classifier.eval()
    trainer.adversary.eval()
    
    real_proofs = []
    fake_proofs = []
    
    with torch.no_grad():
        # Collect proofs from a batch
        batch_data, batch_labels = next(iter(test_loader))
        batch_data = batch_data[:32].to(trainer.device)
        batch_labels = batch_labels[:32]
        
        # Generate real proofs
        real_log_probs = trainer.get_classifier_predictions(batch_data)
        real_proof_batch = trainer.generate_computational_proof(batch_data, real_log_probs)
        
        # Generate fake proofs  
        fake_log_probs, fake_proof_batch = trainer.adversary(batch_data)
        
        real_proofs = real_proof_batch.cpu().numpy()
        fake_proofs = fake_proof_batch.cpu().numpy()
    
    # Statistical analysis
    print(f"Real proofs stats:")
    print(f"  Mean: {np.mean(real_proofs):.4f}, Std: {np.std(real_proofs):.4f}")
    print(f"  Min: {np.min(real_proofs):.4f}, Max: {np.max(real_proofs):.4f}")
    
    print(f"Fake proofs stats:")
    print(f"  Mean: {np.mean(fake_proofs):.4f}, Std: {np.std(fake_proofs):.4f}")
    print(f"  Min: {np.min(fake_proofs):.4f}, Max: {np.max(fake_proofs):.4f}")
    
    # Calculate separability metrics
    real_mean = np.mean(real_proofs, axis=0)
    fake_mean = np.mean(fake_proofs, axis=0)
    
    euclidean_distance = np.linalg.norm(real_mean - fake_mean)
    cosine_similarity = np.dot(real_mean, fake_mean) / (np.linalg.norm(real_mean) * np.linalg.norm(fake_mean))
    
    print(f"\nSeparability metrics:")
    print(f"  Euclidean distance between means: {euclidean_distance:.4f}")
    print(f"  Cosine similarity: {cosine_similarity:.4f}")
    
    # Test verifier predictions
    trainer.verifier.eval()
    with torch.no_grad():
        real_scores = trainer.verifier(batch_data, real_log_probs, real_proof_batch).cpu().numpy()
        fake_scores = trainer.verifier(batch_data, fake_log_probs, fake_proof_batch).cpu().numpy()
    
    print(f"\nVerifier predictions:")
    print(f"  Real proof scores: mean={np.mean(real_scores):.4f}, std={np.std(real_scores):.4f}")
    print(f"  Fake proof scores: mean={np.mean(fake_scores):.4f}, std={np.std(fake_scores):.4f}")
    
    # Plot distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(real_proofs.flatten(), bins=50, alpha=0.7, label='Real Proofs', density=True)
    plt.hist(fake_proofs.flatten(), bins=50, alpha=0.7, label='Fake Proofs', density=True)
    plt.title('Proof Value Distributions')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(real_scores.flatten(), bins=20, alpha=0.7, label='Real Proof Scores', density=True)
    plt.hist(fake_scores.flatten(), bins=20, alpha=0.7, label='Fake Proof Scores', density=True)
    plt.title('Verifier Score Distributions')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(real_mean[:20], 'b-', label='Real Proof Mean', linewidth=2)
    plt.plot(fake_mean[:20], 'r-', label='Fake Proof Mean', linewidth=2)
    plt.title('Proof Patterns (first 20 dims)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/proof_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Proof analysis plot saved to: plots/proof_analysis.png")
    
    return euclidean_distance, cosine_similarity


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


def demonstrate_general_zk_system():
    """Demonstrate the general ZK system with pretrained classifier."""
    print("🔐 General Zero Knowledge ML System Demo")
    print("=" * 60)
    
    # Determine best device
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using CUDA acceleration")
    else:
        device = "cpu"
        print("🐌 Using CPU (consider upgrading for faster training)")
    
    # Step 1: Train/load pretrained classifier
    print("\n📚 Step 1: Setting up pretrained classifier...")
    pretrained_classifier = train_pretrained_classifier(device=device, epochs=8)
    
    # Test classifier accuracy
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    pretrained_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = pretrained_classifier(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    classifier_acc = 100. * correct / total
    print(f"✅ Pretrained classifier test accuracy: {classifier_acc:.2f}%")
    
    # Step 2: Initialize ZK system
    print(f"\n🔧 Step 2: Setting up ZK verification networks...")
    input_dim = 196  # 14x14 flattened
    output_dim = 10  # MNIST digits
    proof_dim = 64
    
    verifier = ZKVerifierNet(input_dim, output_dim, proof_dim, (256, 128))
    adversary = ZKAdversarialNet(input_dim, output_dim, proof_dim, (128, 256, 128))
    
    print(f"Networks: Verifier ({sum(p.numel() for p in verifier.parameters()):,} params), "
          f"Adversary ({sum(p.numel() for p in adversary.parameters()):,} params)")
    
    # Step 3: Initialize training system
    trainer = GeneralZKTraining(
        pretrained_classifier=pretrained_classifier,
        verifier=verifier,
        adversary=adversary,
        device=device
    )
    
    # Step 4: Train ZK verification system
    print(f"\n🏋️ Step 3: Training ZK verification system...")
    training_stats = trainer.train(num_epochs=25, num_samples=3000)
    
    # Step 5: Analyze proof distributions
    test_loader = trainer.load_mnist_data(500)
    euclidean_dist, cosine_sim = analyze_proof_distributions(trainer, test_loader)
    
    # Diagnose the problem
    print(f"\n🔬 PROOF ANALYSIS:")
    print(f"  Euclidean distance: {euclidean_dist:.4f}")
    print(f"  Cosine similarity: {cosine_sim:.4f}")
    
    if euclidean_dist > 2.0:
        print("✅ GOOD: Proof distributions are well separated")
    elif euclidean_dist > 0.5:
        print("⚠️ MODERATE: Some separation between proof types")
    else:
        print("❌ POOR: Proof distributions are too similar")
    
    # Step 6: Performance analysis
    final_real_acc = training_stats["verifier_accuracy_real"][-1]
    final_fake_acc = training_stats["verifier_accuracy_fake"][-1]
    verifier_binary_acc = (final_real_acc + (1 - final_fake_acc)) / 2
    
    print(f"\n📈 FINAL PERFORMANCE ANALYSIS:")
    print(f"   Pretrained Classifier Accuracy: {classifier_acc:.1f}%")
    print(f"   Verifier Real Proof Accuracy: {final_real_acc:.1%}")
    print(f"   Verifier Fake Rejection Rate: {(1-final_fake_acc):.1%}")
    print(f"   Verifier Binary Classifier Accuracy: {verifier_binary_acc:.1%}")
    
    if verifier_binary_acc > 0.8:
        print("🎉 SUCCESS: Verifier successfully learned to distinguish proofs!")
        success_level = "EXCELLENT"
    elif verifier_binary_acc > 0.7:
        print("✅ GOOD: Verifier shows strong discrimination ability")
        success_level = "GOOD"
    elif verifier_binary_acc > 0.6:
        print("⚠️ MODERATE: Verifier shows some discrimination")
        success_level = "PARTIAL"
    else:
        print("❌ FAILURE: Verifier cannot distinguish proofs")
        success_level = "FAILED"
    
    # Plot training progress
    trainer.plot_training_progress(training_stats, "plots/training_progress.png")
    
    # Step 7: Interactive demonstration (only if system works)
    if verifier_binary_acc > 0.6:
        print(f"\n🎮 Interactive Demo (System Performance: {success_level})")
        print("-" * 50)
        
        trainer.classifier.eval()
        trainer.verifier.eval()
        trainer.adversary.eval()
        
        with torch.no_grad():
            for demo_round in range(3):
                print(f"\n--- Round {demo_round + 1} ---")
                
                # Get a real MNIST sample
                test_input, actual_digit = get_real_mnist_sample()
                test_input = test_input.to(device)
                
                print(f"🎯 Ground Truth: Digit {actual_digit}")
                
                # Get classifier prediction
                real_log_probs = trainer.get_classifier_predictions(test_input)
                probs = torch.exp(real_log_probs)
                predicted_digit = torch.argmax(real_log_probs, dim=1).item()
                confidence = torch.max(probs, dim=1)[0].item()
                
                # Generate computational proof
                computational_proof = trainer.generate_computational_proof(test_input, real_log_probs)
                
                # Verify computational proof
                real_verification = trainer.verifier(test_input, real_log_probs, computational_proof)
                real_score = real_verification.item()
                
                # Generate fake proof from adversary
                fake_log_probs, fake_proof = trainer.adversary(test_input)
                fake_verification = trainer.verifier(test_input, fake_log_probs, fake_proof)
                fake_score = fake_verification.item()
                
                print(f"🤖 Classifier Prediction: Digit {predicted_digit} (confidence: {confidence:.1%})")
                print(f"✅ Computational Proof Score: {real_score:.3f} ({'ACCEPTED' if real_score > 0.5 else 'REJECTED'})")
                print(f"❌ Adversarial Proof Score: {fake_score:.3f} ({'ACCEPTED' if fake_score > 0.5 else 'REJECTED'})")
                
                # Status
                classification_correct = predicted_digit == actual_digit
                proof_system_working = real_score > 0.5 and fake_score < 0.5
                
                status = "✅" if classification_correct and proof_system_working else "⚠️"
                print(f"{status} Status: Classification {'✓' if classification_correct else '✗'}, "
                      f"Proof Verification {'✓' if proof_system_working else '✗'}")
    else:
        print(f"\n⚠️ Skipping interactive demo - verifier performance too low ({verifier_binary_acc:.1%})")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 GENERAL ZK SYSTEM DEMO COMPLETE")
    print("=" * 60)
    print(f"📊 Results saved to: plots/")
    print(f"🎯 Final System Performance: {success_level}")
    print("\n💡 Key Insights:")
    print("   ✓ Works with any pretrained classifier")
    print("   ✓ No trusted setup required")
    print("   ✓ Uses computational proofs for verification")
    print(f"   ✓ Runs efficiently on {device.upper()}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    demonstrate_general_zk_system() 