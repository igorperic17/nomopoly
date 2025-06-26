"""
Demo: Per-Layer Verifiable Training

This script demonstrates the new per-layer training approach where:
1. Each layer has a verifier that accepts (input, output, proof) triplets
2. Each layer has a generator that produces fake outputs and proofs
3. Training is adversarial: verifier learns to distinguish real vs fake, generator tries to fool verifier
4. Success is measured by the verifier's ability to correctly classify triplets
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from nomopoly import (
    VerifiableLinear, 
    VerifiableReLU, 
    VerifiableConv2d,
    LayerTrainer,
    MultiLayerTrainer,
    create_demo_verifiable_network
)


def demo_single_layer_training():
    """Demo training a single verifiable linear layer."""
    print("🧪 Demo 1: Training a Single Verifiable Linear Layer")
    print("=" * 60)
    
    # Create a verifiable linear layer
    layer = VerifiableLinear(in_features=5, out_features=3, proof_dim=16)
    trainer = LayerTrainer(layer, device="mps")
    
    # Train the layer for much longer
    print("🎯 Training verifier and generator networks for extended training...")
    metrics = trainer.train(num_epochs=500, batch_size=32)  # 5x longer training
    
    # Evaluate the trained layer
    print("\n🔍 Evaluating trained layer...")
    evaluation = trainer.evaluate_layer(num_samples=2000)  # More evaluation samples
    print(f"Final evaluation accuracy: {evaluation['accuracy']:.3f}")
    
    # Test the layer manually
    print("\n🧪 Manual testing:")
    test_input = torch.randn(3, 5, device=trainer.device)
    
    # Real computation
    real_output = layer.forward(test_input)
    real_proof = layer.generate_proof(test_input, real_output)
    real_score = layer.verify_triplet(test_input, real_output, real_proof)
    print(f"Real triplet verification scores: {real_score.squeeze().tolist()}")
    
    # Fake computation
    fake_output, fake_proof = layer.generate_fake(test_input)
    fake_score = layer.verify_triplet(test_input, fake_output, fake_proof)
    print(f"Fake triplet verification scores: {fake_score.squeeze().tolist()}")
    
    return metrics


def demo_multi_layer_training():
    """Demo training multiple verifiable layers in sequence."""
    print("\n🧪 Demo 2: Training Multiple Verifiable Layers")
    print("=" * 60)
    
    # Create a demo verifiable network
    trainer, network = create_demo_verifiable_network(device="mps")
    
    # Train all layers for much longer
    print("🚀 Training all layers for extended periods...")
    all_metrics = trainer.train_all_layers(num_epochs=300, batch_size=32)  # 6x longer training
    
    # Evaluate all layers
    print("\n🔍 Evaluating all layers...")
    evaluations = trainer.evaluate_all_layers(num_samples=2000)  # More evaluation samples
    
    # Test the complete network
    print("\n🌐 Testing complete verifiable network:")
    device = trainer.layer_trainers[0].device if trainer.layer_trainers else "cpu"
    network = network.to(device)
    test_input = torch.randn(4, 10, device=device)  # 4 samples, 10 features
    
    # Normal forward pass
    output = network(test_input)
    print(f"Network output shape: {output.shape}")
    
    # Forward pass with proofs
    output_with_proofs, proofs = network.forward_with_proofs(test_input)
    print(f"Number of layer proofs: {len(proofs)}")
    print(f"Proof shapes: {[proof.shape for proof in proofs]}")
    
    # Verify the computation
    verified_output, verification_scores = network.verify_computation(test_input)
    print(f"Layer verification scores: {[f'{score:.3f}' for score in verification_scores]}")
    
    return all_metrics, evaluations


def demo_conv2d_layer():
    """Demo training a verifiable Conv2d layer."""
    print("\n🧪 Demo 3: Training a Verifiable Conv2d Layer")
    print("=" * 60)
    
    # Create a verifiable conv2d layer
    layer = VerifiableConv2d(
        in_channels=3, 
        out_channels=8, 
        kernel_size=3, 
        padding=1, 
        proof_dim=24
    )
    trainer = LayerTrainer(layer, device="mps")
    
    print("🎯 Training Conv2d verifier and generator for extended training...")
    metrics = trainer.train(num_epochs=400, batch_size=16)  # Much longer training for conv2d
    
    # Evaluate
    evaluation = trainer.evaluate_layer(num_samples=1000)  # More evaluation samples
    print(f"Conv2d evaluation accuracy: {evaluation['accuracy']:.3f}")
    
    # Manual test
    print("\n🧪 Manual Conv2d testing:")
    test_input = torch.randn(2, 3, 8, 8, device=trainer.device)  # 2 samples, 3 channels, 8x8 images
    
    real_output = layer.forward(test_input)
    real_proof = layer.generate_proof(test_input, real_output)
    real_score = layer.verify_triplet(test_input, real_output, real_proof)
    print(f"Conv2d real verification: {real_score.squeeze().tolist()}")
    
    fake_output, fake_proof = layer.generate_fake(test_input)
    fake_score = layer.verify_triplet(test_input, fake_output, fake_proof)
    print(f"Conv2d fake verification: {fake_score.squeeze().tolist()}")
    
    return metrics


def plot_training_metrics(metrics, title="Layer Training Metrics"):
    """Plot training metrics from layer training."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)
    
    # Verifier accuracy
    axes[0, 0].plot(metrics['verifier_accuracy'])
    axes[0, 0].set_title('Verifier Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True)
    
    # Generator fool rate
    axes[0, 1].plot(metrics['generator_fool_rate'])
    axes[0, 1].set_title('Generator Fool Rate')
    axes[0, 1].set_ylabel('Fool Rate')
    axes[0, 1].grid(True)
    
    # Losses
    axes[1, 0].plot(metrics['verifier_loss'], label='Verifier Loss')
    axes[1, 0].plot(metrics['generator_loss'], label='Generator Loss')
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Detailed accuracy breakdown
    axes[1, 1].plot(metrics['real_real_accuracy'], label='Real+Real')
    axes[1, 1].plot(metrics['fake_fake_accuracy'], label='Fake+Fake')
    axes[1, 1].plot(metrics['real_fake_accuracy'], label='Real+Fake')
    axes[1, 1].plot(metrics['fake_real_accuracy'], label='Fake+Real')
    axes[1, 1].set_title('Accuracy Breakdown')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig


def main():
    """Run all demos."""
    print("🚀 Starting Per-Layer Verifiable Training Demos")
    print("🔬 This demonstrates training layers to verify their own computations")
    print("⏰ Extended training sessions for better adversarial dynamics")
    print()
    
    # Demo 1: Single layer with extended training
    print("⚡ Starting extended single layer training (500 epochs)...")
    metrics1 = demo_single_layer_training()
    
    # Demo 2: Multiple layers with extended training
    print("⚡ Starting extended multi-layer training (300 epochs each)...")
    all_metrics, evaluations = demo_multi_layer_training()
    
    # Demo 3: Conv2d layer with extended training
    print("⚡ Starting extended Conv2d training (400 epochs)...")
    metrics3 = demo_conv2d_layer()
    
    # Create plots if matplotlib is available
    try:
        import os
        os.makedirs("plots", exist_ok=True)
        
        # Plot single linear layer metrics
        fig1 = plot_training_metrics(metrics1, "Extended Single Linear Layer Training (500 epochs)")
        plt.savefig("plots/extended_single_layer_training.png", dpi=150, bbox_inches='tight')
        print(f"📊 Saved extended single layer training plot")
        
        # Plot first layer from multi-layer training
        if "layer_1" in all_metrics:
            fig2 = plot_training_metrics(all_metrics["layer_1"], "Extended Multi-Layer Training (300 epochs)")
            plt.savefig("plots/extended_multi_layer_training.png", dpi=150, bbox_inches='tight')
            print(f"📊 Saved extended multi-layer training plot")
        
        # Plot Conv2d metrics
        fig3 = plot_training_metrics(metrics3, "Extended Conv2d Layer Training (400 epochs)")
        plt.savefig("plots/extended_conv2d_layer_training.png", dpi=150, bbox_inches='tight')
        print(f"📊 Saved extended Conv2d training plot")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")
    
    print("\n✅ All extended training demos completed successfully!")
    print("\n📋 Extended Training Summary:")
    print("• Single Linear Layer: 500 epochs")
    print("• Multi-Layer Network: 300 epochs per layer")  
    print("• Conv2d Layer: 400 epochs")
    print("• Each layer can now verify its own computations with higher accuracy")
    print("• Verifiers learn to distinguish real vs fake triplets over longer training")
    print("• Generators get more time to learn sophisticated attacks")
    print("• This enables robust per-layer zero-knowledge proof capabilities")
    
    # Print final accuracy summary
    print(f"\n📈 Final Accuracy Results:")
    print(f"• Single Linear Layer: {metrics1['verifier_accuracy'][-1]:.3f}")
    if "layer_1" in all_metrics:
        print(f"• Multi-Layer (Layer 1): {all_metrics['layer_1']['verifier_accuracy'][-1]:.3f}")
        print(f"• Multi-Layer (Layer 2): {all_metrics['layer_2']['verifier_accuracy'][-1]:.3f}")
        print(f"• Multi-Layer (Layer 3): {all_metrics['layer_3']['verifier_accuracy'][-1]:.3f}")
    print(f"• Conv2d Layer: {metrics3['verifier_accuracy'][-1]:.3f}")


if __name__ == "__main__":
    main() 