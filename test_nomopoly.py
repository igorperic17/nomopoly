#!/usr/bin/env python3
"""
Simple test script to verify nomopoly package functionality.
"""

import torch
import numpy as np
import os

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    
    try:
        from nomopoly import (
            ZKProverNet, 
            ZKVerifierNet, 
            ZKAdversarialNet,
            AutoZKTraining,
            create_simple_onnx_graph,
            OnnxHandler,
            ZKMLBenchmark
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_network_creation():
    """Test that networks can be created."""
    print("\nTesting network creation...")
    
    try:
        from nomopoly import ZKProverNet, ZKVerifierNet, ZKAdversarialNet
        
        # Create networks
        prover = ZKProverNet(input_dim=2, output_dim=1, proof_dim=32)
        verifier = ZKVerifierNet(input_dim=2, output_dim=1, proof_dim=32)
        adversary = ZKAdversarialNet(input_dim=2, output_dim=1, proof_dim=32)
        
        print("✅ Networks created successfully")
        print(f"  Prover params: {sum(p.numel() for p in prover.parameters()):,}")
        print(f"  Verifier params: {sum(p.numel() for p in verifier.parameters()):,}")
        print(f"  Adversary params: {sum(p.numel() for p in adversary.parameters()):,}")
        
        return True, (prover, verifier, adversary)
    except Exception as e:
        print(f"❌ Network creation failed: {e}")
        return False, None

def test_forward_pass(networks):
    """Test forward passes through networks."""
    print("\nTesting forward passes...")
    
    try:
        prover, verifier, adversary = networks
        
        # Create test input
        test_input = torch.randn(4, 2)
        
        # Test prover
        output, proof = prover(test_input)
        print(f"✅ Prover forward pass: {test_input.shape} -> {output.shape}, {proof.shape}")
        
        # Test verifier
        verification = verifier(test_input, output, proof)
        print(f"✅ Verifier forward pass: verification shape {verification.shape}")
        
        # Test adversary
        fake_output, fake_proof = adversary(test_input)
        print(f"✅ Adversary forward pass: {test_input.shape} -> {fake_output.shape}, {fake_proof.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_onnx_creation():
    """Test ONNX graph creation."""
    print("\nTesting ONNX graph creation...")
    
    try:
        from nomopoly import create_simple_onnx_graph
        
        # Create temporary directory
        os.makedirs("temp_test", exist_ok=True)
        
        # Create ONNX graph
        onnx_path = create_simple_onnx_graph("temp_test/test_sum.onnx")
        
        if os.path.exists(onnx_path):
            print("✅ ONNX graph created successfully")
            
            # Clean up
            os.remove(onnx_path)
            os.rmdir("temp_test")
            
            return True
        else:
            print("❌ ONNX file not found")
            return False
            
    except Exception as e:
        print(f"❌ ONNX creation failed: {e}")
        return False

def test_training_setup():
    """Test training setup."""
    print("\nTesting training setup...")
    
    try:
        from nomopoly import ZKProverNet, ZKVerifierNet, ZKAdversarialNet, AutoZKTraining
        
        # Create small networks for testing
        prover = ZKProverNet(input_dim=2, output_dim=1, proof_dim=16, hidden_dims=(32, 32))
        verifier = ZKVerifierNet(input_dim=2, output_dim=1, proof_dim=16, hidden_dims=(32, 32))
        adversary = ZKAdversarialNet(input_dim=2, output_dim=1, proof_dim=16, hidden_dims=(32, 32))
        
        # Initialize trainer
        trainer = AutoZKTraining(prover, verifier, adversary)
        
        print("✅ Training setup successful")
        print(f"  Device: {trainer.device}")
        
        return True
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔐 NOMOPOLY - Package Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Network creation
    success, networks = test_network_creation()
    if success:
        tests_passed += 1
        
        # Test 3: Forward passes (only if networks created successfully)
        if test_forward_pass(networks):
            tests_passed += 1
    else:
        print("⏭️  Skipping forward pass test due to network creation failure")
    
    # Test 4: ONNX creation
    if test_onnx_creation():
        tests_passed += 1
    
    # Test 5: Training setup
    if test_training_setup():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! nomopoly is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 