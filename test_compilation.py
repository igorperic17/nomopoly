#!/usr/bin/env python3
"""
Test ONNX Compilation Framework

Test script to verify the ONNX compilation framework can successfully 
compile operations into verifiable circuits.
"""

import os
import shutil
from nomopoly import compilation_framework, ops_registry

def cleanup_previous_runs():
    """Clean up any previous test runs."""
    print("🧹 Cleaning up previous test runs...")
    
    dirs_to_clean = ['ops', 'compiled_ops']
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"   Removed: {dir_path}")

def test_model_scanning():
    """Test ONNX model scanning and operation discovery."""
    print("\n🔍 Testing ONNX Model Scanning")
    print("-" * 40)
    
    if not os.path.exists('test_model.onnx'):
        print("❌ test_model.onnx not found")
        print("   Please run: python create_test_onnx_model.py")
        return False
    
    try:
        discovered_ops = compilation_framework.scan_and_register_model('test_model.onnx')
        
        if discovered_ops:
            print(f"✅ Successfully discovered {len(discovered_ops)} operations:")
            for i, op in enumerate(discovered_ops):
                print(f"   {i+1}. {op.folder_name}")
                print(f"      Type: {op.op_type.value}")
                print(f"      Input: {op.input_shape}")
                print(f"      Output: {op.output_shape}")
            return True
        else:
            print("❌ No operations discovered")
            return False
            
    except Exception as e:
        print(f"❌ Error during scanning: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_operation():
    """Test compiling a single operation."""
    print("\n⚙️ Testing Single Operation Compilation")
    print("-" * 45)
    
    # Get all discovered operations
    all_ops = list(ops_registry.discovered_ops.values())
    
    if not all_ops:
        print("❌ No operations available for testing")
        return False
    
    # Find the simplest operation (ReLU if available, otherwise first one)
    test_op = None
    for op in all_ops:
        if op.op_type.value == 'Relu':
            test_op = op
            break
    
    if not test_op:
        test_op = all_ops[0]  # Use first available operation
    
    print(f"🎯 Testing operation: {test_op.folder_name}")
    print(f"   Type: {test_op.op_type.value}")
    print(f"   Shape: {test_op.input_shape} -> {test_op.output_shape}")
    
    try:
        from nomopoly import ONNXOperationCompiler
        compiler = ONNXOperationCompiler(device='mps')
        
        result = compiler.compile_operation(
            op_info=test_op,
            num_epochs=15,  # Short test
            batch_size=4,   # Small batch
            proof_dim=8     # Small proof dimension
        )
        
        if result['success']:
            print(f"✅ Compilation SUCCESSFUL!")
            print(f"   Verifier accuracy: {result['final_verifier_accuracy']:.3f}")
            print(f"   Adversary fool rate: {result['final_adversary_fool_rate']:.3f}")
            print(f"   Compilation time: {result.get('compilation_time', 0):.1f}s")
            
            # Check ONNX files
            onnx_files = []
            if os.path.exists('compiled_ops'):
                onnx_files = [f for f in os.listdir('compiled_ops') if f.endswith('.onnx')]
            
            print(f"   Generated {len(onnx_files)} ONNX files:")
            for f in onnx_files:
                print(f"     - {f}")
            
            return True
        else:
            print(f"❌ Compilation failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during compilation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_onnx_validation():
    """Test validation of compiled ONNX models."""
    print("\n🔍 Testing ONNX Model Validation")
    print("-" * 40)
    
    try:
        validation_results = compilation_framework.validate_compiled_models()
        
        if validation_results:
            valid_count = sum(validation_results.values())
            total_count = len(validation_results)
            
            print(f"📊 Validation results: {valid_count}/{total_count} operations valid")
            
            for op_name, is_valid in validation_results.items():
                status = "✅" if is_valid else "❌"
                print(f"   {status} {op_name}")
            
            return valid_count > 0
        else:
            print("⚠️  No compiled operations to validate")
            return False
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 ONNX Compilation Framework Test")
    print("=" * 50)
    
    # Step 1: Cleanup
    cleanup_previous_runs()
    
    # Step 2: Test model scanning
    scanning_success = test_model_scanning()
    if not scanning_success:
        print("\n❌ Model scanning failed - cannot continue")
        return
    
    # Step 3: Test single operation compilation
    compilation_success = test_single_operation()
    if not compilation_success:
        print("\n❌ Operation compilation failed")
        return
    
    # Step 4: Test ONNX validation
    validation_success = test_onnx_validation()
    
    # Summary
    print("\n🎯 Test Summary")
    print("=" * 20)
    print(f"   Scanning: {'✅' if scanning_success else '❌'}")
    print(f"   Compilation: {'✅' if compilation_success else '❌'}")
    print(f"   Validation: {'✅' if validation_success else '❌'}")
    
    if scanning_success and compilation_success and validation_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("   The ONNX compilation framework is working correctly!")
        print("   Operations can be compiled into verifiable circuits!")
    else:
        print("\n⚠️  Some tests failed")
        print("   Framework needs debugging")

if __name__ == "__main__":
    main() 