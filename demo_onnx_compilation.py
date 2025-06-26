"""
Demo: ONNX Compilation Framework

Demonstrates the complete workflow of the modular ONNX layer compilation framework:
1. Create/load an ONNX model  
2. Scan and discover operations
3. Compile operations into proof-capable components
4. Validate compiled models
5. Show compilation results and artifacts

This showcases the new modular approach to per-layer ZK compilation.
"""

import os
import time
from pathlib import Path

# Import the test model creator
from create_test_onnx_model import create_test_onnx_model

# Import our compilation framework
from nomopoly import (
    compilation_framework,
    ops_registry,
    SupportedOp,
    ONNXCompilationFramework
)


def main():
    """Main demo function."""
    print("🚀 ONNX Compilation Framework Demo")
    print("=" * 50)
    
    # Step 1: Create a test ONNX model
    print("\n📋 Step 1: Creating Test ONNX Model")
    test_model_path = "test_model.onnx"
    
    if not os.path.exists(test_model_path):
        create_test_onnx_model(test_model_path)
    else:
        print(f"✅ Using existing test model: {test_model_path}")
    
    # Step 2: Initialize compilation framework
    print("\n🔧 Step 2: Initializing Compilation Framework")
    
    # Create a custom framework instance for demo (could also use global one)
    framework = ONNXCompilationFramework(
        ops_dir="ops",
        device="mps"  # Use MPS for Apple Silicon, fallback to CPU
    )
    
    print(f"   📁 Operations directory: {framework.ops_dir}")
    print(f"   🖥️  Device: {framework.device}")
    
    # Step 3: Scan the ONNX model
    print("\n🔍 Step 3: Scanning ONNX Model for Operations")
    
    try:
        discovered_ops = framework.scan_and_register_model(test_model_path)
        
        print(f"✅ Successfully scanned model")
        print(f"   📊 Discovered {len(discovered_ops)} unique operations:")
        
        for i, op in enumerate(discovered_ops):
            print(f"     {i+1}. {op.folder_name}")
            print(f"        Type: {op.op_type.value}")
            print(f"        Input shape: {op.input_shape}")
            print(f"        Output shape: {op.output_shape}")
            print(f"        Compiled: {'✅' if op.compilation_complete else '❌'}")
            
    except Exception as e:
        print(f"❌ Error scanning model: {e}")
        return
    
    # Step 4: Show registry status
    print("\n📊 Step 4: Operations Registry Status")
    ops_registry.print_registry_status()
    
    # Step 5: Compile operations to 99% accuracy
    print("\n⚙️ Step 5: Compiling Operations to 99% Accuracy")
    print("   🎯 Training until 99% verifier accuracy (max 1000 epochs)")
    
    start_time = time.time()
    
    try:
        compilation_results = framework.compile_uncompiled_operations(
            num_epochs=100,          # Minimum epochs before target check
            batch_size=16,           # Smaller batch for demo
            proof_dim=16,            # Smaller proof dimension for demo
            force_recompile=True,    # Force recompile to reach 99%
            target_accuracy=0.99,    # 99% target
            max_epochs=1000          # Maximum to prevent infinite training
        )
        
        compilation_time = time.time() - start_time
        
        print(f"\n✅ Compilation completed in {compilation_time:.1f}s")
        print(f"📊 Results:")
        
        successful = 0
        failed = 0
        
        for op_name, result in compilation_results.items():
            if result.get("success", False):
                successful += 1
                print(f"   ✅ {op_name}: Accuracy {result['final_verifier_accuracy']:.3f}")
            else:
                failed += 1
                print(f"   ❌ {op_name}: {result.get('error', 'Unknown error')}")
        
        print(f"\n📈 Summary: {successful} successful, {failed} failed")
        
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return
    
    # Step 6: Validate compiled models
    print("\n🔍 Step 6: Validating Compiled ONNX Models")
    
    try:
        validation_results = framework.validate_compiled_models()
        
        valid_count = sum(validation_results.values())
        total_count = len(validation_results)
        
        print(f"📊 Validation complete: {valid_count}/{total_count} operations have valid models")
        
    except Exception as e:
        print(f"⚠️  Validation error: {e}")
    
    # Step 7: Show final registry status
    print("\n📊 Step 7: Final Registry Status")
    ops_registry.print_registry_status()
    
    # Step 8: Show directory structure
    print("\n📁 Step 8: Generated Directory Structure")
    
    print("   ops/")
    if framework.ops_dir.exists():
        for item in sorted(framework.ops_dir.iterdir()):
            if item.is_dir():
                print(f"     {item.name}/")
                for subitem in sorted(item.iterdir()):
                    if subitem.is_dir():
                        print(f"       {subitem.name}/")
                        # Show contents of subdirectories (like plots)
                        for plot_file in sorted(subitem.iterdir()):
                            print(f"         {plot_file.name}")
                    else:
                        # Show file with size if it's an ONNX file
                        if subitem.suffix == '.onnx':
                            file_size = subitem.stat().st_size / 1024  # KB
                            print(f"       {subitem.name} ({file_size:.1f} KB)")
                        else:
                            print(f"       {subitem.name}")
    
    # Step 9: Summary
    print("\n🎯 Demo Summary")
    print("=" * 30)
    
    compiled_ops = framework.list_compiled_operations()
    uncompiled_ops = framework.list_uncompiled_operations()
    
    print(f"✅ Successfully compiled: {len(compiled_ops)} operations")
    print(f"⏳ Still need compilation: {len(uncompiled_ops)} operations")
    
    if compiled_ops:
        print("\n📦 Generated ONNX Models (all in operation folders):")
        for op in compiled_ops:
            print(f"   🔧 {op.folder_name}:")
            print(f"      Prover: {Path(op.prover_onnx_path).name}")
            print(f"      Verifier: {Path(op.verifier_onnx_path).name}")
            print(f"      Adversary: {Path(op.adversary_onnx_path).name}")
            print(f"      Location: ops/{op.folder_name}/")
    
    print(f"\n⚠️  IMPORTANT: All models compiled with fixed input dimensions!")
    print(f"   Models will break if input shapes differ during inference.")
    
    print(f"\n📊 Framework Features Demonstrated:")
    print(f"   ✅ ONNX model scanning and operation discovery")
    print(f"   ✅ Modular operation registry with metadata")
    print(f"   ✅ Adversarial training to 99% verifier accuracy")
    print(f"   ✅ Adaptive training with early stopping")
    print(f"   ✅ ONNX export of prover/verifier/adversary models")
    print(f"   ✅ Compilation progress tracking and logging")
    print(f"   ✅ Model validation and artifact management")
    print(f"   ✅ Comprehensive training plots for adversarial analysis")


def cleanup_demo_files():
    """Clean up demo files (optional)."""
    print("\n🧹 Cleaning up demo files...")
    
    files_to_remove = [
        "test_model.onnx"
    ]
    
    dirs_to_remove = [
        "ops"
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"   Removed: {file_path}")
    
    import shutil
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"   Removed directory: {dir_path}")
    
    print("✅ Cleanup complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise
    
    # Uncomment the next line if you want to clean up files after demo
    # cleanup_demo_files() 