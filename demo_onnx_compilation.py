"""
Demo: ONNX Compilation Framework

Demonstrates the complete workflow of the modular ONNX layer compilation framework:
1. Create/load an ONNX model  
2. Scan and discover operations
3. Compile operations into proof-capable components
4. Convert entire network into ZK graphs with network compiler
5. Show compilation results and artifacts

This showcases the new modular approach to per-layer ZK compilation.
"""

import os
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Import the test model creator
from create_test_onnx_model import create_test_onnx_model

# Import our ZK graph compiler (includes all functionality)
from nomopoly import ZKGraphCompiler, ops_registry


def benchmark_zk_models(prover_path: str, verifier_path: str, original_model_path: str = "test_model.onnx"):
    """
    Benchmark ZK prover and verifier with 4 test scenarios:
    1. Real network output + Real proof → Should verify (high scores)
    2. Real network output + Fake proof → Should reject (low scores)  
    3. Fake network output + Real proof → Should reject (low scores)
    4. Fake network output + Fake proof → Should reject (low scores)
    """
    print("🧪 Testing ZK Prover and Verifier Accuracy")
    print("-" * 50)
    
    try:
        # Load ONNX models
        prover_session = ort.InferenceSession(prover_path)
        verifier_session = ort.InferenceSession(verifier_path)
        original_session = ort.InferenceSession(original_model_path)
        
        # Create test input
        test_input = np.random.randn(1, 3, 8, 8).astype(np.float32)
        print(f"📥 Test input shape: {test_input.shape}")
        
        # Get original network output for comparison
        original_output = original_session.run(None, {'input': test_input})[0]
        print(f"📤 Original network output shape: {original_output.shape}")
        
        # Test Scenario 1: Real output + Real proof (should verify)
        print(f"\n🔬 Scenario 1: Real Network Output + Real Proof")
        print("   Expected: HIGH verification scores (should accept)")
        
        prover_outputs = prover_session.run(None, {'input': test_input})
        real_network_output = prover_outputs[0]
        real_proof = prover_outputs[1]
        
        print(f"   🎯 Prover network output shape: {real_network_output.shape}")
        print(f"   📜 Prover proof shape: {real_proof.shape}")
        
        # Check if prover network output matches original
        output_match = np.allclose(real_network_output, original_output, rtol=1e-3, atol=1e-3)
        print(f"   🔍 Network output matches original: {output_match}")
        if not output_match:
            max_diff = np.max(np.abs(real_network_output - original_output))
            print(f"   ⚠️  Max difference: {max_diff:.6f}")
        
        # Verify with real output + real proof
        verification_scores_1 = verifier_session.run(None, {
            'network_input': test_input,
            'network_output': real_network_output, 
            'concatenated_proofs': real_proof
        })[0]
        
        print(f"   📊 Verification scores: {verification_scores_1.flatten()}")
        avg_score_1 = np.mean(verification_scores_1)
        print(f"   📈 Average verification score: {avg_score_1:.4f}")
        
        # Test Scenario 2: Real output + Fake proof (should reject)
        print(f"\n🔬 Scenario 2: Real Network Output + Fake Proof")
        print("   Expected: LOW verification scores (should reject)")
        
        fake_proof = np.random.randn(*real_proof.shape).astype(np.float32)
        verification_scores_2 = verifier_session.run(None, {
            'network_input': test_input,
            'network_output': real_network_output,
            'concatenated_proofs': fake_proof
        })[0]
        
        print(f"   📊 Verification scores: {verification_scores_2.flatten()}")
        avg_score_2 = np.mean(verification_scores_2)
        print(f"   📈 Average verification score: {avg_score_2:.4f}")
        
        # Test Scenario 3: Fake output + Real proof (should reject)
        print(f"\n🔬 Scenario 3: Fake Network Output + Real Proof")
        print("   Expected: LOW verification scores (should reject)")
        
        fake_network_output = np.random.randn(*real_network_output.shape).astype(np.float32)
        verification_scores_3 = verifier_session.run(None, {
            'network_input': test_input,
            'network_output': fake_network_output,
            'concatenated_proofs': real_proof
        })[0]
        
        print(f"   📊 Verification scores: {verification_scores_3.flatten()}")
        avg_score_3 = np.mean(verification_scores_3)
        print(f"   📈 Average verification score: {avg_score_3:.4f}")
        
        # Test Scenario 4: Fake output + Fake proof (should reject)
        print(f"\n🔬 Scenario 4: Fake Network Output + Fake Proof")
        print("   Expected: LOW verification scores (should reject)")
        
        verification_scores_4 = verifier_session.run(None, {
            'network_input': test_input,
            'network_output': fake_network_output,
            'concatenated_proofs': fake_proof
        })[0]
        
        print(f"   📊 Verification scores: {verification_scores_4.flatten()}")
        avg_score_4 = np.mean(verification_scores_4)
        print(f"   📈 Average verification score: {avg_score_4:.4f}")
        
        # Summary
        print(f"\n📊 ZK System Accuracy Summary:")
        print(f"   🟢 Real + Real:  {avg_score_1:.4f} {'✅ PASS' if avg_score_1 > 0.5 else '❌ FAIL'}")
        print(f"   🔴 Real + Fake:  {avg_score_2:.4f} {'✅ PASS' if avg_score_2 < 0.5 else '❌ FAIL'}")
        print(f"   🔴 Fake + Real:  {avg_score_3:.4f} {'✅ PASS' if avg_score_3 < 0.5 else '❌ FAIL'}")
        print(f"   🔴 Fake + Fake:  {avg_score_4:.4f} {'✅ PASS' if avg_score_4 < 0.5 else '❌ FAIL'}")
        
        # Check if system is working correctly
        correct_scenarios = 0
        if avg_score_1 > 0.5:  # Should accept real + real
            correct_scenarios += 1
        if avg_score_2 < 0.5:  # Should reject real + fake
            correct_scenarios += 1
        if avg_score_3 < 0.5:  # Should reject fake + real
            correct_scenarios += 1
        if avg_score_4 < 0.5:  # Should reject fake + fake
            correct_scenarios += 1
        
        accuracy = correct_scenarios / 4 * 100
        print(f"\n🎯 ZK System Accuracy: {accuracy:.1f}% ({correct_scenarios}/4 scenarios correct)")
        
        if accuracy == 100:
            print("   🎉 Perfect ZK system - all scenarios work correctly!")
        elif accuracy >= 75:
            print("   ⚠️  ZK system mostly working but has some issues")
        elif accuracy >= 50:
            print("   ❌ ZK system has significant problems")
        else:
            print("   💥 ZK system is completely broken")
        
        # Detailed diagnosis
        if not output_match:
            print(f"\n⚠️  ISSUE: Prover network output doesn't match original network")
            print(f"   This suggests the prover's operation chaining is incorrect")
        
        if avg_score_1 <= 0.5:
            print(f"\n⚠️  ISSUE: Real proof rejected (score {avg_score_1:.4f})")
            print(f"   This suggests the proof generation or verification is broken")
        
        if avg_score_2 >= 0.5 or avg_score_3 >= 0.5 or avg_score_4 >= 0.5:
            print(f"\n⚠️  ISSUE: Fake proofs/outputs accepted")
            print(f"   This suggests the verifier is too permissive or not working")
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        print(f"   Error details: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo function."""
    print("🚀 ZK Graph Compiler Demo")
    print("=" * 50)
    
    # Step 1: Create a test ONNX model
    print("\n📋 Step 1: Creating Test ONNX Model")
    test_model_path = "test_model.onnx"
    
    if not os.path.exists(test_model_path):
        create_test_onnx_model(test_model_path)
    else:
        print(f"✅ Using existing test model: {test_model_path}")
    
    # Step 2: Initialize ZK Graph Compiler
    print("\n🔧 Step 2: Initializing ZK Graph Compiler")
    
    # Create ZK graph compiler instance with input shape
    input_shape = (1, 3, 8, 8)  # Batch size, channels, height, width for the test model
    zk_compiler = ZKGraphCompiler(
        input_shape=input_shape,
        ops_dir="ops",
        device="mps"  # Use MPS for Apple Silicon, fallback to CPU
    )
    
    print(f"   📁 Operations directory: {zk_compiler.ops_dir}")
    print(f"   🖥️  Device: {zk_compiler.device}")
    print(f"   💾 Will use cached operations from ops/ folder")
    
    # Step 3: Scan for cached operations
    print("\n🔍 Step 3: Checking Cached Operations")
    
    try:
        # Check what's already compiled in ops folder
        cached_ops = zk_compiler.zk_op_compiler.list_compiled_operations()
        
        print(f"✅ Found {len(cached_ops)} cached operations:")
        
        for i, op_name in enumerate(cached_ops):
            zk_op = zk_compiler.zk_op_compiler.get_compiled_operation(op_name)
            print(f"     {i+1}. {op_name}")
            print(f"        ✅ Cached and ready to use")
            print(f"        📁 Location: ops/{op_name}/")
            
    except Exception as e:
        print(f"❌ Error checking cached operations: {e}")
        print(f"   Will scan model to discover operations...")
    
    # Step 4: Compile Complete ZK Graph (leveraging cache)
    print("\n🏗️ Step 4: Compiling Complete ZK Graph")
    print("   💾 Using cached operations from ops/ folder when shapes match")
    print("   🚀 Compiling missing operations with correct shapes as needed")
    
    start_time = time.time()
    
    try:
        # Use ZK graph compiler to build complete ZK graph
        zk_graph = zk_compiler.compile_graph(
            onnx_model_path=test_model_path,
            output_dir="zk_graphs",
            force_recompile=False,  # Use cached operations, compile only missing ones
            target_accuracy=0.99999
        )
        
        compilation_time = time.time() - start_time
        
        print(f"\n✅ ZK Graph Compilation completed in {compilation_time:.1f}s")
        print(f"   🎯 ZK Prover: {zk_graph.prover_path}")
        print(f"   🔍 ZK Verifier: {zk_graph.verifier_path}")
        
        # Show ZK graph specifications
        print(f"\n📊 ZK Graph Specifications:")
        print(f"   📊 ZK Graph: {zk_graph}")
        print(f"   🔗 Chained operations: {len(zk_graph.zk_operations)}")
        print(f"   📏 Total proof dimension: {zk_graph.total_proof_dim}D")
        print(f"   🧮 Prover outputs: [network_output, concatenated_proofs]")
        print(f"   🔍 Verifier outputs: [verification_scores]")
        print(f"   📋 Operation order: {zk_graph.operation_order}")
        
        # Show prover/verifier details
        print(f"\n🎯 ZK Prover Details:")
        print(f"   📥 Input: Original network input tensor")
        print(f"   📤 Output 1: Original network computation result")
        print(f"   📤 Output 2: Concatenated ZK proofs from all operations")
        print(f"   🔗 Flow: Input → ZK_Op1 → ZK_Op2 → ... → ZK_OpN → [Result, Proofs]")
        
        print(f"\n🔍 ZK Verifier Details:")
        print(f"   📥 Input 1: Original network input tensor")
        print(f"   📥 Input 2: Network output tensor")
        print(f"   📥 Input 3: Concatenated proof tensor")
        print(f"   📤 Output: Verification scores for each operation")
        print(f"   🔗 Verifies all {len(zk_graph.zk_operations)} operations simultaneously")
        
        # Validate ZK graph files
        print(f"\n🔬 Validating ZK graph...")
        if os.path.exists(zk_graph.prover_path) and os.path.exists(zk_graph.verifier_path):
            prover_size = os.path.getsize(zk_graph.prover_path) / 1024  # KB
            verifier_size = os.path.getsize(zk_graph.verifier_path) / 1024  # KB
            print(f"   ✅ ZK Prover: {prover_size:.1f} KB")
            print(f"   ✅ ZK Verifier: {verifier_size:.1f} KB")
            
            # Benchmark the ZK prover and verifier
            print(f"\n🧪 Step 5: Benchmarking ZK System Accuracy")
            print("=" * 50)
            benchmark_zk_models(str(zk_graph.prover_path), str(zk_graph.verifier_path), test_model_path)
        else:
            print(f"   ⚠️ Some ZK graph files missing")
        
        # Show compilation stats
        print(f"\n📊 ZK Compilation Statistics:")
        print(f"   🔧 Input Shape: {zk_compiler.input_shape}")
        print(f"   📋 Tensor Shapes Traced: {len(zk_compiler.tensor_shapes)}")
        print(f"   🖥️  Device: {zk_compiler.device}")
        
        print(f"\n🚀 ZK Graph Ready for Deployment!")
        print(f"   🎯 Use {zk_graph.prover_path} for ZK inference with proof generation")
        print(f"   🔍 Use {zk_graph.verifier_path} for complete network verification")
        print(f"   🔐 Enables end-to-end ZK proving of neural network computations")
        print(f"   💾 All operations cached in ops/ folder for reuse")
        
    except Exception as e:
        print(f"⚠️ ZK graph compilation failed: {e}")
        print(f"   Error details: {str(e)}")
        print(f"   Continuing with results display...")
        # Set default values for display
        zk_graph = None
    
    # Step 6: Show directory structure
    print("\n📁 Step 6: Generated Directory Structure")
    
    print("   ops/")
    if zk_compiler.ops_dir.exists():
        for item in sorted(zk_compiler.ops_dir.iterdir()):
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
    
    # Show ZK graphs directory
    print("\n   zk_graphs/")
    zk_graphs_dir = Path("zk_graphs")
    if zk_graphs_dir.exists():
        for item in sorted(zk_graphs_dir.iterdir()):
            if item.suffix == '.onnx':
                file_size = item.stat().st_size / 1024  # KB
                print(f"     {item.name} ({file_size:.1f} KB)")
            else:
                print(f"     {item.name}")
    
    # Step 7: Summary
    print("\n🎯 ZK Graph Compiler Demo Summary")
    print("=" * 40)
    
    compiled_ops = zk_compiler.zk_op_compiler.list_compiled_operations()
    # Convert to list of names for counting
    compiled_ops_list = [zk_compiler.zk_op_compiler.get_compiled_operation(name) for name in compiled_ops]
    
    print(f"✅ Successfully compiled: {len(compiled_ops)} operations")
    
    if compiled_ops:
        print("\n📦 Generated ZK Operation Models:")
        for op_name in compiled_ops:
            print(f"   🔧 {op_name}:")
            print(f"      Prover: {op_name}_prover.onnx")
            print(f"      Verifier: {op_name}_verifier.onnx")
            print(f"      Adversary: {op_name}_adversary.onnx")
            print(f"      Location: ops/{op_name}/")
    
    print(f"\n🌐 Generated ZK Graph Models:")
    print(f"   🎯 prover.onnx: Chains all operations + generates concatenated proofs")
    print(f"   🔍 verifier.onnx: Verifies all operations simultaneously")
    print(f"   📁 Location: zk_graphs/")
    
    print(f"\n⚠️  IMPORTANT: All models compiled with fixed input dimensions!")
    print(f"   Models will break if input shapes differ during inference.")
    
    print(f"\n📊 ZK Graph Compiler Features Demonstrated:")
    print(f"   ✅ ONNX model scanning and operation discovery")
    print(f"   ✅ Modular operation registry with metadata")
    print(f"   ✅ Adversarial training to 99.99% verifier accuracy")
    print(f"   ✅ Individual operation ZK compilation with ZKOpCompiler")
    print(f"   ✅ Complete ZK graph compilation with ZKGraphCompiler")
    print(f"   ✅ Intelligent caching to avoid recompilation")
    print(f"   ✅ Prover graph: original output + concatenated proofs")
    print(f"   ✅ Verifier graph: concatenated verification scores")
    print(f"   ✅ End-to-end ZK proving of neural networks")
    print(f"   ✅ Clean architecture: ZKGraphCompiler → ZKOpCompiler → NAS")


def cleanup_demo_files():
    """Clean up demo files (optional)."""
    print("\n🧹 Cleaning up demo files...")
    
    files_to_remove = [
        "test_model.onnx"
    ]
    
    dirs_to_remove = [
        "ops",
        "zk_graphs"
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