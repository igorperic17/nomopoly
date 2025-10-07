#!/usr/bin/env python3
"""
LLaMA Model ZK Compilation Demo

This script demonstrates loading LLaMA models from Hugging Face Hub
and compiling them with Nomopoly's ZK compilation system.
"""

import logging
import sys
from nomopoly.huggingface_loader import HuggingFaceModelLoader
from nomopoly.zk_graph_compiler import ZKGraphCompiler

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger("LLaMADemo")
    
    logger.info("🦙 LLaMA Model ZK Compilation Demo")
    logger.info("=" * 50)
    
    # Initialize HF loader
    hf_loader = HuggingFaceModelLoader()
    
    # Show recommended LLaMA models
    logger.info("📋 Recommended LLaMA models for ZK compilation:")
    recommended_models = hf_loader.get_recommended_models()
    llama_models = {k: v for k, v in recommended_models.items() if 'llama' in k.lower()}
    
    for i, (name, info) in enumerate(llama_models.items(), 1):
        logger.info(f"  {i}. 🦙 {name}: {info['description']} ({info['size']})")
    
    # Let user choose model or use default
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            model_idx = int(sys.argv[1]) - 1
            model_names = list(llama_models.keys())
            if 0 <= model_idx < len(model_names):
                model_name = model_names[model_idx]
            else:
                logger.error(f"❌ Invalid model index: {sys.argv[1]}")
                return
        else:
            model_name = sys.argv[1]
    else:
        # Default to the smallest LLaMA model
        model_name = "JackFram/llama-68m"
    
    logger.info(f"\n🔄 Loading LLaMA model: {model_name}")
    
    try:
        onnx_path, model_info = hf_loader.load_model(
            model_name=model_name,
            max_length=128,  # Smaller sequence length for ZK compilation
            batch_size=1
        )
        
        logger.info(f"✅ LLaMA model loaded successfully!")
        logger.info(f"📊 Model info:")
        logger.info(f"  📝 Type: {model_info.get('model_type', 'unknown')}")
        logger.info(f"  🧠 Hidden size: {model_info.get('hidden_size', 'unknown')}")
        logger.info(f"  📚 Vocab size: {model_info.get('vocab_size', 'unknown')}")
        logger.info(f"  🔢 Layers: {model_info.get('num_layers', 'unknown')}")
        logger.info(f"  🎯 Attention heads: {model_info.get('num_attention_heads', 'unknown')}")
        
        # Validate ONNX
        if hf_loader.validate_onnx_model(onnx_path):
            logger.info("✅ ONNX model validation passed")
        else:
            logger.error("❌ ONNX model validation failed")
            return
        
        # Initialize ZK compiler
        logger.info(f"\n🔧 Initializing ZK Graph Compiler for LLaMA...")
        
        # Determine input shape from model info
        batch_size = model_info.get('batch_size', 1)
        max_length = model_info.get('max_length', 128)
        input_shape = (batch_size, max_length)
        
        # Use dedicated ops directory for LLaMA models
        ops_dir = f"ops_llama_{model_name.replace('/', '_')}"
        
        compiler = ZKGraphCompiler(
            input_shape=input_shape,
            ops_dir=ops_dir,
            device="mps"  # Change to "cuda" or "cpu" as needed
        )
        
        # Compile the LLaMA model
        logger.info(f"🚀 Starting ZK compilation of {model_name}...")
        logger.info("⚠️  Note: This may take several minutes for larger models")
        logger.info(f"💾 Operations will be cached in: {ops_dir}/")
        
        zk_graph = compiler.compile_graph(
            onnx_model_path=onnx_path,
            cache_only=False  # Compile missing operations
        )
        
        if zk_graph:
            logger.info(f"\n🎉 LLaMA ZK compilation successful!")
            logger.info(f"📊 Compiled {len(zk_graph.operations)} operations")
            logger.info(f"🔐 Total proof dimension: {zk_graph.total_proof_dimension}D")
            logger.info(f"💾 ZK Prover size: {len(zk_graph.prover_onnx_bytes) / 1024:.1f} KB")
            logger.info(f"🔍 ZK Verifier size: {len(zk_graph.verifier_onnx_bytes) / 1024:.1f} KB")
            
            # Save the compiled models
            prover_path = f"llama_zk_prover_{model_name.replace('/', '_')}.onnx"
            verifier_path = f"llama_zk_verifier_{model_name.replace('/', '_')}.onnx"
            
            with open(prover_path, 'wb') as f:
                f.write(zk_graph.prover_onnx_bytes)
            with open(verifier_path, 'wb') as f:
                f.write(zk_graph.verifier_onnx_bytes)
                
            logger.info(f"💾 Saved ZK Prover: {prover_path}")
            logger.info(f"💾 Saved ZK Verifier: {verifier_path}")
            
        else:
            logger.error("❌ LLaMA ZK compilation failed")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
🦙 LLaMA Model ZK Compilation Demo

Usage:
  python demo_llama_compilation.py                    # Use default tiny LLaMA (68M)
  python demo_llama_compilation.py 1                  # Use model #1 from list
  python demo_llama_compilation.py 2                  # Use model #2 from list
  python demo_llama_compilation.py JackFram/llama-68m # Use specific model
  
Available Models:
  1. JackFram/llama-68m (68M params) - Ultra-tiny
  2. JackFram/llama-160m (160M params) - Small
  3. TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B params) - Medium
  4. princeton-nlp/Sheared-LLaMA-1.3B (1.3B params) - Large
        """)
        sys.exit(0)
    
    main() 