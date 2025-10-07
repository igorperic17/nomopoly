#!/usr/bin/env python3
"""
Hugging Face Model Compilation Demo

This script demonstrates loading a model from Hugging Face Hub
and compiling it with Nomopoly's ZK compilation system.
"""

import logging
from nomopoly.huggingface_loader import HuggingFaceModelLoader
from nomopoly.zk_graph_compiler import ZKGraphCompiler

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger("HFDemo")
    
    logger.info("🤗 Hugging Face Model Compilation Demo")
    logger.info("=" * 50)
    
    # Initialize HF loader
    hf_loader = HuggingFaceModelLoader()
    
    # Show recommended models
    logger.info("📋 Recommended models for ZK compilation:")
    for name, info in hf_loader.get_recommended_models().items():
        logger.info(f"  🤖 {name}: {info['description']} ({info['size']})")
    
    # Load and convert model
    model_name = "JackFram/llama-68m"
    logger.info(f"\n🔄 Loading model: {model_name}")
    
    try:
        onnx_path, model_info = hf_loader.load_model(
            model_name=model_name,
            max_length=128,
            batch_size=1
        )
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📊 Model info:")
        logger.info(f"  📝 Type: {model_info.get('model_type', 'unknown')}")
        logger.info(f"  🧠 Hidden size: {model_info.get('hidden_size', 'unknown')}")
        logger.info(f"  📚 Vocab size: {model_info.get('vocab_size', 'unknown')}")
        logger.info(f"  🔢 Layers: {model_info.get('num_layers', 'unknown')}")
        
        # Validate ONNX
        if hf_loader.validate_onnx_model(onnx_path):
            logger.info("✅ ONNX model validation passed")
        else:
            logger.error("❌ ONNX model validation failed")
            return
        
        # Initialize ZK compiler
        logger.info(f"\n🔧 Initializing ZK Graph Compiler...")
        
        # Determine input shape from model info
        batch_size = model_info.get('batch_size', 1)
        max_length = model_info.get('max_length', 128)
        input_shape = (batch_size, max_length)
        
        compiler = ZKGraphCompiler(
            input_shape=input_shape,
            ops_dir="ops_hf",
            device="mps"  # Change to "cuda" or "cpu" as needed
        )
        
        # Compile the HF model
        logger.info(f"🚀 Starting ZK compilation of {model_name}...")
        logger.info("⚠️  Note: This may take several minutes for larger models")
        
        zk_graph = compiler.compile_graph(
            onnx_model_path=onnx_path,
            cache_only=False  # Compile missing operations
        )
        
        if zk_graph:
            logger.info(f"\n🎉 ZK compilation successful!")
            logger.info(f"📊 Compiled {len(zk_graph.operations)} operations")
            logger.info(f"🔐 Total proof dimension: {zk_graph.total_proof_dimension}D")
            logger.info(f"💾 ZK Prover size: {len(zk_graph.prover_onnx_bytes) / 1024:.1f} KB")
            logger.info(f"🔍 ZK Verifier size: {len(zk_graph.verifier_onnx_bytes) / 1024:.1f} KB")
        else:
            logger.error("❌ ZK compilation failed")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
