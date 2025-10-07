#!/usr/bin/env python3
"""
Hugging Face Model PyTorch ZK Compilation Demo

This script demonstrates loading a model from Hugging Face Hub
and compiling it with Nomopoly's native PyTorch ZK compilation system.
"""

import logging
import torch
from nomopoly.huggingface_loader import HuggingFaceModelLoader
from nomopoly.pytorch_graph_compiler import PyTorchZKCompiler

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger("HFPyTorchDemo")
    
    logger.info("🤗 Hugging Face PyTorch ZK Compilation Demo")
    logger.info("=" * 50)
    
    # Initialize HF loader
    hf_loader = HuggingFaceModelLoader()
    
    # Show recommended models
    logger.info("📋 Recommended models for PyTorch ZK compilation:")
    for name, info in hf_loader.get_recommended_models().items():
        logger.info(f"  🤖 {name}: {info['description']} ({info['size']})")
    
    # Load and convert model (using DialoGPT which works without extra dependencies)
    model_name = "microsoft/DialoGPT-small"
    logger.info(f"\n🔄 Loading model: {model_name}")
    
    try:
        pytorch_model, dummy_input, model_info = hf_loader.load_model(
            model_name=model_name,
            max_length=64,  # Smaller for faster demo
            batch_size=1
        )
        
        logger.info(f"✅ PyTorch model loaded successfully!")
        logger.info(f"📊 Model info:")
        logger.info(f"  📝 Type: {model_info.get('model_type', 'unknown')}")
        logger.info(f"  🧠 Hidden size: {model_info.get('hidden_size', 'unknown')}")
        logger.info(f"  📚 Vocab size: {model_info.get('vocab_size', 'unknown')}")
        logger.info(f"  🔢 Layers: {model_info.get('num_layers', 'unknown')}")
        logger.info(f"  📐 Input shape: {model_info.get('input_shape', 'unknown')}")
        logger.info(f"  📏 Output shape: {model_info.get('output_shape', 'unknown')}")
        
        # Initialize PyTorch ZK compiler
        logger.info(f"\n🔧 Initializing PyTorch ZK Compiler...")
        
        # Use dedicated ops directory for this model
        ops_dir = f"ops_pytorch_{model_name.replace('/', '_')}"
        
        compiler = PyTorchZKCompiler(
            ops_dir=ops_dir,
            device="cpu"  # Use CPU for better compatibility
        )
        
        # Compile the PyTorch model directly
        logger.info(f"🚀 Starting PyTorch ZK compilation of {model_name}...")
        logger.info("⚠️  Note: This may take several minutes for larger models")
        logger.info(f"💾 Operations will be cached in: {ops_dir}/")
        
        zk_graph = compiler.compile_graph(
            model=pytorch_model,
            input_tensor=dummy_input
        )
        
        if zk_graph:
            logger.info(f"\n🎉 PyTorch ZK compilation successful!")
            logger.info(f"📊 Compiled {len(zk_graph.operations)} operations")
            logger.info(f"🔐 Total proof dimension: {zk_graph.total_proof_dimension}D")
            
            # Test the compiled graph
            logger.info(f"\n🧪 Testing compiled ZK graph...")
            with torch.no_grad():
                output, proof = zk_graph(dummy_input)
                logger.info(f"✅ ZK graph test successful!")
                logger.info(f"📐 Output shape: {output.shape}")
                logger.info(f"🔐 Proof shape: {proof.shape}")
            
            # Save the compiled graph
            graph_path = f"llama_zk_graph_{model_name.replace('/', '_')}.pt"
            torch.save(zk_graph, graph_path)
            logger.info(f"💾 Saved ZK Graph: {graph_path}")
            
            # Export ZK graph structure to ONNX for visualization
            logger.info(f"\n📊 Exporting ZK graph structure to ONNX for visualization...")
            try:
                # Create a structural ONNX representation showing the ZK operation chain
                class ZKGraphONNXWrapper(torch.nn.Module):
                    def __init__(self, zk_graph):
                        super().__init__()
                        self.zk_graph = zk_graph
                        
                        # Create a simplified chain showing the operation types
                        self.operation_chain = torch.nn.Sequential()
                        
                        # Group operations by type for cleaner visualization
                        op_types = []
                        for op_id in zk_graph.operation_order:
                            op_type = op_id.split('_')[0]
                            op_types.append(op_type)
                        
                        # Create representative layers for each operation type
                        unique_ops = []
                        for op_type in ['embedding', 'layernorm', 'dropout']:
                            if op_type in op_types:
                                unique_ops.append(op_type)
                        
                        # Build simplified chain
                        for i, op_type in enumerate(unique_ops):
                            if op_type == 'embedding':
                                layer = torch.nn.Embedding(50257, 768)  # DialoGPT vocab size
                            elif op_type == 'layernorm':
                                layer = torch.nn.LayerNorm(768)
                            elif op_type == 'dropout':
                                layer = torch.nn.Identity()  # Show as identity for ONNX
                            else:
                                layer = torch.nn.Identity()
                            
                            self.operation_chain.add_module(f"zk_{op_type}_{i}", layer)
                        
                        # Single proof generator representing all 64 operations
                        self.global_proof_generator = torch.nn.Sequential(
                            torch.nn.AdaptiveAvgPool1d(1),  # Pool sequence dimension
                            torch.nn.Flatten(),
                            torch.nn.Linear(768, 4096),  # Direct to 4096D proof
                            torch.nn.Tanh()
                        )
                    
                    def forward(self, x):
                        # Convert input tokens to embeddings
                        if x.dtype != torch.long:
                            x = x.long()
                        
                        # Process through simplified operation chain
                        current = x
                        
                        # Embedding layer
                        if hasattr(self.operation_chain, 'zk_embedding_0'):
                            current = self.operation_chain.zk_embedding_0(current)
                        
                        # Apply remaining operations
                        for name, module in self.operation_chain.named_children():
                            if not name.startswith('zk_embedding'):
                                current = module(current)
                        
                        # Generate global proof
                        proof = self.global_proof_generator(current.transpose(1, 2))
                        
                        return current, proof
                
                wrapper = ZKGraphONNXWrapper(zk_graph)
                
                # Export ZK graph structure to ONNX
                zk_onnx_path = f"zk_graph_{model_name.replace('/', '_')}.onnx"
                torch.onnx.export(
                    wrapper,
                    dummy_input.float(),
                    zk_onnx_path,
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=False,  # Keep structure visible
                    input_names=['input'],
                    output_names=['output', 'proof'],
                    dynamic_axes={
                        'input': {0: 'batch_size', 1: 'sequence_length'},
                        'output': {0: 'batch_size', 1: 'sequence_length', 2: 'hidden_size'},
                        'proof': {0: 'proof_dimension'}
                    }
                )
                logger.info(f"📊 ZK graph structure exported: {zk_onnx_path}")
                
                # Also export original model for comparison
                original_onnx_path = f"original_model_{model_name.replace('/', '_')}.onnx"
                torch.onnx.export(
                    pytorch_model,
                    dummy_input,
                    original_onnx_path,
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size', 1: 'sequence_length'},
                        'output': {0: 'batch_size', 1: 'sequence_length', 2: 'hidden_size'},
                    }
                )
                logger.info(f"📊 Original model exported: {original_onnx_path}")
                
                # Verify ONNX models
                import onnx
                zk_onnx_model = onnx.load(zk_onnx_path)
                onnx.checker.check_model(zk_onnx_model)
                logger.info(f"✅ ZK ONNX model verified successfully")
                logger.info(f"📈 ZK ONNX graph has {len(zk_onnx_model.graph.node)} nodes")
                
                original_onnx_model = onnx.load(original_onnx_path)
                onnx.checker.check_model(original_onnx_model)
                logger.info(f"✅ Original ONNX model verified successfully")
                logger.info(f"📈 Original ONNX graph has {len(original_onnx_model.graph.node)} nodes")
                
                # Create detailed metadata
                metadata_path = f"zk_compilation_info_{model_name.replace('/', '_')}.json"
                import json
                zk_metadata = {
                    "model_name": model_name,
                    "original_model_info": model_info,
                    "zk_compilation": {
                        "total_operations": len(zk_graph.operations),
                        "proof_dimension": zk_graph.total_proof_dimension,
                        "operations_list": list(zk_graph.operations.keys()),
                        "operation_order": zk_graph.operation_order
                    },
                    "files": {
                        "zk_graph_pytorch": graph_path,
                        "zk_graph_onnx": zk_onnx_path,
                        "original_onnx": original_onnx_path
                    },
                    "visualization": {
                        "zk_graph_nodes": len(zk_onnx_model.graph.node),
                        "original_nodes": len(original_onnx_model.graph.node),
                        "netron_command": f"netron {zk_onnx_path}"
                    }
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(zk_metadata, f, indent=2)
                logger.info(f"📋 ZK compilation metadata saved: {metadata_path}")
                logger.info(f"💡 Visualize ZK graph with: netron {zk_onnx_path}")
                
            except Exception as e:
                logger.warning(f"⚠️ ONNX export failed (this is okay): {e}")
                logger.info("💡 ZK compilation still successful, ONNX export is optional for visualization")
            
        else:
            logger.error("❌ PyTorch ZK compilation failed")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
