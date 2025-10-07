"""
Hugging Face Model Loader for Nomopoly

This module provides functionality to load models from Hugging Face Hub
and prepare them for direct PyTorch ZK compilation.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModel, AutoTokenizer, AutoConfig


class HuggingFaceModelLoader:
    """
    Loads models from Hugging Face Hub for direct PyTorch ZK compilation.
    """
    
    def __init__(self, cache_dir: str = "hf_models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("HuggingFaceLoader")
        
    def load_model(
        self, 
        model_name: str,
        max_length: int = 128,
        batch_size: int = 1
    ) -> Tuple[torch.nn.Module, torch.Tensor, Dict[str, Any]]:
        """
        Load a model from Hugging Face for PyTorch ZK compilation.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "JackFram/llama-68m")
            max_length: Maximum sequence length for the model
            batch_size: Batch size for compilation
            
        Returns:
            Tuple of (pytorch_model, dummy_input, model_info)
        """
        self.logger.info(f"🤗 Loading Hugging Face model: {model_name}")
        
        # Create model-specific cache directory
        model_cache_dir = self.cache_dir / model_name.replace("/", "_")
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model info already exists
        model_info_path = model_cache_dir / "model_info.json"
        if model_info_path.exists():
            self.logger.info(f"✅ Using cached model info: {model_info_path}")
            model_info = self._load_model_info(model_cache_dir)
        else:
            model_info = {}
        
        # Load model and tokenizer
        self.logger.info(f"📥 Loading PyTorch model from Hugging Face...")
        
        try:
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModel.from_pretrained(
                model_name, 
                torch_dtype=torch.float32,
                trust_remote_code=True  # Some custom models may need this
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input tensor
        dummy_input = torch.randint(0, config.vocab_size, (batch_size, max_length))
        
        # Test the model to ensure it works
        self.logger.info(f"🧪 Testing model with dummy input...")
        try:
            with torch.no_grad():
                test_output = model(dummy_input)
                self.logger.info(f"✅ Model test successful!")
                
                # Extract output information
                if hasattr(test_output, 'last_hidden_state'):
                    output_shape = test_output.last_hidden_state.shape
                elif isinstance(test_output, dict) and 'last_hidden_state' in test_output:
                    output_shape = test_output['last_hidden_state'].shape
                elif isinstance(test_output, (tuple, list)):
                    output_shape = test_output[0].shape
                else:
                    output_shape = test_output.shape
                
                self.logger.info(f"📊 Output shape: {output_shape}")
                
        except Exception as e:
            self.logger.error(f"❌ Model test failed: {e}")
            raise
        
        # Update model information
        model_info.update({
            "model_name": model_name,
            "config": config.to_dict(),
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 'unknown')),
            "num_attention_heads": getattr(config, 'num_attention_heads', getattr(config, 'n_head', 'unknown')),
            "max_length": max_length,
            "batch_size": batch_size,
            "model_type": config.model_type,
            "input_shape": dummy_input.shape,
            "output_shape": output_shape
        })
        
        self._save_model_info(model_cache_dir, model_info)
        
        return model, dummy_input, model_info
    
    def _save_model_info(self, cache_dir: Path, model_info: Dict[str, Any]):
        """Save model information to cache directory."""
        import json
        
        # Convert torch.Size objects to tuples for JSON serialization
        serializable_info = {}
        for key, value in model_info.items():
            if hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
                try:
                    serializable_info[key] = tuple(value)
                except:
                    serializable_info[key] = str(value)
            else:
                serializable_info[key] = value
        
        info_path = cache_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(serializable_info, f, indent=2)
    
    def _load_model_info(self, cache_dir: Path) -> Dict[str, Any]:
        """Load model information from cache directory."""
        import json
        
        info_path = cache_dir / "model_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
        return {}
    
    def get_recommended_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of recommended small LLaMA models suitable for ZK compilation.
        """
        return {
            "microsoft/DialoGPT-small": {
                "description": "Small conversational model (117M params)",
                "size": "117M",
                "use_case": "Dialogue generation",
                "recommended_max_length": 128
            },
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
                "description": "Tiny LLaMA model (1.1B params)",
                "size": "1.1B",
                "use_case": "Small-scale language modeling and chat",
                "recommended_max_length": 256
            },
            "microsoft/DialoGPT-medium": {
                "description": "Medium conversational model (345M params)",
                "size": "345M",
                "use_case": "Better dialogue generation",
                "recommended_max_length": 128
            },
            "JackFram/llama-68m": {
                "description": "Ultra-tiny LLaMA model (68M params)",
                "size": "68M",
                "use_case": "Lightweight language modeling",
                "recommended_max_length": 512
            },
            "JackFram/llama-160m": {
                "description": "Small LLaMA model (160M params)", 
                "size": "160M",
                "use_case": "Small language modeling",
                "recommended_max_length": 512
            },
            "princeton-nlp/Sheared-LLaMA-1.3B": {
                "description": "Sheared LLaMA model (1.3B params)",
                "size": "1.3B",
                "use_case": "Efficient language modeling",
                "recommended_max_length": 256
            }
        }


def create_hf_compilation_demo(model_name: str = "JackFram/llama-68m"):
    """
    Create a demo script for compiling Hugging Face models with PyTorch ZK compilation.
    """
    demo_script = f'''#!/usr/bin/env python3
"""
Hugging Face Model PyTorch ZK Compilation Demo

This script demonstrates loading a model from Hugging Face Hub
and compiling it with Nomopoly's native PyTorch ZK compilation system.
"""

import logging
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
        logger.info(f"  🤖 {{name}}: {{info['description']}} ({{info['size']}})")
    
    # Load and convert model
    model_name = "{model_name}"
    logger.info(f"\\n🔄 Loading model: {{model_name}}")
    
    try:
        pytorch_model, dummy_input, model_info = hf_loader.load_model(
            model_name=model_name,
            max_length=128,
            batch_size=1
        )
        
        logger.info(f"✅ PyTorch model loaded successfully!")
        logger.info(f"📊 Model info:")
        logger.info(f"  📝 Type: {{model_info.get('model_type', 'unknown')}}")
        logger.info(f"  🧠 Hidden size: {{model_info.get('hidden_size', 'unknown')}}")
        logger.info(f"  📚 Vocab size: {{model_info.get('vocab_size', 'unknown')}}")
        logger.info(f"  🔢 Layers: {{model_info.get('num_layers', 'unknown')}}")
        logger.info(f"  📐 Input shape: {{model_info.get('input_shape', 'unknown')}}")
        logger.info(f"  📏 Output shape: {{model_info.get('output_shape', 'unknown')}}")
        
        # Initialize PyTorch ZK compiler
        logger.info(f"\\n🔧 Initializing PyTorch ZK Compiler...")
        
        # Use dedicated ops directory for this model
        ops_dir = f"ops_pytorch_{{model_name.replace('/', '_')}}"
        
        compiler = PyTorchZKCompiler(
            ops_dir=ops_dir,
            device="mps"  # Change to "cuda" or "cpu" as needed
        )
        
        # Compile the PyTorch model directly
        logger.info(f"🚀 Starting PyTorch ZK compilation of {{model_name}}...")
        logger.info("⚠️  Note: This may take several minutes for larger models")
        logger.info(f"💾 Operations will be cached in: {{ops_dir}}/")
        
        zk_graph = compiler.compile_graph(
            model=pytorch_model,
            input_tensor=dummy_input
        )
        
        if zk_graph:
            logger.info(f"\\n🎉 PyTorch ZK compilation successful!")
            logger.info(f"📊 Compiled {{len(zk_graph.operations)}} operations")
            logger.info(f"🔐 Total proof dimension: {{zk_graph.total_proof_dimension}}D")
            
            # Test the compiled graph
            logger.info(f"\\n🧪 Testing compiled ZK graph...")
            with torch.no_grad():
                output, proof = zk_graph(dummy_input)
                logger.info(f"✅ ZK graph test successful!")
                logger.info(f"📐 Output shape: {{output.shape}}")
                logger.info(f"🔐 Proof shape: {{proof.shape}}")
            
            # Save the compiled graph
            import torch
            graph_path = f"llama_zk_graph_{{model_name.replace('/', '_')}}.pt"
            torch.save(zk_graph, graph_path)
            logger.info(f"💾 Saved ZK Graph: {{graph_path}}")
            
        else:
            logger.error("❌ PyTorch ZK compilation failed")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {{e}}")
        import traceback
        logger.error(f"Traceback: {{traceback.format_exc()}}")

if __name__ == "__main__":
    import torch
    main()
'''
    
    with open("demo_pytorch_llama_compilation.py", "w") as f:
        f.write(demo_script)
    
    print("✅ Created demo_pytorch_llama_compilation.py")


if __name__ == "__main__":
    # Create demo when this module is run directly
    create_hf_compilation_demo() 