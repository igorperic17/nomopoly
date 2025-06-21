"""
Nomopoly: Zero Knowledge Machine Learning Framework with Holographic Reduced Representations

A framework for training neural networks with zero-knowledge proof capabilities,
featuring fixed-size proofs using HRR and zkGAP-inspired adversarial training.
"""

from .networks import (
    HolographicMemory,
    HolographicWrapper,
    OriginalMNISTNet,
    ZKProverNet, 
    ZKVerifierNet, 
    ZKAdversarialNet,
    create_holographic_model,
    SimpleONNXComputation
)

from .training import ZKTrainer

from .benchmarks import ZKBenchmark

from .utils import (
    create_simple_onnx_graph,
    convert_pytorch_to_onnx,
    validate_onnx_model,
    OnnxHandler
)

__version__ = "0.3.0"
__author__ = "Nomopoly Team"
__description__ = "Zero Knowledge Machine Learning with Holographic Reduced Representations"

__all__ = [
    # Core HRR components
    "HolographicMemory",
    "HolographicWrapper", 
    "create_holographic_model",
    
    # Network architectures
    "OriginalMNISTNet",
    "ZKProverNet",
    "ZKVerifierNet", 
    "ZKAdversarialNet",
    "SimpleONNXComputation",
    
    # Training system
    "ZKTrainer",
    
    # Benchmarking
    "ZKBenchmark",
    
    # Utilities
    "create_simple_onnx_graph",
    "convert_pytorch_to_onnx", 
    "validate_onnx_model",
    "OnnxHandler",
    
    # Version info
    "__version__",
    "__author__",
    "__description__"
] 