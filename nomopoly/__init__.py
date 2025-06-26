"""
Nomopoly: Zero Knowledge Machine Learning Framework with Holographic Reduced Representations

A framework for training neural networks with zero-knowledge proof capabilities,
featuring fixed-size proofs using HRR and GAN-inspired adversarial training.
"""

from .networks import (
    HolographicMemory,
    HolographicWrapper,
    OriginalMNISTNet,
    ZKProverNet, 
    ZKVerifierNet, 
    ZKAdversarialNet,
    create_holographic_model,
    SimpleONNXComputation,
    # Per-layer verifiable training
    VerifiableLayer,
    VerifiableLinear,
    VerifiableReLU,
    VerifiableConv2d
)

from .training import ZKTrainer

from .layer_training import (
    LayerTrainer,
    MultiLayerTrainer,
    SimpleVerifiableNet,
    create_demo_verifiable_network
)

from .plotting import ZKPlotter, create_training_plots

from .inference import ZKInference, run_inference_analysis

from .benchmarks import ZKBenchmark

from .utils import (
    create_simple_onnx_graph,
    convert_pytorch_to_onnx,
    validate_onnx_model,
    OnnxHandler
)

from .ops_registry import (
    SupportedOp,
    OpCompilationInfo,
    OpsRegistry,
    ops_registry
)

from .onnx_compiler import (
    ONNXOperationWrapper,
    ONNXVerifier,
    ONNXAdversary,
    ONNXOperationCompiler
)

from .compilation_framework import (
    ONNXCompilationFramework,
    compilation_framework
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
    
    # Per-layer verifiable training
    "VerifiableLayer",
    "VerifiableLinear",
    "VerifiableReLU",
    "VerifiableConv2d",
    
    # Training system
    "ZKTrainer",
    
    # Per-layer training system
    "LayerTrainer",
    "MultiLayerTrainer", 
    "SimpleVerifiableNet",
    "create_demo_verifiable_network",
    
    # Plotting and visualization
    "ZKPlotter",
    "create_training_plots",
    
    # Inference and analysis
    "ZKInference",
    "run_inference_analysis",
    
    # Benchmarking
    "ZKBenchmark",
    
    # Utilities
    "create_simple_onnx_graph",
    "convert_pytorch_to_onnx", 
    "validate_onnx_model",
    "OnnxHandler",
    
    # ONNX Operations Registry
    "SupportedOp",
    "OpCompilationInfo", 
    "OpsRegistry",
    "ops_registry",
    
    # ONNX Operation Compiler
    "ONNXOperationWrapper",
    "ONNXVerifier",
    "ONNXAdversary", 
    "ONNXOperationCompiler",
    
    # Compilation Framework
    "ONNXCompilationFramework",
    "compilation_framework",
    
    # Version info
    "__version__",
    "__author__",
    "__description__"
] 