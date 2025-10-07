"""
Nomopoly: ONNX Operation Compilation Framework

A framework for compiling ONNX operations into proof-capable components with
adversarial training for zero-knowledge machine learning.
"""

from .utils import (
    convert_pytorch_to_onnx,
    validate_onnx_model
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

from .network_compiler import (
    network_compiler,
    NetworkCompiler
)

from .zk_op_compiler import (
    ZKOp,
    ZKOpCompiler,
    zk_op_compiler
)

from .zk_graph_compiler import (
    ZKGraph,
    ZKGraphCompiler
)

from .nas_compilation_framework import NASCompilationFramework
from .neural_architecture_search import NeuralArchitectureSearch
from .huggingface_loader import HuggingFaceModelLoader
from .pytorch_graph_compiler import (
    PyTorchOp,
    PyTorchZKOp,
    PyTorchZKGraph,
    PyTorchGraphExtractor,
    PyTorchZKCompiler,
    ZKProver
)

__version__ = "0.4.0"
__author__ = "Nomopoly Team"
__description__ = "ONNX Operation Compilation Framework for Zero Knowledge Machine Learning"

__all__ = [
    # Utilities
    "convert_pytorch_to_onnx", 
    "validate_onnx_model",
    
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
    
    # Network Compiler
    "network_compiler",
    "NetworkCompiler",
    
    # ZK Operation Compiler
    "ZKOp",
    "ZKOpCompiler", 
    "zk_op_compiler",
    
    # ZK Graph Compiler
    "ZKGraph",
    "ZKGraphCompiler",
    
    # NAS Compilation Framework
    "NASCompilationFramework",
    
    # Neural Architecture Search
    "NeuralArchitectureSearch",
    
    # HuggingFace Model Loader
    "HuggingFaceModelLoader",
    
    # PyTorch Graph Compiler
    "PyTorchOp",
    "PyTorchZKOp",
    "PyTorchZKGraph",
    "PyTorchGraphExtractor",
    "PyTorchZKCompiler",
    "ZKProver",
    
    # Version info
    "__version__",
    "__author__",
    "__description__"
] 