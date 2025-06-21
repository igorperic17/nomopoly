"""
nomopoly: No more polynomial commitments!

A neural network compiler for converting ONNX compute graphs into 
Zero Knowledge ML (ZKML) circuits using adversarial training.
"""

from .networks import ZKProverNet, ZKVerifierNet, ZKAdversarialNet
from .training import AutoZKTraining
from .utils import create_simple_onnx_graph, OnnxHandler
from .benchmarks import ZKMLBenchmark

__version__ = "0.1.0"
__author__ = "Igor"

__all__ = [
    "ZKProverNet",
    "ZKVerifierNet", 
    "ZKAdversarialNet",
    "AutoZKTraining",
    "create_simple_onnx_graph",
    "OnnxHandler",
    "ZKMLBenchmark",
] 