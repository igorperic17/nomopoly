"""
NoMoPoly: Zero Knowledge Machine Learning Framework
A framework for building zero-knowledge proofs over machine learning computations.
"""

from .networks import ZKProverNet, ZKVerifierNet, ZKAdversarialNet
from .training import GeneralZKTraining, train_pretrained_classifier, PretrainedMNISTClassifier
from .benchmarks import ZKBenchmark
from .utils import create_simple_onnx_graph, OnnxHandler

__version__ = "0.1.0"
__author__ = "NoMoPoly Team"

__all__ = [
    "ZKProverNet",
    "ZKVerifierNet", 
    "ZKAdversarialNet",
    "GeneralZKTraining",
    "train_pretrained_classifier",
    "PretrainedMNISTClassifier",
    "ZKBenchmark",
    "create_simple_onnx_graph",
    "OnnxHandler"
] 