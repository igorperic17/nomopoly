"""
Utility functions for ONNX graph generation and handling.

This module provides functions to create simple ONNX graphs for testing
and utilities for handling ONNX models within the ZK framework.
"""

import torch
import torch.nn as nn
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import os


def create_simple_onnx_graph(
    save_path: str = "mnist_classifier.onnx",
    input_shape: Tuple[int, ...] = (196,)  # 14x14 flattened
) -> str:
    """
    Create a simple ONNX graph that performs MNIST digit classification.
    
    Takes a flattened 14x14 grayscale image and outputs 10 class probabilities.
    
    Args:
        save_path: Path to save the ONNX model
        input_shape: Shape of the input tensor (flattened image)
        
    Returns:
        Path to the saved ONNX model
    """
    
    # Define input (flattened 14x14 image)
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, input_shape
    )
    
    # Define output (10 class probabilities)
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, (10,)
    )
    
    # Create weight matrices for a simple 2-layer network
    # Hidden layer: 196 -> 64
    W1_data = np.random.randn(input_shape[0], 64).astype(np.float32) * 0.1
    W1_tensor = numpy_helper.from_array(W1_data, name='W1')
    
    b1_data = np.zeros(64, dtype=np.float32)
    b1_tensor = numpy_helper.from_array(b1_data, name='b1')
    
    # Output layer: 64 -> 10
    W2_data = np.random.randn(64, 10).astype(np.float32) * 0.1
    W2_tensor = numpy_helper.from_array(W2_data, name='W2')
    
    b2_data = np.zeros(10, dtype=np.float32)
    b2_tensor = numpy_helper.from_array(b2_data, name='b2')
    
    # Create nodes
    # First layer: input @ W1 + b1
    matmul1_node = helper.make_node(
        'MatMul',
        inputs=['input', 'W1'],
        outputs=['hidden_raw']
    )
    
    add1_node = helper.make_node(
        'Add',
        inputs=['hidden_raw', 'b1'],
        outputs=['hidden_with_bias']
    )
    
    # ReLU activation
    relu_node = helper.make_node(
        'Relu',
        inputs=['hidden_with_bias'],
        outputs=['hidden_activated']
    )
    
    # Second layer: hidden @ W2 + b2
    matmul2_node = helper.make_node(
        'MatMul',
        inputs=['hidden_activated', 'W2'],
        outputs=['logits_raw']
    )
    
    add2_node = helper.make_node(
        'Add',
        inputs=['logits_raw', 'b2'],
        outputs=['logits']
    )
    
    # Softmax for probabilities
    softmax_node = helper.make_node(
        'Softmax',
        inputs=['logits'],
        outputs=['output']
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[matmul1_node, add1_node, relu_node, matmul2_node, add2_node, softmax_node],
        name='MNISTClassifier',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[W1_tensor, b1_tensor, W2_tensor, b2_tensor]
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 10
    
    # Check the model
    onnx.checker.check_model(model)
    
    # Save the model
    onnx.save(model, save_path)
    
    print(f"MNIST classifier ONNX model saved to {save_path}")
    return save_path


def create_complex_onnx_graph(
    save_path: str = "complex_computation.onnx",
    input_shape: Tuple[int, ...] = (4,)
) -> str:
    """
    Create a more complex ONNX graph for future testing.
    
    This creates a graph with multiple operations: MatMul, Add, ReLU.
    
    Args:
        save_path: Path to save the ONNX model
        input_shape: Shape of the input tensor
        
    Returns:
        Path to the saved ONNX model
    """
    
    # Define input
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, input_shape
    )
    
    # Define weight matrix (4x2)
    weight_data = np.random.randn(input_shape[0], 2).astype(np.float32)
    weight_tensor = numpy_helper.from_array(weight_data, name='weight')
    
    # Define bias vector (2,)
    bias_data = np.random.randn(2).astype(np.float32)
    bias_tensor = numpy_helper.from_array(bias_data, name='bias')
    
    # Define output
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, (2,)
    )
    
    # Create nodes
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['input', 'weight'],
        outputs=['matmul_output']
    )
    
    add_node = helper.make_node(
        'Add',
        inputs=['matmul_output', 'bias'],
        outputs=['add_output']
    )
    
    relu_node = helper.make_node(
        'Relu',
        inputs=['add_output'],
        outputs=['output']
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[matmul_node, add_node, relu_node],
        name='ComplexComputation',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_tensor, bias_tensor]
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 10
    
    # Check the model
    onnx.checker.check_model(model)
    
    # Save the model
    onnx.save(model, save_path)
    
    print(f"Complex computation ONNX model saved to {save_path}")
    return save_path


class OnnxHandler:
    """
    Handler class for loading and running ONNX models.
    
    This class provides utilities for working with ONNX models in the context
    of zero knowledge machine learning.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ONNX handler.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_specs = None
        self.output_specs = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load an ONNX model from file.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        
        # Get input specifications
        self.input_specs = {
            input_meta.name: {
                'shape': input_meta.shape,
                'type': input_meta.type
            }
            for input_meta in self.session.get_inputs()
        }
        
        # Get output specifications
        self.output_specs = {
            output_meta.name: {
                'shape': output_meta.shape,
                'type': output_meta.type
            }
            for output_meta in self.session.get_outputs()
        }
        
        print(f"Loaded ONNX model from {model_path}")
        print(f"Input specs: {self.input_specs}")
        print(f"Output specs: {self.output_specs}")
    
    def run_inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on the loaded ONNX model.
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        if self.session is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Map outputs to names
        output_names = [output_meta.name for output_meta in self.session.get_outputs()]
        return {name: output for name, output in zip(output_names, outputs)}
    
    def run_inference_torch(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run inference with PyTorch tensors.
        
        Args:
            inputs: Dictionary mapping input names to PyTorch tensors
            
        Returns:
            Dictionary mapping output names to PyTorch tensors
        """
        # Convert to numpy
        numpy_inputs = {
            name: tensor.detach().cpu().numpy() 
            for name, tensor in inputs.items()
        }
        
        # Run inference
        numpy_outputs = self.run_inference(numpy_inputs)
        
        # Convert back to PyTorch
        device = next(iter(inputs.values())).device
        torch_outputs = {
            name: torch.from_numpy(output).to(device)
            for name, output in numpy_outputs.items()
        }
        
        return torch_outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.session is None:
            return {"error": "No model loaded"}
        
        return {
            "model_path": self.model_path,
            "input_specs": self.input_specs,
            "output_specs": self.output_specs,
            "providers": self.session.get_providers()
        }


def generate_test_data(
    input_shape: Tuple[int, ...],
    num_samples: int = 100,
    value_range: Tuple[float, float] = (0.0, 1.0)
) -> torch.Tensor:
    """
    Generate test data for ONNX model testing.
    
    For MNIST-like data, generates normalized pixel values between 0 and 1.
    
    Args:
        input_shape: Shape of each input sample (e.g., (196,) for 14x14 flattened)
        num_samples: Number of samples to generate
        value_range: Range of values to generate (0-1 for normalized images)
        
    Returns:
        Tensor of test data with shape (num_samples, *input_shape)
    """
    full_shape = (num_samples,) + input_shape
    data = torch.FloatTensor(*full_shape).uniform_(*value_range)
    return data


def benchmark_onnx_model(
    model_path: str,
    test_data: torch.Tensor,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark an ONNX model's performance.
    
    Args:
        model_path: Path to the ONNX model
        test_data: Test data tensor
        num_runs: Number of runs for benchmarking
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    handler = OnnxHandler(model_path)
    
    # Get input name
    input_name = list(handler.input_specs.keys())[0]
    
    # Warm up
    for _ in range(10):
        inputs = {input_name: test_data[0:1].numpy()}
        handler.run_inference(inputs)
    
    # Benchmark
    times = []
    for i in range(num_runs):
        start_time = time.time()
        inputs = {input_name: test_data[i:i+1].numpy()}
        handler.run_inference(inputs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "total_time": np.sum(times)
    }


def convert_pytorch_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    save_path: str,
    input_names: List[str] = None,
    output_names: List[str] = None,
    dynamic_axes: Dict[str, Dict[int, str]] = None
) -> str:
    """
    Convert a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to convert
        dummy_input: Example input for tracing
        save_path: Path to save the ONNX model
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axes specification
        
    Returns:
        Path to the saved ONNX model
    """
    model.eval()
    
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=10,
        do_constant_folding=True
    )
    
    print(f"PyTorch model converted to ONNX and saved at {save_path}")
    return save_path


def validate_onnx_model(model_path: str) -> bool:
    """
    Validate an ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        
    Returns:
        True if model is valid, False otherwise
    """
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"ONNX model {model_path} is valid")
        return True
    except Exception as e:
        print(f"ONNX model {model_path} is invalid: {e}")
        return False 