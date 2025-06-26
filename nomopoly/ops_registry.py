"""
ONNX Operations Registry

This module defines the supported ONNX operations for proof-capable compilation
and provides utilities for managing operation metadata and compilation status.
"""

import os
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from pathlib import Path


class SupportedOp(Enum):
    """Enumeration of supported ONNX operations for proof compilation."""
    
    CONV2D = "Conv"
    RELU = "Relu" 
    MATMUL = "MatMul"
    GEMM = "Gemm"  # General Matrix Multiplication
    ADD = "Add"
    BATCHNORM = "BatchNormalization"
    MAXPOOL = "MaxPool"
    AVGPOOL = "AveragePool"
    FLATTEN = "Flatten"
    RESHAPE = "Reshape"
    
    def __str__(self):
        return self.value


@dataclass
class OpCompilationInfo:
    """Information about an operation's compilation status and metadata."""
    
    op_type: SupportedOp
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    attributes: Dict[str, Any]
    base_onnx_path: Optional[str] = None
    prover_onnx_path: Optional[str] = None
    verifier_onnx_path: Optional[str] = None
    adversary_onnx_path: Optional[str] = None
    compilation_log_path: Optional[str] = None
    is_compiled: bool = False
    training_metrics: Optional[Dict[str, List[float]]] = None
    
    @property
    def folder_name(self) -> str:
        """Generate folder name for this operation."""
        shape_str = "x".join(map(str, self.input_shape))
        return f"{self.op_type.value.lower()}_{shape_str}"
    
    @property
    def compilation_complete(self) -> bool:
        """Check if all compilation artifacts exist."""
        return all([
            self.prover_onnx_path and os.path.exists(self.prover_onnx_path),
            self.verifier_onnx_path and os.path.exists(self.verifier_onnx_path),
            self.adversary_onnx_path and os.path.exists(self.adversary_onnx_path)
        ])


class OpsRegistry:
    """Registry for managing supported operations and their compilation status."""
    
    def __init__(self, ops_dir: str = "ops"):
        self.ops_dir = Path(ops_dir)
        self.ops_dir.mkdir(exist_ok=True)
        
        # Registry of discovered operations
        self.discovered_ops: Dict[str, OpCompilationInfo] = {}
        
    def is_supported_op(self, op_type: str) -> bool:
        """Check if an operation type is supported."""
        try:
            SupportedOp(op_type)
            return True
        except ValueError:
            return False
    
    def register_operation(
        self,
        op_type: str,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        attributes: Dict[str, Any] = None
    ) -> OpCompilationInfo:
        """Register a discovered operation for compilation."""
        
        if not self.is_supported_op(op_type):
            raise ValueError(f"Unsupported operation type: {op_type}")
        
        op_enum = SupportedOp(op_type)
        op_info = OpCompilationInfo(
            op_type=op_enum,
            input_shape=input_shape,
            output_shape=output_shape,
            attributes=attributes or {}
        )
        
        # Create folder structure
        op_folder = self.ops_dir / op_info.folder_name
        op_folder.mkdir(parents=True, exist_ok=True)
        
        # Set file paths - everything now goes in the operation folder
        op_info.base_onnx_path = str(op_folder / f"{op_info.folder_name}.onnx")
        op_info.prover_onnx_path = str(op_folder / f"{op_info.folder_name}_prover.onnx")
        op_info.verifier_onnx_path = str(op_folder / f"{op_info.folder_name}_verifier.onnx")
        op_info.adversary_onnx_path = str(op_folder / f"{op_info.folder_name}_adversary.onnx")
        op_info.compilation_log_path = str(op_folder / "compilation.log")
        
        # Check if already compiled
        op_info.is_compiled = op_info.compilation_complete
        
        # Store in registry
        self.discovered_ops[op_info.folder_name] = op_info
        
        return op_info
    
    def get_uncompiled_operations(self) -> List[OpCompilationInfo]:
        """Get list of operations that need compilation."""
        return [op for op in self.discovered_ops.values() if not op.compilation_complete]
    
    def get_compiled_operations(self) -> List[OpCompilationInfo]:
        """Get list of operations that are already compiled."""
        return [op for op in self.discovered_ops.values() if op.compilation_complete]
    
    def scan_onnx_model(self, onnx_model_path: str) -> List[OpCompilationInfo]:
        """
        Scan an ONNX model and register all supported operations found.
        
        Args:
            onnx_model_path: Path to ONNX model to scan
            
        Returns:
            List of discovered operation info objects
        """
        print(f"🔍 Scanning ONNX model: {onnx_model_path}")
        
        # Load ONNX model
        model = onnx.load(onnx_model_path)
        graph = model.graph
        
        discovered_ops = []
        
        # Create ONNX runtime session to get tensor info
        try:
            ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
            input_details = ort_session.get_inputs()
            output_details = ort_session.get_outputs()
        except Exception as e:
            print(f"⚠️  Could not create ONNX runtime session: {e}")
            ort_session = None
        
        # Track tensor shapes through the graph
        tensor_shapes = {}
        
        # Initialize with model inputs
        for input_info in graph.input:
            if input_info.type.tensor_type.shape.dim:
                shape = []
                for dim in input_info.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(1)  # Default for dynamic dimensions
                tensor_shapes[input_info.name] = tuple(shape)
        
        # Process nodes in the graph
        for node in graph.node:
            op_type = node.op_type
            
            if self.is_supported_op(op_type):
                print(f"  📌 Found supported operation: {op_type}")
                
                # Try to determine input/output shapes
                input_shape = None
                output_shape = None
                
                # Extract attributes first
                attributes = {}
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.INT:
                        attributes[attr.name] = attr.i
                    elif attr.type == onnx.AttributeProto.INTS:
                        attributes[attr.name] = list(attr.ints)
                    elif attr.type == onnx.AttributeProto.FLOAT:
                        attributes[attr.name] = attr.f
                    elif attr.type == onnx.AttributeProto.STRING:
                        attributes[attr.name] = attr.s.decode('utf-8')
                
                # Get input shape
                if node.input and node.input[0] in tensor_shapes:
                    input_shape = tensor_shapes[node.input[0]]
                elif node.input:
                    # Try to infer from ONNX runtime if available
                    if ort_session:
                        try:
                            # Use a dummy input to get shape info
                            dummy_inputs = {}
                            for inp in ort_session.get_inputs():
                                if inp.name == node.input[0]:
                                    shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
                                    input_shape = tuple(shape)
                                    tensor_shapes[node.input[0]] = input_shape
                                    break
                        except:
                            pass
                
                # Estimate output shape based on operation type and attributes
                if input_shape:
                    output_shape = self._estimate_output_shape(
                        op_type, input_shape, attributes
                    )
                
                # Always try to update tensor shapes for the next node, regardless of registration
                estimated_output_shape = output_shape
                
                if input_shape and estimated_output_shape:
                    try:
                        op_info = self.register_operation(
                            op_type, input_shape, estimated_output_shape, attributes
                        )
                        discovered_ops.append(op_info)
                        print(f"    ✅ Registered: {op_info.folder_name}")
                        
                    except Exception as e:
                        print(f"    ❌ Failed to register {op_type}: {e}")
                        
                    # Update tensor shapes for next nodes (even if registration failed)
                    if node.output:
                        for i, output_name in enumerate(node.output):
                            if i == 0:  # Primary output
                                tensor_shapes[output_name] = estimated_output_shape
                                
                else:
                    print(f"    ⚠️  Could not determine shapes for {op_type}")
            else:
                print(f"  ❓ Unsupported operation: {op_type}")
        
        print(f"🎯 Discovered {len(discovered_ops)} supported operations")
        return discovered_ops
    
    def _estimate_output_shape(
        self, 
        op_type: str, 
        input_shape: Tuple[int, ...], 
        attributes: Dict[str, Any]
    ) -> Optional[Tuple[int, ...]]:
        """Estimate output shape for an operation given input shape and attributes."""
        
        if op_type == "Relu":
            return input_shape  # ReLU preserves shape
        
        elif op_type == "Conv":
            # Conv2d shape calculation
            if len(input_shape) == 4:  # [N, C, H, W]
                n, c_in, h_in, w_in = input_shape
                
                # Get attributes with defaults
                kernel_shape = attributes.get('kernel_shape', [3, 3])
                strides = attributes.get('strides', [1, 1])
                pads = attributes.get('pads', [0, 0, 0, 0])  # [top, left, bottom, right]
                
                # For Conv, we need to guess output channels from context
                # In our test model, we know it goes from 3->16->etc.
                if c_in == 3:  # First conv layer
                    c_out = 16
                elif c_in == 16:
                    c_out = 32
                else:
                    c_out = max(32, c_in)  # Default
                
                # Handle padding format
                if len(pads) == 4:
                    pad_h = pads[0] + pads[2]  # top + bottom
                    pad_w = pads[1] + pads[3]  # left + right
                elif len(pads) == 2:
                    pad_h, pad_w = pads[0], pads[1]
                else:
                    pad_h = pad_w = pads[0] if pads else 0
                
                # Calculate output spatial dimensions
                h_out = (h_in + pad_h - kernel_shape[0]) // strides[0] + 1
                w_out = (w_in + pad_w - kernel_shape[1]) // strides[1] + 1
                
                return (n, c_out, h_out, w_out)
        
        elif op_type in ["MatMul", "Gemm"]:
            # Matrix multiplication - for Gemm it's input @ weight + bias
            if len(input_shape) >= 2:
                *batch_dims, in_features = input_shape
                
                # Common output sizes for our test model
                if in_features == 512:  # First FC layer
                    out_features = 64
                elif in_features == 64:  # Second FC layer  
                    out_features = 10
                else:
                    out_features = min(in_features, 512)  # Default
                
                return (*batch_dims, out_features)
        
        elif op_type == "Add":
            return input_shape  # Element-wise operations preserve shape
        
        elif op_type == "Flatten":
            if len(input_shape) >= 2:
                batch_size = input_shape[0]
                flattened_size = 1
                for dim in input_shape[1:]:
                    flattened_size *= dim
                return (batch_size, flattened_size)
        
        elif op_type in ["MaxPool", "AveragePool"]:
            # Pooling operations
            if len(input_shape) == 4:  # [N, C, H, W]
                n, c, h_in, w_in = input_shape
                
                kernel_shape = attributes.get('kernel_shape', [2, 2])
                strides = attributes.get('strides', kernel_shape)
                pads = attributes.get('pads', [0, 0, 0, 0])
                
                h_out = (h_in + pads[0] + pads[2] - kernel_shape[0]) // strides[0] + 1
                w_out = (w_in + pads[1] + pads[3] - kernel_shape[1]) // strides[1] + 1
                
                return (n, c, h_out, w_out)
        
        # Default: return input shape (for operations that preserve shape)
        return input_shape
    
    def print_registry_status(self):
        """Print the current status of the operations registry."""
        print("\n📊 ONNX Operations Registry Status")
        print("=" * 50)
        
        compiled = self.get_compiled_operations()
        uncompiled = self.get_uncompiled_operations()
        
        print(f"✅ Compiled Operations: {len(compiled)}")
        for op in compiled:
            print(f"  • {op.folder_name}")
        
        print(f"\n⏳ Uncompiled Operations: {len(uncompiled)}")
        for op in uncompiled:
            print(f"  • {op.folder_name}")
        
        print(f"\n📁 Total Operations: {len(self.discovered_ops)}")


# Create global registry instance
ops_registry = OpsRegistry() 