#!/usr/bin/env python3
"""
Debug Tensor Shape Tracking

Debug script to understand why ONNX tensor shape tracking isn't working.
"""

import onnx
import onnxruntime as ort
from nomopoly.ops_registry import SupportedOp

def debug_onnx_model(model_path):
    """Debug ONNX model structure and tensor shapes."""
    print(f"🔍 Debugging ONNX model: {model_path}")
    print("=" * 60)
    
    # Load ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    
    print(f"📊 Model Overview:")
    print(f"   Nodes: {len(graph.node)}")
    print(f"   Inputs: {len(graph.input)}")
    print(f"   Outputs: {len(graph.output)}")
    
    # Show inputs
    print(f"\n📥 Model Inputs:")
    tensor_shapes = {}
    for i, input_info in enumerate(graph.input):
        print(f"   {i+1}. {input_info.name}")
        if input_info.type.tensor_type.shape.dim:
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(f"dynamic({dim.dim_param})" if dim.dim_param else "?")
            print(f"      Shape: {shape}")
            
            # Store concrete shapes
            concrete_shape = [dim if isinstance(dim, int) else 1 for dim in shape]
            tensor_shapes[input_info.name] = tuple(concrete_shape)
        else:
            print(f"      Shape: No shape info")
    
    # Show all nodes with details
    print(f"\n📋 Model Nodes:")
    for i, node in enumerate(graph.node):
        print(f"   {i+1}. {node.op_type}")
        print(f"      Name: {node.name}")
        print(f"      Inputs: {list(node.input)}")
        print(f"      Outputs: {list(node.output)}")
        
        # Show attributes
        if node.attribute:
            print(f"      Attributes:")
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    print(f"        {attr.name}: {attr.i}")
                elif attr.type == onnx.AttributeProto.INTS:
                    print(f"        {attr.name}: {list(attr.ints)}")
                elif attr.type == onnx.AttributeProto.FLOAT:
                    print(f"        {attr.name}: {attr.f}")
                elif attr.type == onnx.AttributeProto.STRING:
                    print(f"        {attr.name}: {attr.s.decode('utf-8')}")
        
        # Check if supported
        supported = node.op_type in [op.value for op in SupportedOp]
        print(f"      Supported: {'✅' if supported else '❌'}")
        
        # Try to determine input shape
        input_shape = None
        if node.input and node.input[0] in tensor_shapes:
            input_shape = tensor_shapes[node.input[0]]
            print(f"      Input shape: {input_shape}")
        else:
            print(f"      Input shape: ❓ Unknown")
        
        print()
    
    # Try ONNX Runtime
    print(f"🏃 ONNX Runtime Analysis:")
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        print(f"   Runtime inputs:")
        for inp in session.get_inputs():
            print(f"     {inp.name}: {inp.shape} ({inp.type})")
        
        print(f"   Runtime outputs:")
        for out in session.get_outputs():
            print(f"     {out.name}: {out.shape} ({out.type})")
            
    except Exception as e:
        print(f"   ❌ ONNX Runtime error: {e}")

if __name__ == "__main__":
    debug_onnx_model("test_model.onnx") 