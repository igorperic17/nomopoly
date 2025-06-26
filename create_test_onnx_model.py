"""
Create Test ONNX Model

Generate a simple ONNX model with multiple supported operations
for testing the modular compilation framework.
"""

import torch
import torch.nn as nn
import torch.onnx
import os


class SimpleTestNet(nn.Module):
    """Simple network with multiple operation types for testing."""
    
    def __init__(self):
        super(SimpleTestNet, self).__init__()
        
        # Conv2d -> ReLU -> MaxPool -> Flatten -> Linear -> ReLU -> Linear
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Conv
        self.relu1 = nn.ReLU()  # ReLU
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPool
        self.flatten = nn.Flatten()  # Flatten
        self.fc1 = nn.Linear(16 * 4 * 4, 64)  # Linear (GEMM)
        self.relu2 = nn.ReLU()  # ReLU  
        self.fc2 = nn.Linear(64, 10)  # Linear (GEMM)
        
    def forward(self, x):
        x = self.conv1(x)  # Conv operation
        x = self.relu1(x)  # ReLU operation
        x = self.pool1(x)  # MaxPool operation
        x = self.flatten(x)  # Flatten operation
        x = self.fc1(x)  # Linear/GEMM operation
        x = self.relu2(x)  # ReLU operation
        x = self.fc2(x)  # Linear/GEMM operation
        return x


def create_test_onnx_model(output_path: str = "test_model.onnx"):
    """Create and export a test ONNX model."""
    print(f"🏗️  Creating test ONNX model: {output_path}")
    
    # Create model
    model = SimpleTestNet()
    model.eval()
    
    # Create dummy input: batch_size=1, channels=3, height=8, width=8
    dummy_input = torch.randn(1, 3, 8, 8)
    
    # Test the model
    with torch.no_grad():
        output = model(dummy_input)
        print(f"   Model output shape: {output.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Test ONNX model created: {output_path}")
    
    # Print model info
    import onnx
    onnx_model = onnx.load(output_path)
    
    print(f"📊 Model Info:")
    print(f"   Nodes: {len(onnx_model.graph.node)}")
    print(f"   Operations found:")
    
    op_types = set()
    for node in onnx_model.graph.node:
        op_types.add(node.op_type)
        print(f"     - {node.op_type}")
    
    print(f"   Unique operation types: {len(op_types)}")
    
    return output_path


if __name__ == "__main__":
    create_test_onnx_model() 