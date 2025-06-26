"""
Core neural network architectures for Zero Knowledge ML using Holographic Reduced Representations.

This module implements the HRR-based approach for fixed-size proof generation
regardless of network complexity, inspired by GAN's breakthrough methodology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import onnx
import onnxruntime as ort


class HolographicMemory(nn.Module):
    """
    Holographic Reduced Representation (HRR) memory system for neural network proofs.
    Creates fixed-size embeddings regardless of network complexity using circular convolution.
    
    Key innovations:
    - Fixed memory size regardless of network complexity
    - Circular convolution for binding activations with positions
    - Superposition memory with exponential decay
    - Deterministic position encoding
    """
    
    def __init__(self, memory_size=512):
        super(HolographicMemory, self).__init__()
        self.memory_size = memory_size
        self.memory = None
        self.position_vectors = {}  # Cache for position encoding vectors
        
    def _circular_convolution(self, a, b):
        """
        ONNX-compatible circular convolution using efficient element-wise operations.
        This implements the core HRR binding operation: bind(a,b) = circular_conv(a,b)
        """
        batch_size, dim = a.shape
        result = torch.zeros_like(a)
        
        # Efficient circular convolution
        for i in range(dim):
            shifted_b = torch.roll(b, shifts=i, dims=-1)
            result[:, i] = torch.sum(a * shifted_b, dim=-1)
        
        return result
    
    def _generate_position_vector(self, position, device):
        """Generate a unique, deterministic position encoding vector for each layer."""
        if position not in self.position_vectors:
            # Create deterministic but unique vector for this position
            original_state = torch.get_rng_state()
            torch.manual_seed(position + 42)  # Deterministic seed
            vector = torch.randn(self.memory_size, device=device)
            vector = F.normalize(vector, p=2, dim=0)  # Unit vector
            self.position_vectors[position] = vector
            torch.set_rng_state(original_state)
        return self.position_vectors[position].to(device)
    
    def _compress_activation(self, activation, device):
        """Compress any activation tensor to fixed memory size."""
        # Flatten the activation
        flat_activation = activation.view(activation.shape[0], -1)  # [batch, features]
        
        # Compress to memory size using adaptive pooling
        if flat_activation.shape[1] > self.memory_size:
            # Use adaptive pooling for compression
            reshaped = flat_activation.unsqueeze(1)  # [batch, 1, features]
            compressed = F.adaptive_avg_pool1d(reshaped, self.memory_size)
            compressed = compressed.squeeze(1)  # [batch, memory_size]
        elif flat_activation.shape[1] < self.memory_size:
            # Pad with zeros if too small
            padding_size = self.memory_size - flat_activation.shape[1]
            compressed = F.pad(flat_activation, (0, padding_size), 'constant', 0)
        else:
            compressed = flat_activation
            
        # Normalize to unit vectors for stable binding
        compressed = F.normalize(compressed, p=2, dim=1)
        return compressed
    
    def bind_activation(self, activation, layer_position, device):
        """
        Bind an activation into the holographic memory using circular convolution.
        
        Args:
            activation: Tensor from any layer (gradients preserved!)
            layer_position: Integer position of the layer
            device: Device to operate on
        """
        # Compress activation to fixed size
        compressed = self._compress_activation(activation, device)
        
        # Get position encoding vector
        position_vec = self._generate_position_vector(layer_position, device).detach()
        
        # Bind activation with position using circular convolution
        bound_activation = self._circular_convolution(
            compressed, 
            position_vec.unsqueeze(0).expand(compressed.shape[0], -1)
        )
        
        # Superposition: accumulate into memory with exponential decay
        if self.memory is None:
            self.memory = bound_activation
        else:
            # Exponential decay prevents saturation (key HRR property)
            decay_factor = 0.95
            self.memory = decay_factor * self.memory.clone() + bound_activation
            
        # Normalize to prevent unbounded growth
        self.memory = F.normalize(self.memory, p=2, dim=1)
        
        return self.memory
    
    def reset_memory(self):
        """Reset memory for new computation."""
        self.memory = None
    
    def get_memory_trace(self):
        """Get the current holographic memory trace (preserving gradients)."""
        if self.memory is None:
            return torch.zeros(1, self.memory_size, requires_grad=True)
        return self.memory


class OriginalMNISTNet(nn.Module):
    """Original MNIST classification network that we'll augment with HRR."""
    
    def __init__(self, input_dim: int = 196, output_dim: int = 10):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x1 = F.relu(self.layer1(x))
        x1 = self.dropout1(x1)
        x2 = F.relu(self.layer2(x1))
        x2 = self.dropout2(x2)
        output = F.log_softmax(self.layer3(x2), dim=-1)
        return output


class HolographicWrapper(nn.Module):
    """
    Wrapper that integrates holographic memory directly into any PyTorch model.
    
    This is the core innovation: fixed-size proofs regardless of network complexity!
    """
    
    def __init__(self, base_model, proof_size=64, memory_size=512):
        super(HolographicWrapper, self).__init__()
        self.base_model = base_model
        self.memory_size = memory_size
        
        # Proof generator: memory_size -> proof_size
        self.proof_generator = nn.Sequential(
            nn.Linear(memory_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, proof_size),
            nn.Tanh()  # Normalize proof values
        )
        
    def _bind_intermediate_activations(self, x, holographic_memory):
        """
        Explicitly bind intermediate activations during forward pass.
        This replaces hook-based approaches with direct integration.
        """
        device = x.device
        layer_position = 0
        
        # Handle MNIST network architecture
        if hasattr(self.base_model, 'layer1') and hasattr(self.base_model, 'layer2'):
            # MNIST architecture
            x1 = F.relu(self.base_model.layer1(x))
            holographic_memory.bind_activation(x1, layer_position, device)
            layer_position += 1
            
            x1_dropped = self.base_model.dropout1(x1)
            holographic_memory.bind_activation(x1_dropped, layer_position, device)
            layer_position += 1
            
            x2 = F.relu(self.base_model.layer2(x1_dropped))
            holographic_memory.bind_activation(x2, layer_position, device)
            layer_position += 1
            
            x2_dropped = self.base_model.dropout2(x2)
            holographic_memory.bind_activation(x2_dropped, layer_position, device)
            layer_position += 1
            
            output = F.log_softmax(self.base_model.layer3(x2_dropped), dim=-1)
            holographic_memory.bind_activation(output, layer_position, device)
            
        else:
            # Fallback: run model and bind final output
            output = self.base_model(x)
            holographic_memory.bind_activation(output, 0, device)
            
        return output, holographic_memory
    
    def forward(self, x):
        """
        Forward pass with integrated holographic memory.
        Returns (output, proof) where proof is FIXED-SIZE regardless of network size.
        """
        device = x.device
        holographic_memory = HolographicMemory(memory_size=self.memory_size).to(device)
        holographic_memory.reset_memory()
        
        # Perform forward pass with memory binding
        output, bound_memory = self._bind_intermediate_activations(x, holographic_memory)
        
        # Generate fixed-size proof from holographic memory
        memory_trace = bound_memory.get_memory_trace()
        proof = self.proof_generator(memory_trace)
        
        return output, proof
    
    def export_original_vs_augmented(self, save_dir: str = "exported_models"):
        """Export both original and augmented models to ONNX for comparison."""
        import os
        from .utils import convert_pytorch_to_onnx, validate_onnx_model
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Export original model  
        print("Exporting original model...")
        dummy_input = torch.randn(1, 196)  # MNIST 14x14 flattened
        
        # Extract the base model for export
        original_path = os.path.join(save_dir, "original_mnist.onnx")
        convert_pytorch_to_onnx(
            model=self.base_model, 
            dummy_input=dummy_input,
            output_path=original_path,
            input_names=['input'],
            output_names=['output']
        )
        
        # Validate original
        if validate_onnx_model(original_path):
            print(f"✅ Original model exported and validated: {original_path}")
        else:
            print(f"❌ Original model validation failed: {original_path}")
        
        # Export augmented model (with proofs)
        print("Exporting augmented model with proofs...")
        augmented_path = os.path.join(save_dir, "augmented_mnist_with_proofs.onnx")
        convert_pytorch_to_onnx(
            model=self,
            dummy_input=dummy_input,
            output_path=augmented_path,
            input_names=['input'],
            output_names=['output', 'proof']
        )
        
        # Validate augmented
        if validate_onnx_model(augmented_path):
            print(f"✅ Augmented model exported and validated: {augmented_path}")
        else:
            print(f"❌ Augmented model validation failed: {augmented_path}")
        
        return {
            "original": original_path,
            "augmented": augmented_path
        }


class ZKProverNet(nn.Module):
    """
    Zero Knowledge Prover Network - generates authentic proofs for computation verification.
    This is the FROZEN inference network that never gets trained.
    """
    
    def __init__(self, input_dim: int, output_dim: int, proof_dim: int = 64):
        super(ZKProverNet, self).__init__()
        
        # Store dimensions for reference
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proof_dim = proof_dim
        
        # Create the HRR-augmented network
        original_model = OriginalMNISTNet(input_dim, output_dim)
        self.holographic_model = HolographicWrapper(original_model, proof_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate authentic computation result and proof."""
        return self.holographic_model(x)
    
    def get_original_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get output from the original model without proof generation."""
        return self.holographic_model.base_model(x)
    
    def export_networks(self, save_dir: str = "exported_models"):
        """Export prover networks to ONNX."""
        return self.holographic_model.export_original_vs_augmented(save_dir)


class ZKVerifierNet(nn.Module):
    """
    Zero Knowledge Verifier Network - binary classifier for (input, output, proof) triplets.
    Learns to distinguish authentic proofs from fake ones.
    """
    
    def __init__(self, input_dim: int, output_dim: int, proof_dim: int = 64):
        super(ZKVerifierNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proof_dim = proof_dim
        self.layers = None  # Will be built dynamically
        
    def _build_layers(self, input_dim, output_dim, proof_dim, device):
        """Build verifier layers dynamically on first forward pass."""
        total_input_dim = input_dim + output_dim + proof_dim
        
        self.layers = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Binary classification: real (1) vs fake (0)
        ).to(device)
        
    def forward(self, input_data: torch.Tensor, output_data: torch.Tensor, proof: torch.Tensor) -> torch.Tensor:
        """
        Verify if the (input, output, proof) triplet is authentic.
        
        Args:
            input_data: Original input to the computation
            output_data: Claimed output from the computation  
            proof: Proof that the computation was performed correctly
            
        Returns:
            Binary classification score: 1 = authentic, 0 = fake
        """
        device = input_data.device
        
        # Build layers on first forward pass
        if self.layers is None:
            self._build_layers(self.input_dim, self.output_dim, self.proof_dim, device)
        
        # Flatten all inputs if needed
        input_flat = input_data.view(input_data.shape[0], -1)
        output_flat = output_data.view(output_data.shape[0], -1) 
        proof_flat = proof.view(proof.shape[0], -1)
        
        # Concatenate triplet
        triplet = torch.cat([input_flat, output_flat, proof_flat], dim=1)
        
        # Binary classification
        verification_score = self.layers(triplet)
        
        return verification_score


class ZKAdversarialNet(nn.Module):
    """
    Zero Knowledge Adversarial Network - generates fake proofs to fool the verifier.
    
    This is the "generator" in our GAN-like setup that tries to produce convincing 
    fake proofs given just the input.
    """
    
    def __init__(self, input_dim: int, output_dim: int, proof_dim: int = 64):
        super(ZKAdversarialNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proof_dim = proof_dim
        
        # Build the fake output generator
        self.output_generator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            nn.LogSoftmax(dim=-1)  # Match original network output format
        )
        
        # Build the fake proof generator  
        self.proof_generator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, proof_dim),
            nn.Tanh()  # Normalize fake proof values
        )
        
        # Combined generator (sometimes generates both together for consistency)
        combined_output_dim = output_dim + proof_dim
        self.combined_generator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, combined_output_dim)
        )
        
    def forward(self, x: torch.Tensor, mode: str = "mixed") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate fake output and fake proof to fool the verifier.
        
        Args:
            x: Input tensor
            mode: Generation mode
                - "separate": Use separate generators for output and proof
                - "combined": Use combined generator for consistency
                - "mixed": Randomly choose between separate and combined
                - "corrupted": Generate corrupted version of real output
                
        Returns:
            Tuple of (fake_output, fake_proof)
        """
        if mode == "separate":
            fake_output = self.output_generator(x)
            fake_proof = self.proof_generator(x)
            
        elif mode == "combined":
            combined = self.combined_generator(x)
            fake_output = combined[:, :self.output_dim]
            fake_proof = combined[:, self.output_dim:]
            
            # Apply appropriate activations
            fake_output = F.log_softmax(fake_output, dim=-1)
            fake_proof = torch.tanh(fake_proof)
            
        elif mode == "mixed":
            # Randomly choose generation strategy
            if torch.rand(1).item() < 0.5:
                fake_output = self.output_generator(x)
                fake_proof = self.proof_generator(x)
            else:
                combined = self.combined_generator(x)
                fake_output = combined[:, :self.output_dim]
                fake_proof = combined[:, self.output_dim:]
                fake_output = F.log_softmax(fake_output, dim=-1)
                fake_proof = torch.tanh(fake_proof)
            
        elif mode == "corrupted":
            # Generate slightly corrupted outputs (harder to detect)
            fake_output = self.generate_corrupted_output(x)
            fake_proof = self.proof_generator(x)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return fake_output, fake_proof
    
    def generate_corrupted_output(self, x: torch.Tensor) -> torch.Tensor:
        """Generate subtly corrupted output that's harder to detect."""
        base_output = self.output_generator(x)
        noise = torch.randn_like(base_output) * 0.1  # Small corruption
        return base_output + noise
    
    def generate_random_output(self, x: torch.Tensor) -> torch.Tensor:
        """Generate completely random output (easy to detect)."""
        batch_size = x.shape[0]
        return torch.randn(batch_size, self.output_dim, device=x.device)


def create_holographic_model(base_model, proof_size=64, memory_size=512):
    """
    Factory function to create HRR-augmented models.
    
    Args:
        base_model: Original PyTorch model to augment
        proof_size: Size of the generated proof vector
        memory_size: Size of the holographic memory
        
    Returns:
        HolographicWrapper instance
    """
    return HolographicWrapper(base_model, proof_size, memory_size)


class SimpleONNXComputation(nn.Module):
    """Simple computation that can be easily exported to ONNX for testing."""
    
    def __init__(self, onnx_model_path: Optional[str] = None):
        super(SimpleONNXComputation, self).__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder computation
        return self.linear(x)


# ===== PER-LAYER VERIFIABLE TRAINING SYSTEM =====

class VerifiableLayer(nn.Module):
    """
    Base class for verifiable layers that can be trained with adversarial proof generation.
    
    Each verifiable layer:
    1. Performs a specific computation (linear, relu, conv2d, etc.)
    2. Has a verifier that takes (input, output, proof) and outputs binary classification
    3. Has a generator that takes input and produces fake output + proof
    """
    
    def __init__(self, proof_dim: int = 32):
        super(VerifiableLayer, self).__init__()
        self.proof_dim = proof_dim
        self.verifier = None  # Will be built dynamically
        self.generator = None  # Will be built dynamically
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the layer computation."""
        raise NotImplementedError("Subclasses must implement forward")
        
    def generate_proof(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Generate a proof for the computation (input, output)."""
        raise NotImplementedError("Subclasses must implement generate_proof")
        
    def verify_triplet(self, x: torch.Tensor, output: torch.Tensor, proof: torch.Tensor) -> torch.Tensor:
        """Verify if (input, output, proof) triplet is authentic."""
        raise NotImplementedError("Subclasses must implement verify_triplet")
        
    def generate_fake(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate fake output and proof given input."""
        raise NotImplementedError("Subclasses must implement generate_fake")


class VerifiableLinear(VerifiableLayer):
    """Verifiable linear layer with adversarial training."""
    
    def __init__(self, in_features: int, out_features: int, proof_dim: int = 32):
        super(VerifiableLinear, self).__init__(proof_dim)
        self.in_features = in_features
        self.out_features = out_features
        
        # The actual linear computation
        self.linear = nn.Linear(in_features, out_features)
        
        # Verifier: takes (input, output, proof) -> binary classification
        self.verifier = nn.Sequential(
            nn.Linear(in_features + out_features + proof_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Generator: takes input -> fake output + proof
        self.generator = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_features + proof_dim),
        )
        
        # Proof generator: takes (input, output) -> proof
        self.proof_gen = nn.Sequential(
            nn.Linear(in_features + out_features, 64),
            nn.ReLU(),
            nn.Linear(64, proof_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform linear computation."""
        return self.linear(x)
        
    def generate_proof(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Generate authentic proof for the linear computation."""
        combined = torch.cat([x, output], dim=-1)
        return self.proof_gen(combined)
        
    def verify_triplet(self, x: torch.Tensor, output: torch.Tensor, proof: torch.Tensor) -> torch.Tensor:
        """Verify (input, output, proof) triplet."""
        triplet = torch.cat([x, output, proof], dim=-1)
        return self.verifier(triplet)
        
    def generate_fake(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate fake output and proof."""
        fake_combined = self.generator(x)
        fake_output = fake_combined[..., :self.out_features]
        fake_proof = torch.tanh(fake_combined[..., self.out_features:])
        return fake_output, fake_proof


class VerifiableReLU(VerifiableLayer):
    """Verifiable ReLU layer with adversarial training."""
    
    def __init__(self, num_features: int, proof_dim: int = 32):
        super(VerifiableReLU, self).__init__(proof_dim)
        self.num_features = num_features
        
        # Verifier: takes (input, output, proof) -> binary classification
        self.verifier = nn.Sequential(
            nn.Linear(num_features * 2 + proof_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Generator: takes input -> fake output + proof
        self.generator = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_features + proof_dim),
        )
        
        # Proof generator: takes (input, output) -> proof
        self.proof_gen = nn.Sequential(
            nn.Linear(num_features * 2, 64),
            nn.ReLU(),
            nn.Linear(64, proof_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform ReLU computation."""
        return F.relu(x)
        
    def generate_proof(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Generate authentic proof for ReLU computation."""
        combined = torch.cat([x, output], dim=-1)
        return self.proof_gen(combined)
        
    def verify_triplet(self, x: torch.Tensor, output: torch.Tensor, proof: torch.Tensor) -> torch.Tensor:
        """Verify (input, output, proof) triplet."""
        triplet = torch.cat([x, output, proof], dim=-1)
        return self.verifier(triplet)
        
    def generate_fake(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate fake output and proof."""
        fake_combined = self.generator(x)
        fake_output = fake_combined[..., :self.num_features]
        fake_proof = torch.tanh(fake_combined[..., self.num_features:])
        return fake_output, fake_proof


class VerifiableConv2d(VerifiableLayer):
    """Verifiable Conv2d layer with adversarial training."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, proof_dim: int = 32):
        super(VerifiableConv2d, self).__init__(proof_dim)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # The actual conv2d computation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Initialize as None - will be built dynamically based on input/output shapes
        self.verifier = None
        self.generator = None
        self.proof_gen = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform Conv2d computation."""
        return self.conv2d(x)
        
    def _build_verifier_generator(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], device: str):
        """Build verifier and generator networks based on actual tensor shapes."""
        input_size = torch.prod(torch.tensor(input_shape)).item()
        output_size = torch.prod(torch.tensor(output_shape)).item()
        
        # Verifier: takes (input, output, proof) -> binary classification
        self.verifier = nn.Sequential(
            nn.Linear(input_size + output_size + self.proof_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Generator: takes input -> fake output + proof
        self.generator = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_size + self.proof_dim),
        ).to(device)
        
        # Proof generator: takes (input, output) -> proof
        self.proof_gen = nn.Sequential(
            nn.Linear(input_size + output_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.proof_dim),
            nn.Tanh()
        ).to(device)
        
    def generate_proof(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Generate authentic proof for conv2d computation."""
        if self.proof_gen is None:
            self._build_verifier_generator(x.shape[1:], output.shape[1:], x.device)
            
        x_flat = x.view(x.shape[0], -1)
        output_flat = output.view(output.shape[0], -1)
        combined = torch.cat([x_flat, output_flat], dim=-1)
        return self.proof_gen(combined)
        
    def verify_triplet(self, x: torch.Tensor, output: torch.Tensor, proof: torch.Tensor) -> torch.Tensor:
        """Verify (input, output, proof) triplet."""
        if self.verifier is None:
            self._build_verifier_generator(x.shape[1:], output.shape[1:], x.device)
            
        x_flat = x.view(x.shape[0], -1)
        output_flat = output.view(output.shape[0], -1)
        triplet = torch.cat([x_flat, output_flat, proof], dim=-1)
        return self.verifier(triplet)
        
    def generate_fake(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate fake output and proof."""
        if self.generator is None:
            # Need to do a forward pass to determine output shape
            with torch.no_grad():
                temp_output = self.forward(x)
            self._build_verifier_generator(x.shape[1:], temp_output.shape[1:], x.device)
            
        x_flat = x.view(x.shape[0], -1)
        fake_combined = self.generator(x_flat)
        
        # Calculate output size from conv2d operation
        with torch.no_grad():
            temp_output = self.forward(x[:1])  # Use single sample to get shape
            output_size = torch.prod(torch.tensor(temp_output.shape[1:])).item()
            
        fake_output_flat = fake_combined[..., :output_size]
        fake_proof = torch.tanh(fake_combined[..., output_size:])
        
        # Reshape fake output to match conv2d output shape
        fake_output = fake_output_flat.view(x.shape[0], *temp_output.shape[1:])
        
        return fake_output, fake_proof 