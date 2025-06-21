"""
Core neural network architectures for Zero Knowledge ML using Holographic Reduced Representations.

This module implements the HRR-based approach for fixed-size proof generation
regardless of network complexity, inspired by zkGAP's breakthrough methodology.
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
            
        return output
    
    def forward(self, x):
        """Forward pass with integrated holographic memory."""
        # Create fresh holographic memory for this forward pass
        holographic_memory = HolographicMemory(self.memory_size)
        
        # Run model with holographic memory integration
        result = self._bind_intermediate_activations(x, holographic_memory)
        
        # Generate proof from holographic memory
        memory_trace = holographic_memory.get_memory_trace()
        
        # Ensure memory trace matches batch size
        if memory_trace.shape[0] != x.shape[0]:
            memory_trace = memory_trace.expand(x.shape[0], -1)
            
        proof = self.proof_generator(memory_trace)
        
        return result, proof
    
    def export_original_vs_augmented(self, save_dir: str = "exported_models"):
        """Export both original and augmented networks to compare architectures."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Store original device
        original_device = next(self.parameters()).device
        
        # Export original network
        self.base_model.cpu()
        dummy_input = torch.randn(1, 196)  # MNIST input size
        
        torch.onnx.export(
            self.base_model,
            dummy_input,
            f"{save_dir}/original_mnist_net.onnx",
            export_params=True,
            opset_version=10,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Export holographic wrapper
        self.cpu()
        torch.onnx.export(
            self,
            dummy_input,
            f"{save_dir}/holographic_wrapper.onnx",
            export_params=True,
            opset_version=10,
            input_names=['input'],
            output_names=['output', 'proof'],
            dynamic_axes={
                'input': {0: 'batch_size'}, 
                'output': {0: 'batch_size'},
                'proof': {0: 'batch_size'}
            }
        )
        
        # Move back to original device
        self.to(original_device)
        self.base_model.to(original_device)
        
        print(f"✅ Exported networks to {save_dir}/:")
        print(f"   - original_mnist_net.onnx (original network)")
        print(f"   - holographic_wrapper.onnx (HRR-augmented network)")


class ZKProverNet(nn.Module):
    """
    ZK Prover Network: Uses HolographicWrapper for fixed-size proof generation.
    This is our main "inference network" that generates authentic proofs.
    """
    
    def __init__(self, input_dim: int, output_dim: int, proof_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proof_dim = proof_dim
        
        # Create original network
        original_net = OriginalMNISTNet(input_dim, output_dim)
        
        # Wrap with holographic memory
        self.holographic_model = HolographicWrapper(original_net, proof_dim, memory_size=512)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate both classification and proof using HRR."""
        return self.holographic_model(x)
    
    def get_original_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the original computation result."""
        return self.holographic_model.base_model(x)
    
    def export_networks(self, save_dir: str = "exported_models"):
        """Export original vs holographic networks."""
        self.holographic_model.export_original_vs_augmented(save_dir)


class ZKVerifierNet(nn.Module):
    """
    ZK Verifier Network: Validates (input, output, proof) triplets.
    Verifies that the proof is authentic for the given input-output pair.
    """
    
    def __init__(self, input_dim: int, output_dim: int, proof_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proof_dim = proof_dim
        self.layers = None
        
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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)
        
    def forward(self, input_data: torch.Tensor, output_data: torch.Tensor, proof: torch.Tensor) -> torch.Tensor:
        """
        Verify if the (input, output, proof) triplet is valid.
        
        Args:
            input_data: Original input to the computation
            output_data: Claimed output of the computation
            proof: Proof that the computation was performed correctly
            
        Returns:
            Verification score (0-1): 1 = valid triplet, 0 = invalid triplet
        """
        if self.layers is None:
            # Build layers dynamically
            self._build_layers(input_data.shape[-1], output_data.shape[-1], proof.shape[-1], input_data.device)
        
        # Concatenate input, output, and proof
        combined = torch.cat([input_data, output_data, proof], dim=-1)
        
        # Return verification score
        return self.layers(combined)


class ZKAdversarialNet(nn.Module):
    """
    ZK Adversarial Network: Generates fake outputs and fake proofs.
    Creates different types of adversarial samples to test verifier robustness.
    """
    
    def __init__(self, input_dim: int, output_dim: int, proof_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proof_dim = proof_dim
        
        # Network for generating slightly different outputs (corrupted but plausible)
        self.corrupted_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),  # Higher dropout for more variation
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, output_dim),
            nn.LogSoftmax(dim=-1)
        )
        
        # Network for generating completely random outputs
        self.random_output_generator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.LogSoftmax(dim=-1)
        )
        
        # Fake proof generator that tries to create convincing proofs for wrong outputs
        self.fake_proof_generator = nn.Sequential(
            nn.Linear(input_dim + output_dim, 256),  # Takes input + fake output
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, proof_dim),
            nn.Tanh()  # Bounded output
        )
        
        # Noise layer for proof variation
        self.proof_noise_layer = nn.Sequential(
            nn.Linear(proof_dim, proof_dim),
            nn.ReLU(),
            nn.Linear(proof_dim, proof_dim)
        )
        
    def forward(self, x: torch.Tensor, mode: str = "mixed") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate fake outputs and fake proofs.
        
        Args:
            x: Input tensor
            mode: Type of fake output to generate
                - "corrupted": Slightly different from correct output
                - "random": Completely random output
                - "mixed": Randomly choose between corrupted and random
                
        Returns:
            Tuple of (fake_output, fake_proof)
        """
        batch_size = x.shape[0]
        
        if mode == "mixed":
            # Randomly choose between corrupted and random for each sample
            use_corrupted = torch.rand(batch_size, 1, device=x.device) > 0.5
            
            corrupted_output = self.corrupted_classifier(x)
            random_output = self.random_output_generator(x)
            
            # Mix outputs based on random choice
            fake_output = torch.where(use_corrupted, corrupted_output, random_output)
            
        elif mode == "corrupted":
            fake_output = self.corrupted_classifier(x)
            
        elif mode == "random":
            fake_output = self.random_output_generator(x)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Generate fake proof for the fake output
        proof_input = torch.cat([x, fake_output], dim=-1)
        base_fake_proof = self.fake_proof_generator(proof_input)
        
        # Add noise to make proof more convincing
        fake_proof = base_fake_proof + self.proof_noise_layer(base_fake_proof) * 0.1
        
        return fake_output, fake_proof
    
    def generate_corrupted_output(self, x: torch.Tensor) -> torch.Tensor:
        """Generate slightly corrupted but plausible output."""
        return self.corrupted_classifier(x)
    
    def generate_random_output(self, x: torch.Tensor) -> torch.Tensor:
        """Generate completely random output."""
        return self.random_output_generator(x)


# Factory function for easy creation
def create_holographic_model(base_model, proof_size=64, memory_size=512):
    """
    Create a holographic memory-enabled model.
    
    Args:
        base_model: The original PyTorch model
        proof_size: Size of the generated proof (fixed!)
        memory_size: Size of the holographic memory
        
    Returns:
        HolographicWrapper model that generates fixed-size proofs
    """
    return HolographicWrapper(base_model, proof_size, memory_size)


# Keep compatibility classes
class SimpleONNXComputation(nn.Module):
    """Simple wrapper for ONNX model computation (for compatibility)."""
    
    def __init__(self, onnx_model_path: Optional[str] = None):
        super().__init__()
        self.onnx_model_path = onnx_model_path
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder computation
        return F.log_softmax(x @ torch.randn(x.shape[-1], 10), dim=-1) 