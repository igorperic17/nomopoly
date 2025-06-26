"""
ONNX Operation Compiler

This module compiles individual ONNX operations into proof-capable components:
- Prover: Generates authentic proofs for operation execution
- Verifier: Validates (input, output, proof) triplets  
- Adversary: Generates fake proofs to test verifier robustness

Each operation is compiled with fixed input dimensions and exported as ONNX models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import logging
import os
import json
from datetime import datetime

from .ops_registry import OpCompilationInfo, SupportedOp
from .utils import convert_pytorch_to_onnx, validate_onnx_model


class ONNXOperationWrapper(nn.Module):
    """
    Wrapper for individual ONNX operations with proof generation capabilities.
    This replaces the VerifiableLayer approach with direct ONNX compilation.
    """
    
    def __init__(self, op_info: OpCompilationInfo, proof_dim: int = 32):
        super(ONNXOperationWrapper, self).__init__()
        self.op_info = op_info
        self.proof_dim = proof_dim
        
        # Create the actual operation implementation
        self.operation = self._create_operation_layer()
        
        # Proof generator for authentic proofs
        input_size = np.prod(op_info.input_shape)
        output_size = np.prod(op_info.output_shape)
        
        self.proof_generator = nn.Sequential(
            nn.Linear(input_size + output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, proof_dim),
            nn.Tanh()
        )
        
    def _create_operation_layer(self) -> nn.Module:
        """Create the actual PyTorch operation based on ONNX op type."""
        op_type = self.op_info.op_type
        attrs = self.op_info.attributes
        
        if op_type == SupportedOp.RELU:
            return nn.ReLU()
        
        elif op_type == SupportedOp.CONV2D:
            # Extract Conv2d parameters from attributes
            kernel_shape = attrs.get('kernel_shape', [3, 3])
            strides = attrs.get('strides', [1, 1])
            pads = attrs.get('pads', [0, 0, 0, 0])
            
            # Get input/output channels from shapes
            if len(self.op_info.input_shape) == 4:
                in_channels = self.op_info.input_shape[1]
                out_channels = self.op_info.output_shape[1]
            else:
                in_channels = 3  # Default
                out_channels = 32  # Default
            
            padding = pads[0] if isinstance(pads, list) else pads
            
            return nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_shape,
                stride=strides,
                padding=padding
            )
        
        elif op_type in [SupportedOp.MATMUL, SupportedOp.GEMM]:
            # Linear layer for matrix multiplication
            if len(self.op_info.input_shape) >= 2:
                in_features = self.op_info.input_shape[-1]
                out_features = self.op_info.output_shape[-1]
            else:
                in_features = 64  # Default
                out_features = 64  # Default
            
            return nn.Linear(in_features, out_features)
        
        elif op_type == SupportedOp.ADD:
            return nn.Identity()  # Addition is handled in forward
        
        elif op_type == SupportedOp.MAXPOOL:
            kernel_shape = attrs.get('kernel_shape', [2, 2])
            strides = attrs.get('strides', kernel_shape)
            pads = attrs.get('pads', [0, 0, 0, 0])
            padding = pads[0] if isinstance(pads, list) else pads
            
            return nn.MaxPool2d(
                kernel_size=kernel_shape,
                stride=strides,
                padding=padding
            )
        
        elif op_type == SupportedOp.AVGPOOL:
            kernel_shape = attrs.get('kernel_shape', [2, 2])
            strides = attrs.get('strides', kernel_shape)
            pads = attrs.get('pads', [0, 0, 0, 0])
            padding = pads[0] if isinstance(pads, list) else pads
            
            return nn.AvgPool2d(
                kernel_size=kernel_shape,
                stride=strides,
                padding=padding
            )
        
        elif op_type == SupportedOp.FLATTEN:
            return nn.Flatten()
        
        else:
            # Default identity operation
            return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the operation."""
        if self.op_info.op_type == SupportedOp.ADD:
            # For Add operation, we need two inputs - use x + x for demo
            return self.operation(x) + x
        else:
            return self.operation(x)
    
    def generate_proof(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
        """Generate an authentic proof for the operation execution."""
        input_flat = input_tensor.view(input_tensor.shape[0], -1)
        output_flat = output_tensor.view(output_tensor.shape[0], -1)
        combined = torch.cat([input_flat, output_flat], dim=-1)
        return self.proof_generator(combined)


class ONNXVerifier(nn.Module):
    """Verifier network for validating (input, output, proof) triplets."""
    
    def __init__(self, op_info: OpCompilationInfo, proof_dim: int = 32):
        super(ONNXVerifier, self).__init__()
        self.op_info = op_info
        self.proof_dim = proof_dim
        
        input_size = np.prod(op_info.input_shape)
        output_size = np.prod(op_info.output_shape)
        total_size = input_size + output_size + proof_dim
        
        self.verifier = nn.Sequential(
            nn.Linear(total_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, proof: torch.Tensor) -> torch.Tensor:
        """Verify if the triplet is authentic."""
        input_flat = input_tensor.view(input_tensor.shape[0], -1)
        output_flat = output_tensor.view(output_tensor.shape[0], -1)
        proof_flat = proof.view(proof.shape[0], -1)
        
        triplet = torch.cat([input_flat, output_flat, proof_flat], dim=-1)
        return self.verifier(triplet)


class ONNXAdversary(nn.Module):
    """Adversary network for generating fake outputs and proofs."""
    
    def __init__(self, op_info: OpCompilationInfo, proof_dim: int = 32):
        super(ONNXAdversary, self).__init__()
        self.op_info = op_info
        self.proof_dim = proof_dim
        
        input_size = np.prod(op_info.input_shape)
        output_size = np.prod(op_info.output_shape)
        
        # Fake output generator
        self.output_generator = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
        # Fake proof generator
        self.proof_generator = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, proof_dim),
            nn.Tanh()
        )
    
    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate fake output and proof."""
        input_flat = input_tensor.view(input_tensor.shape[0], -1)
        
        fake_output_flat = self.output_generator(input_flat)
        fake_proof = self.proof_generator(input_flat)
        
        # Reshape output to match expected shape
        fake_output = fake_output_flat.view(-1, *self.op_info.output_shape[1:])
        
        return fake_output, fake_proof


class ONNXOperationCompiler:
    """
    Compiler for individual ONNX operations into proof-capable components.
    """
    
    def __init__(self, device: str = "mps"):
        # Use MPS if available, otherwise fallback
        if device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.criterion = nn.BCELoss()
    
    def compile_operation(
        self, 
        op_info: OpCompilationInfo, 
        num_epochs: int = 200,
        batch_size: int = 32,
        proof_dim: int = 32
    ) -> Dict[str, Any]:
        """
        Compile a single ONNX operation into proof-capable components.
        
        Args:
            op_info: Operation information and metadata
            num_epochs: Number of training epochs
            batch_size: Training batch size
            proof_dim: Dimension of proof vectors
            
        Returns:
            Dictionary with compilation results and metrics
        """
        print(f"\n🔧 Compiling operation: {op_info.folder_name}")
        print(f"   Input shape: {op_info.input_shape}")
        print(f"   Output shape: {op_info.output_shape}")
        
        # Set up logging
        log_file = op_info.compilation_log_path
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(f"compiler_{op_info.folder_name}")
        
        logger.info(f"Starting compilation of {op_info.folder_name}")
        logger.warning(f"IMPORTANT: Models compiled with fixed input dimensions {op_info.input_shape}")
        logger.warning("Models will break if input shape differs during inference!")
        
        # Create networks
        prover = ONNXOperationWrapper(op_info, proof_dim).to(self.device)
        verifier = ONNXVerifier(op_info, proof_dim).to(self.device)
        adversary = ONNXAdversary(op_info, proof_dim).to(self.device)
        
        # Freeze prover operation (only train proof generator)
        for param in prover.operation.parameters():
            param.requires_grad = False
        
        # Optimizers
        verifier_optimizer = optim.Adam(verifier.parameters(), lr=0.002)
        adversary_optimizer = optim.Adam(adversary.parameters(), lr=0.0005)
        prover_proof_optimizer = optim.Adam(prover.proof_generator.parameters(), lr=0.001)
        
        # Training metrics
        metrics = {
            "verifier_loss": [],
            "adversary_loss": [],
            "verifier_accuracy": [],
            "adversary_fool_rate": []
        }
        
        # Training loop
        logger.info(f"Training for {num_epochs} epochs...")
        
        for epoch in tqdm(range(num_epochs), desc=f"Compiling {op_info.op_type.value}"):
            # Generate training data
            input_data = self._generate_input_data(op_info, batch_size)
            
            # === STEP 1: Generate real examples ===
            with torch.no_grad():
                real_output = prover.forward(input_data)
            real_proof = prover.generate_proof(input_data, real_output)
            
            # === STEP 2: Generate fake examples ===
            fake_output, fake_proof = adversary(input_data)
            
            # === STEP 3: Train Verifier ===
            verifier_optimizer.zero_grad()
            
            # Real triplets should be accepted
            real_scores = verifier(input_data, real_output.detach(), real_proof.detach())
            real_loss = self.criterion(real_scores, torch.ones_like(real_scores))
            
            # Fake triplets should be rejected  
            fake_scores = verifier(input_data, fake_output.detach(), fake_proof.detach())
            fake_loss = self.criterion(fake_scores, torch.zeros_like(fake_scores))
            
            verifier_loss = real_loss + fake_loss
            verifier_loss.backward()
            verifier_optimizer.step()
            
            # === STEP 4: Train Adversary ===
            adversary_optimizer.zero_grad()
            
            # Generate fresh fake samples for adversary training
            adv_fake_output, adv_fake_proof = adversary(input_data)
            adv_scores = verifier(input_data, adv_fake_output, adv_fake_proof)
            adversary_loss = self.criterion(adv_scores, torch.ones_like(adv_scores))
            adversary_loss.backward()
            adversary_optimizer.step()
            
            # === STEP 5: Update prover proof generator ===
            prover_proof_optimizer.zero_grad()
            
            # Generate fresh real samples for prover training
            fresh_real_output = prover.forward(input_data)
            fresh_real_proof = prover.generate_proof(input_data, fresh_real_output)
            fresh_real_scores = verifier(input_data, fresh_real_output.detach(), fresh_real_proof)
            prover_proof_loss = self.criterion(fresh_real_scores, torch.ones_like(fresh_real_scores))
            prover_proof_loss.backward()
            prover_proof_optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                # Recalculate scores for metrics
                eval_real_scores = verifier(input_data, real_output.detach(), real_proof.detach())
                eval_fake_scores = verifier(input_data, fake_output.detach(), fake_proof.detach())
                
                real_acc = (eval_real_scores > 0.5).float().mean().item()
                fake_acc = (eval_fake_scores < 0.5).float().mean().item()
                verifier_acc = (real_acc + fake_acc) / 2
                fool_rate = (adv_scores > 0.5).float().mean().item()
                
                metrics["verifier_loss"].append(verifier_loss.item())
                metrics["adversary_loss"].append(adversary_loss.item())
                metrics["verifier_accuracy"].append(verifier_acc)
                metrics["adversary_fool_rate"].append(fool_rate)
            
            # Log progress
            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch {epoch + 1}: Verifier Acc: {verifier_acc:.3f}, "
                          f"Adversary Fool Rate: {fool_rate:.3f}")
        
        logger.info(f"Training completed! Final verifier accuracy: {metrics['verifier_accuracy'][-1]:.3f}")
        
        # === STEP 6: Export to ONNX ===
        dummy_input = self._generate_input_data(op_info, 1)
        
        try:
            # Ensure output directories exist
            from pathlib import Path
            for path in [op_info.prover_onnx_path, op_info.verifier_onnx_path, op_info.adversary_onnx_path]:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export prover
            dummy_output = prover.forward(dummy_input)
            dummy_proof = prover.generate_proof(dummy_input, dummy_output)
            
            self._export_prover_onnx(prover, dummy_input, op_info.prover_onnx_path)
            self._export_verifier_onnx(verifier, dummy_input, dummy_output, dummy_proof, op_info.verifier_onnx_path)
            self._export_adversary_onnx(adversary, dummy_input, op_info.adversary_onnx_path)
            
            logger.info("✅ Successfully exported all ONNX models")
            
        except Exception as e:
            logger.error(f"❌ Failed to export ONNX models: {e}")
            return {"success": False, "error": str(e)}
        
        # Save metrics
        metrics_path = op_info.compilation_log_path.replace('.log', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"📊 Saved training metrics to {metrics_path}")
        
        # Update op_info
        op_info.training_metrics = metrics
        op_info.is_compiled = True
        
        return {
            "success": True,
            "metrics": metrics,
            "final_verifier_accuracy": metrics["verifier_accuracy"][-1],
            "final_adversary_fool_rate": metrics["adversary_fool_rate"][-1]
        }
    
    def _generate_input_data(self, op_info: OpCompilationInfo, batch_size: int) -> torch.Tensor:
        """Generate random input data matching the operation's input shape."""
        shape = (batch_size,) + op_info.input_shape[1:]  # Skip batch dimension
        return torch.randn(shape, device=self.device)
    
    def _export_prover_onnx(self, prover: ONNXOperationWrapper, dummy_input: torch.Tensor, output_path: str):
        """Export prover to ONNX format."""
        prover.eval()
        
        class ProverWrapper(nn.Module):
            def __init__(self, prover_net):
                super().__init__()
                self.prover = prover_net
            
            def forward(self, x):
                output = self.prover.forward(x)
                proof = self.prover.generate_proof(x, output)
                return output, proof
        
        wrapper = ProverWrapper(prover)
        
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output', 'proof'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
                'proof': {0: 'batch_size'}
            }
        )
    
    def _export_verifier_onnx(self, verifier: ONNXVerifier, dummy_input: torch.Tensor, 
                             dummy_output: torch.Tensor, dummy_proof: torch.Tensor, output_path: str):
        """Export verifier to ONNX format."""
        verifier.eval()
        
        torch.onnx.export(
            verifier,
            (dummy_input, dummy_output, dummy_proof),
            output_path,
            export_params=True,
            opset_version=11,
            input_names=['input', 'output', 'proof'],
            output_names=['verification_score'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
                'proof': {0: 'batch_size'},
                'verification_score': {0: 'batch_size'}
            }
        )
    
    def _export_adversary_onnx(self, adversary: ONNXAdversary, dummy_input: torch.Tensor, output_path: str):
        """Export adversary to ONNX format."""
        adversary.eval()
        
        torch.onnx.export(
            adversary,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['fake_output', 'fake_proof'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'fake_output': {0: 'batch_size'},
                'fake_proof': {0: 'batch_size'}
            }
        ) 