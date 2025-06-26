"""
ONNX Operation Compiler with Neural Architecture Search

This module compiles individual ONNX operations into proof-capable components:
- Prover: Generates authentic proofs for operation execution
- Verifier: Validates (input, output, proof) triplets  
- Adversary: Generates fake proofs to test verifier robustness

Enhanced with Neural Architecture Search (NAS) to achieve 99.999% accuracy.
Each operation is compiled with evolved architectures and exported as ONNX models.
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
import random
import copy
from datetime import datetime
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass, asdict

from .ops_registry import OpCompilationInfo, SupportedOp
from .utils import convert_pytorch_to_onnx, validate_onnx_model


class ActivationType(Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SWISH = "swish"
    MISH = "mish"
    ELU = "elu"
    TANH = "tanh"
    SIGMOID = "sigmoid"


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw" 
    SGD = "sgd"
    RMSPROP = "rmsprop"


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search."""
    # Architecture parameters
    hidden_layers: List[int]
    activation: ActivationType
    dropout_rates: List[float]
    use_batch_norm: bool
    use_layer_norm: bool
    use_residual: bool
    
    # Training parameters
    optimizer: OptimizerType
    learning_rate: float
    weight_decay: float
    batch_size: int
    
    # Advanced techniques
    use_gradient_clipping: bool
    gradient_clip_value: float
    use_label_smoothing: bool
    label_smoothing: float
    use_mixup: bool
    mixup_alpha: float
    
    # Ensemble parameters
    ensemble_size: int
    use_ensemble: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['activation'] = self.activation.value
        result['optimizer'] = self.optimizer.value
        return result


class AdvancedVerifier(nn.Module):
    """Advanced verifier with NAS-evolved architecture."""
    
    def __init__(self, op_info: OpCompilationInfo, config: NASConfig, proof_dim: int = 64):
        super().__init__()
        self.config = config
        self.proof_dim = proof_dim
        
        input_size = np.prod(op_info.input_shape)
        output_size = np.prod(op_info.output_shape)
        total_size = input_size + output_size + proof_dim
        
        # Build evolved architecture
        layers = []
        current_size = total_size
        
        for i, hidden_size in enumerate(config.hidden_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            
            layers.append(self._get_activation(config.activation))
            
            if i < len(config.dropout_rates):
                layers.append(nn.Dropout(config.dropout_rates[i]))
            
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: ActivationType) -> nn.Module:
        """Get activation function."""
        if activation == ActivationType.RELU:
            return nn.ReLU()
        elif activation == ActivationType.LEAKY_RELU:
            return nn.LeakyReLU(0.2)
        elif activation == ActivationType.GELU:
            return nn.GELU()
        elif activation == ActivationType.SWISH:
            return nn.SiLU()
        elif activation == ActivationType.MISH:
            return nn.Mish()
        elif activation == ActivationType.ELU:
            return nn.ELU()
        elif activation == ActivationType.TANH:
            return nn.Tanh()
        elif activation == ActivationType.SIGMOID:
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def forward(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, proof: torch.Tensor) -> torch.Tensor:
        """Forward pass with evolved architecture."""
        input_flat = input_tensor.view(input_tensor.shape[0], -1)
        output_flat = output_tensor.view(output_tensor.shape[0], -1)
        proof_flat = proof.view(proof.shape[0], -1)
        
        triplet = torch.cat([input_flat, output_flat, proof_flat], dim=-1)
        return self.network(triplet)


class EnsembleVerifier(nn.Module):
    """Ensemble of verifiers for ultra-high precision."""
    
    def __init__(self, verifiers: List[nn.Module]):
        super().__init__()
        self.verifiers = nn.ModuleList(verifiers)
    
    def forward(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, proof: torch.Tensor) -> torch.Tensor:
        """Ensemble prediction through voting."""
        outputs = []
        for verifier in self.verifiers:
            outputs.append(verifier(input_tensor, output_tensor, proof))
        
        # Average ensemble prediction
        return torch.stack(outputs).mean(dim=0)


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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute the operation AND generate proof in one forward pass.
        
        Returns:
            Tuple[output, proof] - Original operation output + authenticity proof
        """
        # Execute the original operation
        if self.op_info.op_type == SupportedOp.ADD:
            # For Add operation, we need two inputs - use x + x for demo
            output = self.operation(x) + x
        else:
            output = self.operation(x)
        
        # Generate proof for this execution
        proof = self.generate_proof(x, output)
        
        return output, proof
    
    def generate_proof(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
        """Generate an authentic proof for the operation execution."""
        input_flat = input_tensor.view(input_tensor.shape[0], -1)
        output_flat = output_tensor.view(output_tensor.shape[0], -1)
        combined = torch.cat([input_flat, output_flat], dim=-1)
        return self.proof_generator(combined)
    
    def execute_only(self, x: torch.Tensor) -> torch.Tensor:
        """Execute only the operation (for backward compatibility)."""
        if self.op_info.op_type == SupportedOp.ADD:
            return self.operation(x) + x
        else:
            return self.operation(x)


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
        
        # NAS search space for ultra-precision
        self.search_space = {
            'hidden_layers': [
                [256, 128], [512, 256], [1024, 512], [2048, 1024],
                [256, 256, 128], [512, 512, 256], [1024, 1024, 512],
                [256, 512, 256], [512, 1024, 512], [1024, 2048, 1024],
                [512, 1024, 2048, 1024, 512], [1024, 2048, 4096, 2048, 1024]
            ],
            'activations': list(ActivationType),
            'dropout_rates': [
                [0.1], [0.2], [0.3], [0.0],
                [0.1, 0.2], [0.2, 0.3], [0.1, 0.3],
                [0.1, 0.2, 0.3], [0.0, 0.1, 0.2]
            ],
            'optimizers': list(OptimizerType),
            'learning_rates': [0.0001, 0.00005, 0.00001, 0.000005, 0.000001],
            'batch_sizes': [32, 64, 128, 256],
            'weight_decays': [0.0, 1e-5, 1e-4, 1e-3]
        }
    
    def compile_operation(
        self, 
        op_info: OpCompilationInfo, 
        num_epochs: int = 200,
        batch_size: int = 32,
        proof_dim: int = 32,
        target_accuracy: float = 0.99,
        max_epochs: int = 1000
    ) -> Dict[str, Any]:
        """
        Compile a single ONNX operation into proof-capable components.
        
        Args:
            op_info: Operation information and metadata
            num_epochs: Minimum number of training epochs (will continue until target_accuracy)
            batch_size: Training batch size
            proof_dim: Dimension of proof vectors
            target_accuracy: Target verifier accuracy (default 99%)
            max_epochs: Maximum number of epochs to prevent infinite training
            
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
        # Adaptive learning rates for ultra-high precision
        base_verifier_lr = 0.001 if target_accuracy >= 0.999 else 0.002
        base_adversary_lr = 0.0002 if target_accuracy >= 0.999 else 0.0005
        base_prover_lr = 0.0005 if target_accuracy >= 0.999 else 0.001
        
        verifier_optimizer = optim.Adam(verifier.parameters(), lr=base_verifier_lr, weight_decay=1e-5)
        adversary_optimizer = optim.Adam(adversary.parameters(), lr=base_adversary_lr, weight_decay=1e-5)
        prover_proof_optimizer = optim.Adam(prover.proof_generator.parameters(), lr=base_prover_lr, weight_decay=1e-5)
        
        # Learning rate schedulers for ultra-high precision
        verifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            verifier_optimizer, mode='max', factor=0.8, patience=30, min_lr=1e-6
        )
        adversary_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            adversary_optimizer, mode='min', factor=0.8, patience=30, min_lr=1e-6
        )
        
        # Training metrics
        metrics = {
            "verifier_loss": [],
            "adversary_loss": [],
            "verifier_accuracy": [],
            "adversary_fool_rate": []
        }
        
        # Training loop with target accuracy
        logger.info(f"Training until {target_accuracy:.2%} accuracy (min {num_epochs}, max {max_epochs} epochs)...")
        
        best_accuracy = 0.0
        epochs_without_improvement = 0
        patience = 100 if target_accuracy >= 0.999 else 50  # Extended patience for ultra-high precision
        
        epoch = 0
        pbar = tqdm(desc=f"Compiling {op_info.op_type.value}")
        
        while epoch < max_epochs:
            # Generate training data
            input_data = self._generate_input_data(op_info, batch_size)
            
            # === STEP 1: Generate real examples ===
            with torch.no_grad():
                real_output, real_proof = prover.forward(input_data)
            
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
            fresh_real_output, fresh_real_proof = prover.forward(input_data)
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
            
            # Update progress bar
            pbar.set_postfix({
                'Epoch': epoch + 1,
                'Acc': f"{verifier_acc:.4f}",
                'Target': f"{target_accuracy:.4f}",
                'Best': f"{best_accuracy:.4f}"
            })
            pbar.update(1)
            
            # Check for improvement
            if verifier_acc > best_accuracy:
                best_accuracy = verifier_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Update learning rate schedulers for ultra-high precision
            if target_accuracy >= 0.999:
                verifier_scheduler.step(verifier_acc)
                adversary_scheduler.step(metrics["adversary_loss"][-1] if metrics["adversary_loss"] else 1.0)
            
            # Log progress
            if (epoch + 1) % 50 == 0:
                current_verifier_lr = verifier_optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch + 1}: Verifier Acc: {verifier_acc:.4f}, "
                          f"Adversary Fool Rate: {fool_rate:.3f}, Best: {best_accuracy:.4f}, LR: {current_verifier_lr:.2e}")
            
            # Check termination conditions
            epoch += 1
            
            # Target accuracy reached and minimum epochs completed
            if verifier_acc >= target_accuracy and epoch >= num_epochs:
                if target_accuracy >= 0.999:
                    logger.info(f"🏆 ULTRA-PRECISION ACHIEVED! {target_accuracy:.2%} accuracy reached at epoch {epoch}!")
                else:
                    logger.info(f"🎯 Target accuracy {target_accuracy:.1%} reached at epoch {epoch}!")
                break
                
            # Early stopping if no improvement
            if epochs_without_improvement >= patience and epoch >= num_epochs:
                logger.info(f"⏹️  Early stopping: No improvement for {patience} epochs")
                break
        
        pbar.close()
        
        final_accuracy = metrics['verifier_accuracy'][-1]
        if final_accuracy >= target_accuracy:
            if target_accuracy >= 0.999:
                logger.info(f"🏆 ULTRA-PRECISION TRAINING COMPLETE! Achieved: {final_accuracy:.4f} (target: {target_accuracy:.4f})")
            else:
                logger.info(f"✅ Training completed! Target accuracy achieved: {final_accuracy:.3f}")
        else:
            precision_str = f"{target_accuracy:.4f}" if target_accuracy >= 0.999 else f"{target_accuracy:.3f}"
            logger.info(f"⚠️  Training completed at max epochs. Final accuracy: {final_accuracy:.4f} (target: {precision_str})")
        
        # === STEP 6: Export to ONNX ===
        dummy_input = self._generate_input_data(op_info, 1)
        
        try:
            # Ensure output directories exist (all paths should now be in the operation folder)
            from pathlib import Path
            for path in [op_info.prover_onnx_path, op_info.verifier_onnx_path, op_info.adversary_onnx_path]:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export prover
            dummy_output, dummy_proof = prover.forward(dummy_input)
            
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
        
        # Generate training plots
        self._plot_training_metrics(op_info, metrics, logger)
        
        # Update op_info
        op_info.training_metrics = metrics
        op_info.is_compiled = True
        
        return {
            "success": True,
            "metrics": metrics,
            "final_verifier_accuracy": metrics["verifier_accuracy"][-1],
            "final_adversary_fool_rate": metrics["adversary_fool_rate"][-1]
        }
    
    def evolve_architecture_to_precision(
        self,
        op_info: OpCompilationInfo,
        target_accuracy: float = 0.99999,
        max_generations: int = 20,
        population_size: int = 10,
        proof_dim: int = 64
    ) -> Dict[str, Any]:
        """
        Use Neural Architecture Search to evolve models until target accuracy (99.999%).
        
        Args:
            op_info: Operation information and metadata
            target_accuracy: Target accuracy (default 99.999% for 5 nines)
            max_generations: Maximum generations for evolution
            population_size: Size of population in each generation
            proof_dim: Dimension of proof vectors
            
        Returns:
            Dictionary with evolution results and best configuration
        """
        
        print(f"\n🧬 EVOLVING ARCHITECTURE FOR {op_info.folder_name} TO {target_accuracy:.5f} ACCURACY")
        print(f"🎯 Target: 5 NINES (99.999%) precision")
        
        # Set up logging
        log_file = op_info.compilation_log_path.replace('.log', '_nas.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(f"nas_{op_info.folder_name}")
        
        logger.info(f"🧬 Starting NAS evolution for {op_info.folder_name}")
        logger.info(f"🎯 Target accuracy: {target_accuracy:.5f}")
        logger.info(f"🔄 Max generations: {max_generations}")
        logger.info(f"👥 Population size: {population_size}")
        
        # Initialize population
        population = []
        for _ in range(population_size):
            config = self._create_random_nas_config()
            population.append(config)
        
        best_config = None
        best_accuracy = 0.0
        evolution_history = []
        
        for generation in range(max_generations):
            logger.info(f"\n🧬 GENERATION {generation + 1}/{max_generations}")
            
            # Evaluate each individual in population
            fitness_scores = []
            generation_results = []
            
            for i, config in enumerate(population):
                logger.info(f"   🧪 Evaluating individual {i + 1}/{len(population)}")
                
                try:
                    # Train with this configuration
                    accuracy = self._evaluate_nas_config(op_info, config, proof_dim, logger)
                    fitness_scores.append(accuracy)
                    
                    generation_results.append({
                        'individual': i,
                        'config': config.to_dict(),
                        'accuracy': accuracy
                    })
                    
                    # Update best
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_config = copy.deepcopy(config)
                        logger.info(f"🎯 NEW BEST: {accuracy:.5f} accuracy")
                    
                    # Check if target reached
                    if accuracy >= target_accuracy:
                        logger.info(f"🏆 TARGET ACHIEVED: {accuracy:.5f} >= {target_accuracy:.5f}")
                        
                        # Save best configuration
                        self._save_nas_config(op_info, best_config, best_accuracy)
                        
                        return {
                            "success": True,
                            "target_achieved": True,
                            "best_accuracy": accuracy,
                            "best_config": config.to_dict(),
                            "generation": generation + 1,
                            "evolution_history": evolution_history
                        }
                        
                except Exception as e:
                    logger.error(f"❌ Individual {i} failed: {str(e)}")
                    fitness_scores.append(0.0)
                    generation_results.append({
                        'individual': i,
                        'config': config.to_dict(),
                        'accuracy': 0.0,
                        'error': str(e)
                    })
            
            # Record generation statistics
            gen_best = max(fitness_scores) if fitness_scores else 0.0
            gen_avg = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
            
            evolution_history.append({
                'generation': generation + 1,
                'best_accuracy': gen_best,
                'average_accuracy': gen_avg,
                'results': generation_results
            })
            
            logger.info(f"   📊 Generation {generation + 1} - Best: {gen_best:.5f}, Avg: {gen_avg:.5f}")
            
            # Evolution: Create next generation
            if generation < max_generations - 1:  # Don't evolve on last generation
                population = self._evolve_population(population, fitness_scores, population_size)
        
        # Evolution completed without reaching target
        logger.info(f"🧬 Evolution completed. Best accuracy: {best_accuracy:.5f}")
        
        if best_config:
            self._save_nas_config(op_info, best_config, best_accuracy)
        
        return {
            "success": True,
            "target_achieved": best_accuracy >= target_accuracy,
            "best_accuracy": best_accuracy,
            "best_config": best_config.to_dict() if best_config else None,
            "evolution_history": evolution_history
        }
    
    def _create_random_nas_config(self) -> NASConfig:
        """Create a random NAS configuration."""
        hidden_layers = random.choice(self.search_space['hidden_layers'])
        dropout_rates = random.choice(self.search_space['dropout_rates'])
        
        # Ensure dropout rates match hidden layers
        while len(dropout_rates) < len(hidden_layers):
            dropout_rates.append(dropout_rates[-1])
        dropout_rates = dropout_rates[:len(hidden_layers)]
        
        return NASConfig(
            hidden_layers=hidden_layers,
            activation=random.choice(self.search_space['activations']),
            dropout_rates=dropout_rates,
            use_batch_norm=random.choice([True, False]),
            use_layer_norm=random.choice([True, False]),
            use_residual=random.choice([True, False]),
            optimizer=random.choice(self.search_space['optimizers']),
            learning_rate=random.choice(self.search_space['learning_rates']),
            weight_decay=random.choice(self.search_space['weight_decays']),
            batch_size=random.choice(self.search_space['batch_sizes']),
            use_gradient_clipping=random.choice([True, False]),
            gradient_clip_value=random.uniform(0.5, 2.0),
            use_label_smoothing=random.choice([True, False]),
            label_smoothing=random.uniform(0.05, 0.2),
            use_mixup=random.choice([True, False]),
            mixup_alpha=random.uniform(0.1, 0.4),
            ensemble_size=random.choice([1, 3, 5]),
            use_ensemble=random.choice([True, False])
        )
    
    def _evaluate_nas_config(
        self, 
        op_info: OpCompilationInfo, 
        config: NASConfig, 
        proof_dim: int,
        logger
    ) -> float:
        """Evaluate a NAS configuration and return accuracy."""
        
        try:
            # Create models with this configuration
            if config.use_ensemble and config.ensemble_size > 1:
                verifiers = []
                for _ in range(config.ensemble_size):
                    verifiers.append(AdvancedVerifier(op_info, config, proof_dim))
                verifier = EnsembleVerifier(verifiers).to(self.device)
            else:
                verifier = AdvancedVerifier(op_info, config, proof_dim).to(self.device)
            
            adversary = ONNXAdversary(op_info, proof_dim).to(self.device)
            prover = ONNXOperationWrapper(op_info, proof_dim).to(self.device)
            
            # Create optimizer
            if config.optimizer == OptimizerType.ADAM:
                verifier_opt = optim.Adam(verifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                adversary_opt = optim.Adam(adversary.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            elif config.optimizer == OptimizerType.ADAMW:
                verifier_opt = optim.AdamW(verifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                adversary_opt = optim.AdamW(adversary.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            elif config.optimizer == OptimizerType.SGD:
                verifier_opt = optim.SGD(verifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
                adversary_opt = optim.SGD(adversary.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
            else:  # RMSPROP
                verifier_opt = optim.RMSprop(verifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                adversary_opt = optim.RMSprop(adversary.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            
            # Training loop
            best_accuracy = 0.0
            patience_counter = 0
            patience = 50
            max_epochs = 300  # Shorter evaluation epochs
            
            for epoch in range(max_epochs):
                # Generate training data
                input_data = self._generate_input_data(op_info, config.batch_size)
                
                # Apply mixup if enabled
                if config.use_mixup:
                    lam = np.random.beta(config.mixup_alpha, config.mixup_alpha)
                    batch_size = input_data.size(0)
                    index = torch.randperm(batch_size).to(input_data.device)
                    input_data = lam * input_data + (1 - lam) * input_data[index, :]
                
                # Training step
                verifier_acc = self._nas_train_step(
                    prover, verifier, adversary,
                    verifier_opt, adversary_opt,
                    input_data, config
                )
                
                # Track best accuracy
                if verifier_acc > best_accuracy:
                    best_accuracy = verifier_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    break
                
                # Check for ultra-precision early
                if verifier_acc >= 0.99999:
                    logger.info(f"🎯 ULTRA-PRECISION ACHIEVED: {verifier_acc:.5f}")
                    return verifier_acc
            
            return best_accuracy
            
        except Exception as e:
            logger.error(f"❌ Configuration evaluation failed: {str(e)}")
            return 0.0
    
    def _nas_train_step(
        self,
        prover: ONNXOperationWrapper,
        verifier: nn.Module,
        adversary: ONNXAdversary,
        verifier_opt: torch.optim.Optimizer,
        adversary_opt: torch.optim.Optimizer,
        input_data: torch.Tensor,
        config: NASConfig
    ) -> float:
        """Single training step for NAS evaluation."""
        
        # Generate real and fake examples
        with torch.no_grad():
            real_output, real_proof = prover(input_data)
        fake_output, fake_proof = adversary(input_data)
        
        # Train verifier
        verifier_opt.zero_grad()
        
        # Real examples (should be accepted)
        real_scores = verifier(input_data, real_output.detach(), real_proof.detach())
        real_targets = torch.ones_like(real_scores)
        
        # Fake examples (should be rejected)
        fake_scores = verifier(input_data, fake_output.detach(), fake_proof.detach())
        fake_targets = torch.zeros_like(fake_scores)
        
        # Combine targets and scores
        all_scores = torch.cat([real_scores, fake_scores])
        all_targets = torch.cat([real_targets, fake_targets])
        
        # Apply label smoothing if enabled
        if config.use_label_smoothing:
            all_targets = all_targets * (1.0 - config.label_smoothing) + config.label_smoothing * 0.5
        
        verifier_loss = F.binary_cross_entropy(all_scores, all_targets)
        verifier_loss.backward()
        
        # Gradient clipping if enabled
        if config.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), config.gradient_clip_value)
        
        verifier_opt.step()
        
        # Train adversary
        adversary_opt.zero_grad()
        fake_output, fake_proof = adversary(input_data)
        adversary_scores = verifier(input_data, fake_output, fake_proof)
        adversary_loss = F.binary_cross_entropy(adversary_scores, torch.ones_like(adversary_scores))
        adversary_loss.backward()
        
        if config.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(adversary.parameters(), config.gradient_clip_value)
        
        adversary_opt.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = (all_scores > 0.5).float()
            accuracy = (predictions == (all_targets > 0.5).float()).float().mean().item()
        
        return accuracy
    
    def _evolve_population(
        self, 
        population: List[NASConfig], 
        fitness_scores: List[float], 
        population_size: int
    ) -> List[NASConfig]:
        """Evolve population using selection, crossover, and mutation."""
        
        # Sort by fitness
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        
        # Elite selection (top 20%)
        elite_size = max(1, population_size // 5)
        elite = [population[i] for i in sorted_indices[:elite_size]]
        
        # Create next generation
        next_population = elite.copy()  # Keep elite
        
        while len(next_population) < population_size:
            if random.random() < 0.7 and len(elite) >= 2:  # Crossover
                parent1, parent2 = random.sample(elite, 2)
                child = self._crossover_configs(parent1, parent2)
            else:  # Mutation
                parent = random.choice(elite)
                child = self._mutate_config(parent)
            
            next_population.append(child)
        
        return next_population
    
    def _crossover_configs(self, config1: NASConfig, config2: NASConfig) -> NASConfig:
        """Crossover two configurations."""
        new_config = copy.deepcopy(config1)
        
        # Random crossover of attributes
        if random.random() < 0.5:
            new_config.hidden_layers = config2.hidden_layers
        if random.random() < 0.5:
            new_config.activation = config2.activation
        if random.random() < 0.5:
            new_config.learning_rate = config2.learning_rate
        if random.random() < 0.5:
            new_config.batch_size = config2.batch_size
        if random.random() < 0.5:
            new_config.optimizer = config2.optimizer
        
        # Ensure dropout rates match hidden layers
        dropout_rates = new_config.dropout_rates
        while len(dropout_rates) < len(new_config.hidden_layers):
            dropout_rates.append(dropout_rates[-1])
        new_config.dropout_rates = dropout_rates[:len(new_config.hidden_layers)]
        
        return new_config
    
    def _mutate_config(self, config: NASConfig) -> NASConfig:
        """Mutate a configuration."""
        new_config = copy.deepcopy(config)
        
        mutation_rate = 0.3
        
        if random.random() < mutation_rate:
            new_config.hidden_layers = random.choice(self.search_space['hidden_layers'])
        if random.random() < mutation_rate:
            new_config.activation = random.choice(self.search_space['activations'])
        if random.random() < mutation_rate:
            new_config.learning_rate = random.choice(self.search_space['learning_rates'])
        if random.random() < mutation_rate:
            new_config.batch_size = random.choice(self.search_space['batch_sizes'])
        if random.random() < mutation_rate:
            new_config.optimizer = random.choice(self.search_space['optimizers'])
        
        # Ensure dropout rates match hidden layers
        dropout_rates = random.choice(self.search_space['dropout_rates'])
        while len(dropout_rates) < len(new_config.hidden_layers):
            dropout_rates.append(dropout_rates[-1])
        new_config.dropout_rates = dropout_rates[:len(new_config.hidden_layers)]
        
        return new_config
    
    def _save_nas_config(self, op_info: OpCompilationInfo, config: NASConfig, accuracy: float):
        """Save the best NAS configuration."""
        config_path = op_info.compilation_log_path.replace('.log', '_best_config.json')
        
        config_data = {
            "best_config": config.to_dict(),
            "achieved_accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
            "operation": op_info.folder_name,
            "target_achieved": accuracy >= 0.99999
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"💾 Saved best configuration to {config_path}")
    
    def _generate_input_data(self, op_info: OpCompilationInfo, batch_size: int) -> torch.Tensor:
        """Generate random input data matching the operation's input shape."""
        shape = (batch_size,) + op_info.input_shape[1:]  # Skip batch dimension
        return torch.randn(shape, device=self.device)
    
    def _export_prover_onnx(self, prover: ONNXOperationWrapper, dummy_input: torch.Tensor, output_path: str):
        """Export prover to ONNX format."""
        prover.eval()
        
        # The prover now directly returns (output, proof) tuple
        torch.onnx.export(
            prover,
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
    
    def _plot_training_metrics(self, op_info: OpCompilationInfo, metrics: Dict[str, List], logger):
        """Generate comprehensive training plots for adversarial setup."""
        try:
            # Create plots directory in the operation folder
            from pathlib import Path
            op_folder = Path(op_info.compilation_log_path).parent
            plots_dir = op_folder / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Set up the plotting style
            plt.style.use('default')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Adversarial Training Metrics: {op_info.folder_name}', fontsize=16, fontweight='bold')
            
            epochs = range(1, len(metrics['verifier_loss']) + 1)
            
            # Plot 1: Loss curves
            ax1.plot(epochs, metrics['verifier_loss'], 'b-', label='Verifier Loss', linewidth=2)
            ax1.plot(epochs, metrics['adversary_loss'], 'r-', label='Adversary Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Verifier accuracy over time
            ax2.plot(epochs, [acc * 100 for acc in metrics['verifier_accuracy']], 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Verifier Accuracy')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 105)
            
            # Add horizontal line at 50% (random guessing)
            ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Random Baseline')
            ax2.legend()
            
            # Plot 3: Adversary fool rate
            ax3.plot(epochs, [rate * 100 for rate in metrics['adversary_fool_rate']], 'orange', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Fool Rate (%)')
            ax3.set_title('Adversary Success Rate (Fooling Verifier)')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 105)
            
            # Plot 4: Adversarial dynamics (accuracy vs fool rate)
            ax4.scatter([acc * 100 for acc in metrics['verifier_accuracy']], 
                       [rate * 100 for rate in metrics['adversary_fool_rate']], 
                       c=epochs, cmap='viridis', alpha=0.7, s=20)
            ax4.set_xlabel('Verifier Accuracy (%)')
            ax4.set_ylabel('Adversary Fool Rate (%)')
            ax4.set_title('Adversarial Dynamics (Color = Epoch)')
            ax4.grid(True, alpha=0.3)
            
            # Add colorbar for the scatter plot
            cbar = plt.colorbar(ax4.collections[0], ax=ax4)
            cbar.set_label('Epoch')
            
            # Add final performance annotations
            final_acc = metrics['verifier_accuracy'][-1] * 100
            final_fool = metrics['adversary_fool_rate'][-1] * 100
            
            # Annotate final performance
            ax2.annotate(f'Final: {final_acc:.1f}%', 
                        xy=(len(epochs), final_acc), xytext=(len(epochs)*0.7, final_acc + 10),
                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                        fontsize=10, fontweight='bold')
            
            ax3.annotate(f'Final: {final_fool:.1f}%', 
                        xy=(len(epochs), final_fool), xytext=(len(epochs)*0.7, final_fool + 10),
                        arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7),
                        fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = plots_dir / f"{op_info.folder_name}_training_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📈 Saved training plots to {plot_path}")
            
            # Generate a summary statistics plot
            self._plot_training_summary(op_info, metrics, plots_dir, logger)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate training plots: {e}")
    
    def _plot_training_summary(self, op_info: OpCompilationInfo, metrics: Dict[str, List], plots_dir, logger):
        """Generate a summary statistics plot."""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Calculate moving averages for smoother trends
            window_size = max(1, len(metrics['verifier_accuracy']) // 10)
            
            def moving_average(data, window):
                return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
            
            epochs = range(1, len(metrics['verifier_accuracy']) + 1)
            
            verifier_smooth = moving_average(metrics['verifier_accuracy'], window_size)
            adversary_smooth = moving_average(metrics['adversary_fool_rate'], window_size)
            
            # Plot smoothed curves
            ax.plot(epochs, [acc * 100 for acc in verifier_smooth], 'b-', linewidth=3, label='Verifier Accuracy (smoothed)')
            ax.plot(epochs, [100 - rate * 100 for rate in adversary_smooth], 'r-', linewidth=3, label='Verifier Robustness (smoothed)')
            
            # Add raw data as lighter lines
            ax.plot(epochs, [acc * 100 for acc in metrics['verifier_accuracy']], 'b-', alpha=0.3, linewidth=1)
            ax.plot(epochs, [100 - rate * 100 for rate in metrics['adversary_fool_rate']], 'r-', alpha=0.3, linewidth=1)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Training Summary: {op_info.folder_name}\n'
                        f'Operation: {op_info.op_type.value} | Input: {op_info.input_shape} → Output: {op_info.output_shape}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
            
            # Add performance statistics
            final_acc = metrics['verifier_accuracy'][-1] * 100
            final_robust = (1 - metrics['adversary_fool_rate'][-1]) * 100
            avg_acc = np.mean(metrics['verifier_accuracy']) * 100
            
            stats_text = f'Final Accuracy: {final_acc:.1f}%\nFinal Robustness: {final_robust:.1f}%\nAvg Accuracy: {avg_acc:.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save the summary plot
            summary_path = plots_dir / f"{op_info.folder_name}_training_summary.png"
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📈 Saved training summary to {summary_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate summary plot: {e}") 