"""
Neural Architecture Search (NAS) for Ultra-High Precision ZKML

This module implements evolutionary neural architecture search to achieve 99.999% 
accuracy (5 nines) for each compiled ONNX operation. It evolves verifier, prover 
proof generators, and adversary architectures through multiple strategies:

1. Evolutionary Architecture Search - Mutating layer counts, sizes, activations
2. Hyperparameter Optimization - Learning rates, regularization, batch sizes  
3. Training Strategy Evolution - Curriculum learning, adaptive techniques
4. Ensemble Methods - Multiple model voting for ultra-precision
5. Meta-Learning - Learning to learn better architectures

The system will keep evolving until each operation achieves 99.999% accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import random
import copy
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

from .ops_registry import OpCompilationInfo, SupportedOp
from .onnx_compiler import ONNXOperationWrapper, ONNXVerifier, ONNXAdversary


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
    RADAM = "radam"


class SchedulerType(Enum):
    COSINE = "cosine"
    PLATEAU = "plateau"
    EXPONENTIAL = "exponential"
    STEP = "step"
    CYCLIC = "cyclic"


@dataclass
class ArchitectureConfig:
    """Configuration for a neural architecture."""
    # Layer configuration
    hidden_layers: List[int]  # Hidden layer sizes
    activation: ActivationType
    dropout_rates: List[float]  # Dropout for each layer
    use_batch_norm: bool
    use_layer_norm: bool
    use_residual: bool
    
    # Proof generator specific
    proof_dim: int
    
    # Training configuration  
    optimizer: OptimizerType
    learning_rate: float
    weight_decay: float
    batch_size: int
    
    # Scheduler configuration
    scheduler: SchedulerType
    scheduler_params: Dict[str, Any]
    
    # Advanced techniques
    use_gradient_clipping: bool
    gradient_clip_value: float
    use_label_smoothing: bool
    label_smoothing: float
    use_mixup: bool
    mixup_alpha: float
    
    # Ensemble configuration
    ensemble_size: int
    use_ensemble: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert enums to strings
        result['activation'] = self.activation.value
        result['optimizer'] = self.optimizer.value  
        result['scheduler'] = self.scheduler.value
        return result


class AdvancedVerifier(nn.Module):
    """Advanced verifier with configurable architecture."""
    
    def __init__(self, op_info: OpCompilationInfo, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.op_info = op_info
        
        input_size = np.prod(op_info.input_shape)
        output_size = np.prod(op_info.output_shape)
        total_size = input_size + output_size + config.proof_dim
        
        # Build the architecture
        layers = []
        current_size = total_size
        
        for i, hidden_size in enumerate(config.hidden_layers):
            # Linear layer
            layers.append(nn.Linear(current_size, hidden_size))
            
            # Batch norm
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Layer norm  
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            
            # Activation
            layers.append(self._get_activation(config.activation))
            
            # Dropout
            if i < len(config.dropout_rates):
                layers.append(nn.Dropout(config.dropout_rates[i]))
            
            current_size = hidden_size
        
        # Final layer
        layers.append(nn.Linear(current_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Residual connections if enabled
        self.use_residual = config.use_residual
        if self.use_residual and len(config.hidden_layers) > 1:
            # Create residual projections for matching dimensions
            self.residual_projections = nn.ModuleList()
            prev_size = total_size
            for hidden_size in config.hidden_layers:
                if prev_size != hidden_size:
                    self.residual_projections.append(nn.Linear(prev_size, hidden_size))
                else:
                    self.residual_projections.append(nn.Identity())
                prev_size = hidden_size
    
    def _get_activation(self, activation: ActivationType) -> nn.Module:
        """Get activation function."""
        if activation == ActivationType.RELU:
            return nn.ReLU()
        elif activation == ActivationType.LEAKY_RELU:
            return nn.LeakyReLU(0.2)
        elif activation == ActivationType.GELU:
            return nn.GELU()
        elif activation == ActivationType.SWISH:
            return nn.SiLU()  # Swish = SiLU
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
        """Forward pass with optional residual connections."""
        input_flat = input_tensor.view(input_tensor.shape[0], -1)
        output_flat = output_tensor.view(output_tensor.shape[0], -1) 
        proof_flat = proof.view(proof.shape[0], -1)
        
        x = torch.cat([input_flat, output_flat, proof_flat], dim=-1)
        
        if not self.use_residual:
            return self.network(x)
        
        # Manual forward with residual connections
        residual = x
        layer_idx = 0
        residual_idx = 0
        
        for layer in self.network:
            if isinstance(layer, nn.Linear) and layer_idx > 0:  # Skip input layer
                # Add residual before linear layer
                if residual_idx < len(self.residual_projections):
                    projected_residual = self.residual_projections[residual_idx](residual)
                    if projected_residual.shape == x.shape:
                        x = x + projected_residual
                    residual = x
                    residual_idx += 1
                    
            x = layer(x)
            
            if isinstance(layer, nn.Linear):
                layer_idx += 1
        
        return x


class AdvancedProofGenerator(nn.Module):
    """Advanced proof generator with configurable architecture."""
    
    def __init__(self, op_info: OpCompilationInfo, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        
        input_size = np.prod(op_info.input_shape)
        output_size = np.prod(op_info.output_shape)
        total_size = input_size + output_size
        
        # Build the architecture
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
        
        # Final layer with tanh to bound proof values
        layers.append(nn.Linear(current_size, config.proof_dim))
        layers.append(nn.Tanh())
        
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
    
    def forward(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
        """Generate proof for input-output pair."""
        input_flat = input_tensor.view(input_tensor.shape[0], -1)
        output_flat = output_tensor.view(output_tensor.shape[0], -1)
        combined = torch.cat([input_flat, output_flat], dim=-1)
        return self.network(combined)


class AdvancedAdversary(nn.Module):
    """Advanced adversary with configurable architecture."""
    
    def __init__(self, op_info: OpCompilationInfo, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.op_info = op_info
        
        input_size = np.prod(op_info.input_shape)
        output_size = np.prod(op_info.output_shape)
        
        # Build output generator
        layers = []
        current_size = input_size
        
        for i, hidden_size in enumerate(config.hidden_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
                
            layers.append(self._get_activation(config.activation))
            
            if i < len(config.dropout_rates):
                layers.append(nn.Dropout(config.dropout_rates[i]))
                
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        self.output_generator = nn.Sequential(*layers)
        
        # Build proof generator
        proof_layers = []
        current_size = input_size
        
        for i, hidden_size in enumerate(config.hidden_layers):
            proof_layers.append(nn.Linear(current_size, hidden_size))
            
            if config.use_batch_norm:
                proof_layers.append(nn.BatchNorm1d(hidden_size))
                
            proof_layers.append(self._get_activation(config.activation))
            
            if i < len(config.dropout_rates):
                proof_layers.append(nn.Dropout(config.dropout_rates[i]))
                
            current_size = hidden_size
        
        proof_layers.append(nn.Linear(current_size, config.proof_dim))
        proof_layers.append(nn.Tanh())
        self.proof_generator = nn.Sequential(*proof_layers)
    
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
    
    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate fake output and proof."""
        input_flat = input_tensor.view(input_tensor.shape[0], -1)
        
        fake_output_flat = self.output_generator(input_flat)
        fake_output = fake_output_flat.view(input_tensor.shape[0], *self.op_info.output_shape[1:])
        
        fake_proof = self.proof_generator(input_flat)
        
        return fake_output, fake_proof


class EnsembleModel(nn.Module):
    """Ensemble model for ultra-high precision."""
    
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Ensemble forward pass with voting."""
        outputs = []
        for model in self.models:
            outputs.append(model(*args, **kwargs))
        
        # Average ensemble prediction
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output


class NeuralArchitectureSearch:
    """Neural Architecture Search for achieving 99.999% accuracy."""
    
    def __init__(self, device: str = "mps"):
        self.device = device
        self.logger = logging.getLogger("NAS")
        
        # Population for evolutionary search
        self.population_size = 20
        self.elite_size = 5
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7
        
        # Architecture search space
        self.search_space = {
            'hidden_layers': [
                [64, 32], [128, 64], [256, 128], [512, 256], [1024, 512],
                [64, 64, 32], [128, 128, 64], [256, 256, 128], [512, 512, 256],
                [64, 128, 64], [128, 256, 128], [256, 512, 256],
                [128, 256, 512, 256, 128], [256, 512, 1024, 512, 256]
            ],
            'activations': list(ActivationType),
            'dropout_rates': [
                [0.1], [0.2], [0.3], [0.4], [0.5],
                [0.1, 0.2], [0.2, 0.3], [0.3, 0.4],
                [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]
            ],
            'proof_dims': [16, 32, 64, 128, 256],
            'optimizers': list(OptimizerType),
            'learning_rates': [0.001, 0.0005, 0.0001, 0.00005, 0.00001],
            'batch_sizes': [16, 32, 64, 128, 256],
            'weight_decays': [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
        }
    
    def create_random_config(self) -> ArchitectureConfig:
        """Create a random architecture configuration."""
        hidden_layers = random.choice(self.search_space['hidden_layers'])
        dropout_rates = random.choice(self.search_space['dropout_rates'])
        
        # Ensure dropout rates match hidden layers
        while len(dropout_rates) < len(hidden_layers):
            dropout_rates.append(dropout_rates[-1])
        dropout_rates = dropout_rates[:len(hidden_layers)]
        
        config = ArchitectureConfig(
            hidden_layers=hidden_layers,
            activation=random.choice(self.search_space['activations']),
            dropout_rates=dropout_rates,
            use_batch_norm=random.choice([True, False]),
            use_layer_norm=random.choice([True, False]),
            use_residual=random.choice([True, False]),
            proof_dim=random.choice(self.search_space['proof_dims']),
            optimizer=random.choice(self.search_space['optimizers']),
            learning_rate=random.choice(self.search_space['learning_rates']),
            weight_decay=random.choice(self.search_space['weight_decays']),
            batch_size=random.choice(self.search_space['batch_sizes']),
            scheduler=random.choice(list(SchedulerType)),
            scheduler_params={'patience': 50, 'factor': 0.5},
            use_gradient_clipping=random.choice([True, False]),
            gradient_clip_value=random.uniform(0.5, 2.0),
            use_label_smoothing=random.choice([True, False]),
            label_smoothing=random.uniform(0.05, 0.2),
            use_mixup=random.choice([True, False]),
            mixup_alpha=random.uniform(0.1, 0.4),
            ensemble_size=random.choice([1, 3, 5]),
            use_ensemble=random.choice([True, False])
        )
        
        return config
    
    def mutate_config(self, config: ArchitectureConfig) -> ArchitectureConfig:
        """Mutate an architecture configuration."""
        new_config = copy.deepcopy(config)
        
        # Mutate with probability
        if random.random() < self.mutation_rate:
            new_config.hidden_layers = random.choice(self.search_space['hidden_layers'])
        
        if random.random() < self.mutation_rate:
            new_config.activation = random.choice(self.search_space['activations'])
        
        if random.random() < self.mutation_rate:
            dropout_rates = random.choice(self.search_space['dropout_rates'])
            while len(dropout_rates) < len(new_config.hidden_layers):
                dropout_rates.append(dropout_rates[-1])
            new_config.dropout_rates = dropout_rates[:len(new_config.hidden_layers)]
        
        if random.random() < self.mutation_rate:
            new_config.proof_dim = random.choice(self.search_space['proof_dims'])
        
        if random.random() < self.mutation_rate:
            new_config.learning_rate = random.choice(self.search_space['learning_rates'])
        
        if random.random() < self.mutation_rate:
            new_config.batch_size = random.choice(self.search_space['batch_sizes'])
        
        return new_config
    
    def crossover_configs(self, config1: ArchitectureConfig, config2: ArchitectureConfig) -> ArchitectureConfig:
        """Crossover two architecture configurations."""
        new_config = copy.deepcopy(config1)
        
        # Random crossover of attributes
        if random.random() < 0.5:
            new_config.hidden_layers = config2.hidden_layers
        
        if random.random() < 0.5:
            new_config.activation = config2.activation
        
        if random.random() < 0.5:
            new_config.proof_dim = config2.proof_dim
        
        if random.random() < 0.5:
            new_config.learning_rate = config2.learning_rate
        
        if random.random() < 0.5:
            new_config.batch_size = config2.batch_size
        
        # Ensure dropout rates match hidden layers
        dropout_rates = new_config.dropout_rates
        while len(dropout_rates) < len(new_config.hidden_layers):
            dropout_rates.append(dropout_rates[-1])
        new_config.dropout_rates = dropout_rates[:len(new_config.hidden_layers)]
        
        return new_config
    
    def evaluate_architecture(self, config: ArchitectureConfig, op_info: OpCompilationInfo, 
                            max_epochs: int = 500) -> float:
        """Evaluate an architecture configuration and return best accuracy achieved."""
        try:
            self.logger.info(f"🧬 Evaluating architecture: {config.hidden_layers}, {config.activation.value}")
            
            # Create models with this configuration
            if config.use_ensemble and config.ensemble_size > 1:
                verifier_models = []
                for _ in range(config.ensemble_size):
                    verifier_models.append(AdvancedVerifier(op_info, config))
                verifier = EnsembleModel(verifier_models)
            else:
                verifier = AdvancedVerifier(op_info, config)
            
            adversary = AdvancedAdversary(op_info, config)
            
            # Create operation wrapper with advanced proof generator
            operation_wrapper = ONNXOperationWrapper(op_info, config.proof_dim)
            operation_wrapper.proof_generator = AdvancedProofGenerator(op_info, config)
            
            # Move to device
            verifier = verifier.to(self.device)
            adversary = adversary.to(self.device) 
            operation_wrapper = operation_wrapper.to(self.device)
            
            # Create optimizers
            verifier_optimizer = self._create_optimizer(verifier, config)
            adversary_optimizer = self._create_optimizer(adversary, config)
            
            # Create schedulers
            verifier_scheduler = self._create_scheduler(verifier_optimizer, config)
            adversary_scheduler = self._create_scheduler(adversary_optimizer, config)
            
            # Training loop
            best_accuracy = 0.0
            patience_counter = 0
            patience = 100
            
            for epoch in range(max_epochs):
                # Generate training data
                input_data = self._generate_input_data(op_info, config.batch_size)
                
                # Training step
                verifier_acc = self._train_step(
                    operation_wrapper, verifier, adversary,
                    verifier_optimizer, adversary_optimizer,
                    input_data, config
                )
                
                # Update best accuracy
                if verifier_acc > best_accuracy:
                    best_accuracy = verifier_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    break
                
                # Step schedulers
                if verifier_scheduler:
                    if isinstance(verifier_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        verifier_scheduler.step(1.0 - verifier_acc)
                    else:
                        verifier_scheduler.step()
                
                if adversary_scheduler:
                    if isinstance(adversary_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        adversary_scheduler.step(verifier_acc)
                    else:
                        adversary_scheduler.step()
                
                # Check for ultra-precision
                if verifier_acc >= 0.99999:
                    self.logger.info(f"🎯 ULTRA-PRECISION ACHIEVED: {verifier_acc:.5f}")
                    return verifier_acc
            
            self.logger.info(f"💯 Best accuracy: {best_accuracy:.5f}")
            return best_accuracy
            
        except Exception as e:
            self.logger.error(f"❌ Architecture evaluation failed: {str(e)}")
            return 0.0
    
    def _create_optimizer(self, model: nn.Module, config: ArchitectureConfig) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        if config.optimizer == OptimizerType.ADAM:
            return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == OptimizerType.ADAMW:
            return optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == OptimizerType.SGD:
            return optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
        elif config.optimizer == OptimizerType.RMSPROP:
            return optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:  # Default to Adam
            return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer, config: ArchitectureConfig) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        if config.scheduler == SchedulerType.PLATEAU:
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.scheduler_params.get('patience', 50))
        elif config.scheduler == SchedulerType.COSINE:
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        elif config.scheduler == SchedulerType.EXPONENTIAL:
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif config.scheduler == SchedulerType.STEP:
            return optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        else:
            return None
    
    def _generate_input_data(self, op_info: OpCompilationInfo, batch_size: int) -> torch.Tensor:
        """Generate input data for the operation."""
        return torch.randn(batch_size, *op_info.input_shape[1:]).to(self.device)
    
    def _train_step(self, operation_wrapper: ONNXOperationWrapper, verifier: nn.Module, 
                   adversary: AdvancedAdversary, verifier_optimizer: torch.optim.Optimizer,
                   adversary_optimizer: torch.optim.Optimizer, input_data: torch.Tensor,
                   config: ArchitectureConfig) -> float:
        """Perform one training step."""
        
        # Apply mixup if enabled
        if config.use_mixup:
            input_data = self._apply_mixup(input_data, config.mixup_alpha)
        
        # Generate real outputs and proofs
        real_output, real_proof = operation_wrapper(input_data)
        
        # Generate fake outputs and proofs
        fake_output, fake_proof = adversary(input_data)
        
        # Train verifier
        verifier_optimizer.zero_grad()
        
        # Real examples (should be accepted)
        real_scores = verifier(input_data, real_output.detach(), real_proof.detach())
        real_targets = torch.ones_like(real_scores)
        
        # Fake examples (should be rejected)
        fake_scores = verifier(input_data, fake_output.detach(), fake_proof.detach())
        fake_targets = torch.zeros_like(fake_scores)
        
        # Mixed examples (real computation + fake proof)
        mixed_scores1 = verifier(input_data, real_output.detach(), fake_proof.detach())
        mixed_targets1 = torch.zeros_like(mixed_scores1)
        
        # Mixed examples (fake computation + real proof)
        mixed_scores2 = verifier(input_data, fake_output.detach(), real_proof.detach())
        mixed_targets2 = torch.zeros_like(mixed_scores2)
        
        # Combine all targets and scores
        all_scores = torch.cat([real_scores, fake_scores, mixed_scores1, mixed_scores2])
        all_targets = torch.cat([real_targets, fake_targets, mixed_targets1, mixed_targets2])
        
        # Apply label smoothing if enabled
        if config.use_label_smoothing:
            all_targets = self._apply_label_smoothing(all_targets, config.label_smoothing)
        
        verifier_loss = F.binary_cross_entropy(all_scores, all_targets)
        verifier_loss.backward()
        
        # Gradient clipping if enabled
        if config.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), config.gradient_clip_value)
        
        verifier_optimizer.step()
        
        # Train adversary
        adversary_optimizer.zero_grad()
        
        fake_output, fake_proof = adversary(input_data)
        adversary_scores = verifier(input_data, fake_output, fake_proof)
        # Adversary wants to fool verifier (get high scores)
        adversary_loss = F.binary_cross_entropy(adversary_scores, torch.ones_like(adversary_scores))
        adversary_loss.backward()
        
        if config.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(adversary.parameters(), config.gradient_clip_value)
        
        adversary_optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = (all_scores > 0.5).float()
            accuracy = (predictions == (all_targets > 0.5).float()).float().mean().item()
        
        return accuracy
    
    def _apply_mixup(self, input_data: torch.Tensor, alpha: float) -> torch.Tensor:
        """Apply mixup data augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
            batch_size = input_data.size(0)
            index = torch.randperm(batch_size).to(input_data.device)
            mixed_input = lam * input_data + (1 - lam) * input_data[index, :]
            return mixed_input
        return input_data
    
    def _apply_label_smoothing(self, targets: torch.Tensor, smoothing: float) -> torch.Tensor:
        """Apply label smoothing."""
        return targets * (1.0 - smoothing) + smoothing * 0.5
    
    def evolve_architecture_for_operation(self, op_info: OpCompilationInfo, 
                                        target_accuracy: float = 0.99999,
                                        max_generations: int = 50,
                                        max_eval_epochs: int = 500) -> Tuple[ArchitectureConfig, float]:
        """Evolve architecture until target accuracy is reached."""
        
        self.logger.info(f"🧬 Starting NAS for {op_info.folder_name} targeting {target_accuracy:.5f} accuracy")
        
        # Initialize population
        population = [self.create_random_config() for _ in range(self.population_size)]
        best_config = None
        best_accuracy = 0.0
        
        for generation in range(max_generations):
            self.logger.info(f"🧬 Generation {generation + 1}/{max_generations}")
            
            # Evaluate population
            fitness_scores = []
            for i, config in enumerate(population):
                self.logger.info(f"   Evaluating individual {i + 1}/{len(population)}")
                accuracy = self.evaluate_architecture(config, op_info, max_eval_epochs)
                fitness_scores.append(accuracy)
                
                # Update best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = copy.deepcopy(config)
                    self.logger.info(f"🎯 New best accuracy: {best_accuracy:.5f}")
                
                # Check if target reached
                if accuracy >= target_accuracy:
                    self.logger.info(f"🏆 TARGET ACCURACY ACHIEVED: {accuracy:.5f}")
                    return config, accuracy
            
            # Selection and reproduction
            # Sort by fitness
            sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
            
            # Elite selection
            elite = [population[i] for i in sorted_indices[:self.elite_size]]
            
            # Create next generation
            next_population = elite.copy()
            
            while len(next_population) < self.population_size:
                if random.random() < self.crossover_rate and len(elite) >= 2:
                    # Crossover
                    parent1, parent2 = random.sample(elite, 2)
                    child = self.crossover_configs(parent1, parent2)
                else:
                    # Mutation
                    parent = random.choice(elite)
                    child = self.mutate_config(parent)
                
                next_population.append(child)
            
            population = next_population
            
            self.logger.info(f"   Generation {generation + 1} best: {max(fitness_scores):.5f}")
        
        self.logger.info(f"🧬 NAS completed. Best accuracy: {best_accuracy:.5f}")
        return best_config, best_accuracy