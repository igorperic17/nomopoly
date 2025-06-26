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
import matplotlib.pyplot as plt

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
        
        # Training loop with target accuracy
        logger.info(f"Training until {target_accuracy:.1%} accuracy (min {num_epochs}, max {max_epochs} epochs)...")
        
        best_accuracy = 0.0
        epochs_without_improvement = 0
        patience = 50  # Early stopping patience
        
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
                'Acc': f"{verifier_acc:.3f}",
                'Target': f"{target_accuracy:.3f}",
                'Best': f"{best_accuracy:.3f}"
            })
            pbar.update(1)
            
            # Check for improvement
            if verifier_acc > best_accuracy:
                best_accuracy = verifier_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Log progress
            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch {epoch + 1}: Verifier Acc: {verifier_acc:.3f}, "
                          f"Adversary Fool Rate: {fool_rate:.3f}, Best: {best_accuracy:.3f}")
            
            # Check termination conditions
            epoch += 1
            
            # Target accuracy reached and minimum epochs completed
            if verifier_acc >= target_accuracy and epoch >= num_epochs:
                logger.info(f"🎯 Target accuracy {target_accuracy:.1%} reached at epoch {epoch}!")
                break
                
            # Early stopping if no improvement
            if epochs_without_improvement >= patience and epoch >= num_epochs:
                logger.info(f"⏹️  Early stopping: No improvement for {patience} epochs")
                break
        
        pbar.close()
        
        final_accuracy = metrics['verifier_accuracy'][-1]
        if final_accuracy >= target_accuracy:
            logger.info(f"✅ Training completed! Target accuracy achieved: {final_accuracy:.3f}")
        else:
            logger.info(f"⚠️  Training completed at max epochs. Final accuracy: {final_accuracy:.3f} (target: {target_accuracy:.3f})")
        
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