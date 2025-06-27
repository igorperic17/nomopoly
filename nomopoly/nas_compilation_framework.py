"""
NAS-Enhanced ONNX Compilation Framework

This framework combines the ONNX compilation system with Neural Architecture Search (NAS)
to achieve ultra-high precision (99.999% accuracy) for each compiled operation.

The system evolves architectures through multiple strategies:
1. Evolutionary search across architecture configurations
2. Hyperparameter optimization for training strategies  
3. Ensemble methods for ultra-precision
4. Advanced training techniques (mixup, label smoothing, etc.)

It will keep training and evolving until 99.999% accuracy is achieved for each operation.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from .ops_registry import ops_registry, OpCompilationInfo, SupportedOp
from .neural_architecture_search import NeuralArchitectureSearch, ArchitectureConfig
from .onnx_compiler import ONNXOperationCompiler


class NASCompilationFramework:
    """
    NAS-Enhanced compilation framework for achieving 99.999% accuracy.
    
    This framework orchestrates the Neural Architecture Search process:
    1. Scan ONNX models to discover operations
    2. For each operation, evolve architectures until 99.999% accuracy
    3. Export the best-performing architectures as ONNX models
    4. Comprehensive logging and metrics tracking
    """
    
    def __init__(
        self,
        ops_dir: str = "ops",
        device: str = "mps",
        target_accuracy: float = 0.99999,
        max_generations: int = 50,
        max_eval_epochs: int = 500
    ):
        self.ops_dir = Path(ops_dir)
        self.device = device
        self.target_accuracy = target_accuracy
        self.max_generations = max_generations
        self.max_eval_epochs = max_eval_epochs
        
        # Create operations directory
        self.ops_dir.mkdir(exist_ok=True)
        
        # Initialize NAS and compiler
        self.nas = NeuralArchitectureSearch(device=device)
        self.compiler = NASEnabledCompiler(device=device)
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up comprehensive logging."""
        log_file = self.ops_dir / "nas_compilation.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("NASCompilationFramework")
        
    def evolve_all_operations_to_precision(
        self,
        force_recompile: bool = False
    ) -> Dict[str, Dict]:
        """
        Evolve all operations until they reach 99.999% accuracy.
        
        Args:
            force_recompile: If True, recompile even already compiled operations
            
        Returns:
            Dictionary mapping operation names to evolution results
        """
        
        self.logger.info(f"🧬🎯 STARTING ULTRA-PRECISION EVOLUTION TO {self.target_accuracy:.5f} ACCURACY")
        self.logger.info(f"🔧 Targeting 5 NINES (99.999%) for all operations")
        
        if force_recompile:
            uncompiled_ops = list(ops_registry.discovered_ops.values())
            self.logger.info(f"🔄 Force evolving {len(uncompiled_ops)} operations")
        else:
            uncompiled_ops = ops_registry.get_uncompiled_operations()
            self.logger.info(f"🔧 Found {len(uncompiled_ops)} operations to evolve")
        
        if not uncompiled_ops:
            self.logger.info("✅ All operations already evolved to target precision!")
            return {}
        
        evolution_results = {}
        total_start_time = time.time()
        successful_evolutions = 0
        
        for i, op_info in enumerate(uncompiled_ops):
            self.logger.info(f"\n🧬 EVOLVING OPERATION {i+1}/{len(uncompiled_ops)}: {op_info.folder_name}")
            self.logger.info(f"🎯 Target: {self.target_accuracy:.5f} accuracy (5 nines)")
            
            start_time = time.time()
            
            try:
                # Run NAS evolution for this operation
                best_config, best_accuracy = self.nas.evolve_architecture_for_operation(
                    op_info=op_info,
                    target_accuracy=self.target_accuracy,
                    max_generations=self.max_generations,
                    max_eval_epochs=self.max_eval_epochs
                )
                
                evolution_time = time.time() - start_time
                
                if best_accuracy >= self.target_accuracy:
                    self.logger.info(f"🏆 ULTRA-PRECISION ACHIEVED: {best_accuracy:.5f}")
                    successful_evolutions += 1
                    
                    # Compile the best architecture to ONNX
                    compilation_result = self._compile_best_architecture(
                        op_info, best_config, best_accuracy
                    )
                    
                    evolution_results[op_info.folder_name] = {
                        "success": True,
                        "target_achieved": True,
                        "final_accuracy": best_accuracy,
                        "best_config": best_config.to_dict(),
                        "evolution_time": evolution_time,
                        "compilation_result": compilation_result
                    }
                    
                else:
                    self.logger.warning(f"⚠️ Did not reach target. Best: {best_accuracy:.5f}")
                    
                    evolution_results[op_info.folder_name] = {
                        "success": True,
                        "target_achieved": False,
                        "final_accuracy": best_accuracy,
                        "best_config": best_config.to_dict() if best_config else None,
                        "evolution_time": evolution_time
                    }
                
            except Exception as e:
                evolution_time = time.time() - start_time
                error_msg = f"Exception during evolution: {str(e)}"
                self.logger.error(f"❌ {op_info.folder_name}: {error_msg}")
                
                evolution_results[op_info.folder_name] = {
                    "success": False,
                    "target_achieved": False,
                    "error": error_msg,
                    "evolution_time": evolution_time
                }
        
        total_time = time.time() - total_start_time
        
        # Comprehensive summary
        self._log_evolution_summary(evolution_results, total_time, successful_evolutions)
        
        return evolution_results
    
    def _compile_best_architecture(
        self, 
        op_info: OpCompilationInfo, 
        config: ArchitectureConfig, 
        accuracy: float
    ) -> Dict:
        """Compile the best evolved architecture to ONNX."""
        
        self.logger.info(f"📦 Compiling best architecture for {op_info.folder_name}")
        
        try:
            # Use the enhanced compiler with NAS-evolved configuration
            result = self.compiler.compile_operation_with_config(
                op_info=op_info,
                config=config,
                target_accuracy=accuracy,
                max_epochs=1000  # More epochs for final compilation
            )
            
            # Save the architecture configuration
            config_path = self.ops_dir / op_info.folder_name / "best_architecture.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump({
                    "config": config.to_dict(),
                    "achieved_accuracy": accuracy,
                    "compilation_timestamp": time.time()
                }, f, indent=2)
            
            self.logger.info(f"✅ Architecture compiled and saved for {op_info.folder_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to compile architecture: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _log_evolution_summary(
        self, 
        evolution_results: Dict[str, Dict], 
        total_time: float, 
        successful_evolutions: int
    ):
        """Log comprehensive evolution summary."""
        
        total_operations = len(evolution_results)
        failed_evolutions = [r for r in evolution_results.values() if not r.get("success", False)]
        target_achieved = [r for r in evolution_results.values() if r.get("target_achieved", False)]
        
        self.logger.info(f"\n🧬🎯 ULTRA-PRECISION EVOLUTION SUMMARY:")
        self.logger.info(f"   🎯 Target Accuracy: {self.target_accuracy:.5f} (5 nines)")
        self.logger.info(f"   📊 Total Operations: {total_operations}")
        self.logger.info(f"   🏆 Target Achieved: {len(target_achieved)}")
        self.logger.info(f"   ✅ Successful Evolution: {successful_evolutions}")
        self.logger.info(f"   ❌ Failed Evolution: {len(failed_evolutions)}")
        self.logger.info(f"   ⏱️  Total Evolution Time: {total_time:.1f}s")
        
        if target_achieved:
            self.logger.info(f"\n🏆 OPERATIONS ACHIEVING ULTRA-PRECISION:")
            for op_name, result in evolution_results.items():
                if result.get("target_achieved", False):
                    accuracy = result["final_accuracy"]
                    time_taken = result["evolution_time"]
                    self.logger.info(f"   🎯 {op_name}: {accuracy:.5f} accuracy in {time_taken:.1f}s")
        
        if successful_evolutions > 0:
            # Calculate statistics for successful evolutions
            successful_results = [r for r in evolution_results.values() if r.get("success", False)]
            avg_accuracy = sum(r["final_accuracy"] for r in successful_results) / len(successful_results)
            avg_time = sum(r["evolution_time"] for r in successful_results) / len(successful_results)
            
            self.logger.info(f"\n📈 EVOLUTION STATISTICS:")
            self.logger.info(f"   📊 Average Accuracy: {avg_accuracy:.5f}")
            self.logger.info(f"   ⏱️  Average Evolution Time: {avg_time:.1f}s")
            self.logger.info(f"   🎯 Ultra-Precision Rate: {len(target_achieved)}/{total_operations} ({len(target_achieved)/total_operations*100:.1f}%)")
        
        if failed_evolutions:
            self.logger.warning(f"\n❌ FAILED EVOLUTIONS:")
            for op_name, result in evolution_results.items():
                if not result.get("success", False):
                    error = result.get("error", "Unknown error")
                    self.logger.warning(f"   ❌ {op_name}: {error}")
    
    def scan_and_evolve_model(
        self,
        onnx_model_path: str,
        force_recompile: bool = False
    ) -> Dict[str, Dict]:
        """
        Complete workflow: scan a model and evolve all its operations to 99.999% accuracy.
        
        Args:
            onnx_model_path: Path to ONNX model to process
            force_recompile: If True, recompile even already compiled operations
            
        Returns:
            Dictionary mapping operation names to evolution results
        """
        
        self.logger.info(f"🔍 Scanning and evolving ONNX model: {onnx_model_path}")
        
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        # Scan the model first
        discovered_ops = ops_registry.scan_onnx_model(onnx_model_path)
        self.logger.info(f"✅ Discovered {len(discovered_ops)} operations")
        
        # Evolve all operations
        return self.evolve_all_operations_to_precision(force_recompile=force_recompile)
    
    def validate_ultra_precision(self) -> Dict[str, bool]:
        """
        Validate that all compiled operations achieve ultra-precision.
        
        Returns:
            Dictionary mapping operation names to validation status
        """
        
        self.logger.info("🔍 Validating ultra-precision for all operations...")
        
        validation_results = {}
        
        for op_info in ops_registry.discovered_ops.values():
            config_path = self.ops_dir / op_info.folder_name / "best_architecture.json"
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        data = json.load(f)
                    
                    achieved_accuracy = data.get("achieved_accuracy", 0.0)
                    is_ultra_precise = achieved_accuracy >= self.target_accuracy
                    
                    validation_results[op_info.folder_name] = is_ultra_precise
                    
                    if is_ultra_precise:
                        self.logger.info(f"✅ {op_info.folder_name}: {achieved_accuracy:.5f} - ULTRA-PRECISE")
                    else:
                        self.logger.warning(f"⚠️ {op_info.folder_name}: {achieved_accuracy:.5f} - BELOW TARGET")
                        
                except Exception as e:
                    self.logger.error(f"❌ {op_info.folder_name}: Validation failed - {str(e)}")
                    validation_results[op_info.folder_name] = False
            else:
                self.logger.warning(f"⚠️ {op_info.folder_name}: No architecture config found")
                validation_results[op_info.folder_name] = False
        
        # Summary
        total_ops = len(validation_results)
        ultra_precise_ops = sum(validation_results.values())
        precision_rate = ultra_precise_ops / total_ops * 100 if total_ops > 0 else 0
        
        self.logger.info(f"\n🎯 ULTRA-PRECISION VALIDATION SUMMARY:")
        self.logger.info(f"   📊 Total Operations: {total_ops}")
        self.logger.info(f"   🏆 Ultra-Precise: {ultra_precise_ops}")
        self.logger.info(f"   📈 Precision Rate: {precision_rate:.1f}%")
        
        return validation_results
    
    def continue_evolution_until_target(
        self,
        max_attempts: int = 10
    ) -> Dict[str, Dict]:
        """
        Continue evolution attempts until all operations reach target accuracy.
        
        Args:
            max_attempts: Maximum number of evolution attempts per operation
            
        Returns:
            Final evolution results
        """
        
        self.logger.info(f"🔄 CONTINUOUS EVOLUTION UNTIL {self.target_accuracy:.5f} ACCURACY")
        self.logger.info(f"🎯 Maximum {max_attempts} attempts per operation")
        
        final_results = {}
        
        for attempt in range(max_attempts):
            self.logger.info(f"\n🔄 EVOLUTION ATTEMPT {attempt + 1}/{max_attempts}")
            
            # Check which operations still need evolution
            validation_results = self.validate_ultra_precision()
            operations_needing_evolution = [
                op_name for op_name, is_precise in validation_results.items() 
                if not is_precise
            ]
            
            if not operations_needing_evolution:
                self.logger.info("🏆 ALL OPERATIONS HAVE ACHIEVED ULTRA-PRECISION!")
                break
            
            self.logger.info(f"🔧 {len(operations_needing_evolution)} operations need evolution:")
            for op_name in operations_needing_evolution:
                self.logger.info(f"   🎯 {op_name}")
            
            # Run evolution on operations that need it
            attempt_results = self.evolve_all_operations_to_precision(force_recompile=True)
            
            # Update final results
            final_results.update(attempt_results)
            
            # Check if we've achieved the target
            post_validation = self.validate_ultra_precision()
            remaining_ops = [
                op_name for op_name, is_precise in post_validation.items() 
                if not is_precise
            ]
            
            if not remaining_ops:
                self.logger.info("🏆 ALL OPERATIONS ACHIEVED ULTRA-PRECISION!")
                break
            else:
                self.logger.info(f"🔄 {len(remaining_ops)} operations still need evolution")
        
        return final_results


# Enhanced compiler with NAS integration
class NASEnabledCompiler(ONNXOperationCompiler):
    """ONNX compiler enhanced with NAS capabilities."""
    
    def __init__(self, device: str = "mps"):
        """Initialize the NAS-enabled compiler."""
        super().__init__(device=device)
        # Set up logger for NAS compilation
        self.logger = logging.getLogger("NASEnabledCompiler")
    
    def compile_operation_with_config(
        self,
        op_info: OpCompilationInfo,
        config: ArchitectureConfig,
        target_accuracy: float = 0.99999,
        max_epochs: int = 1000
    ) -> Dict[str, Any]:
        """Compile operation using NAS-evolved configuration."""
        
        from .neural_architecture_search import AdvancedVerifier, AdvancedAdversary, AdvancedProofGenerator
        
        self.logger.info(f"🧬 Compiling {op_info.folder_name} with evolved architecture")
        
        try:
            # Create models with NAS configuration
            if config.use_ensemble and config.ensemble_size > 1:
                from .neural_architecture_search import EnsembleModel
                verifier_models = []
                for _ in range(config.ensemble_size):
                    verifier_models.append(AdvancedVerifier(op_info, config))
                verifier = EnsembleModel(verifier_models)
            else:
                verifier = AdvancedVerifier(op_info, config)
            
            adversary = AdvancedAdversary(op_info, config)
            
            # Create operation wrapper with advanced proof generator
            operation_wrapper = self._create_operation_wrapper(op_info, config.proof_dim)
            operation_wrapper.proof_generator = AdvancedProofGenerator(op_info, config)
            
            # Move to device
            verifier = verifier.to(self.device)
            adversary = adversary.to(self.device)
            operation_wrapper = operation_wrapper.to(self.device)
            
            # Train with the evolved configuration
            result = self._train_with_config(
                operation_wrapper, verifier, adversary, 
                op_info, config, target_accuracy, max_epochs
            )
            
            if result["success"]:
                # Export the trained models
                self._export_nas_models(operation_wrapper, verifier, adversary, op_info, config)
                
                # Generate training plots if metrics are available
                if "metrics" in result:
                    self._plot_training_metrics(op_info, result["metrics"])
                
                # Generate comprehensive benchmarks
                self._generate_benchmarks(operation_wrapper, verifier, adversary, op_info, result)
                
                # Save compilation metadata
                self._save_compilation_metadata(op_info, config, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ NAS compilation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_operation_wrapper(self, op_info: OpCompilationInfo, proof_dim: int):
        """Create operation wrapper."""
        from .onnx_compiler import ONNXOperationWrapper
        return ONNXOperationWrapper(op_info, proof_dim)
    
    def _train_with_config(
        self, operation_wrapper, verifier, adversary, 
        op_info: OpCompilationInfo, config: ArchitectureConfig,
        target_accuracy: float, max_epochs: int
    ) -> Dict[str, Any]:
        """Train models with NAS configuration using full adversarial training."""
        
        import torch
        import torch.optim as optim
        import torch.nn.functional as F
        from tqdm import tqdm
        import time
        
        self.logger.info(f"🔧 Starting full training with NAS config: {config.hidden_layers}, {config.activation.value}")
        self.logger.info(f"🎯 Target accuracy: {target_accuracy:.5f}")
        self.logger.info(f"📊 Max epochs: {max_epochs}")
        
        # Create optimizers based on NAS config
        if config.optimizer.value == "adam":
            verifier_optimizer = optim.Adam(verifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            adversary_optimizer = optim.Adam(adversary.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            prover_optimizer = optim.Adam(operation_wrapper.proof_generator.parameters(), lr=config.learning_rate * 0.5, weight_decay=config.weight_decay)
        elif config.optimizer.value == "adamw":
            verifier_optimizer = optim.AdamW(verifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            adversary_optimizer = optim.AdamW(adversary.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            prover_optimizer = optim.AdamW(operation_wrapper.proof_generator.parameters(), lr=config.learning_rate * 0.5, weight_decay=config.weight_decay)
        else:
            verifier_optimizer = optim.Adam(verifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            adversary_optimizer = optim.Adam(adversary.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            prover_optimizer = optim.Adam(operation_wrapper.proof_generator.parameters(), lr=config.learning_rate * 0.5, weight_decay=config.weight_decay)
        
        # Learning rate schedulers
        verifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(verifier_optimizer, mode='max', factor=0.8, patience=50, min_lr=1e-6)
        adversary_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adversary_optimizer, mode='min', factor=0.8, patience=50, min_lr=1e-6)
        
        # Training metrics
        metrics = {
            "verifier_loss": [],
            "adversary_loss": [],
            "prover_loss": [],
            "verifier_accuracy": [],
            "adversary_fool_rate": [],
            "real_accept_rate": [],
            "fake_reject_rate": [],
            "learning_rates": {"verifier": [], "adversary": []}
        }
        
        best_accuracy = 0.0
        patience_counter = 0
        patience = 150 if target_accuracy >= 0.999 else 100
        
        # Progress bar
        pbar = tqdm(range(max_epochs), desc=f"Training {op_info.folder_name}")
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            
            # Generate training data
            input_data = torch.randn(config.batch_size, *op_info.input_shape[1:]).to(self.device)
            
            # === STEP 1: Generate real examples ===
            with torch.no_grad():
                real_output, real_proof = operation_wrapper(input_data)
            
            # === STEP 2: Generate fake examples ===
            fake_output, fake_proof = adversary(input_data)
            
            # === STEP 3: Train Verifier ===
            verifier_optimizer.zero_grad()
            
            # Real examples (should be accepted)
            real_scores = verifier(input_data, real_output.detach(), real_proof.detach())
            real_loss = F.binary_cross_entropy(real_scores, torch.ones_like(real_scores))
            
            # Fake examples (should be rejected)
            fake_scores = verifier(input_data, fake_output.detach(), fake_proof.detach())
            fake_loss = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))
            
            # Mixed examples for robustness
            mixed_scores1 = verifier(input_data, real_output.detach(), fake_proof.detach())
            mixed_loss1 = F.binary_cross_entropy(mixed_scores1, torch.zeros_like(mixed_scores1))
            
            mixed_scores2 = verifier(input_data, fake_output.detach(), real_proof.detach())
            mixed_loss2 = F.binary_cross_entropy(mixed_scores2, torch.zeros_like(mixed_scores2))
            
            verifier_loss = real_loss + fake_loss + 0.5 * (mixed_loss1 + mixed_loss2)
            verifier_loss.backward()
            
            # Gradient clipping if enabled
            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(verifier.parameters(), config.gradient_clip_value)
            
            verifier_optimizer.step()
            
            # === STEP 4: Train Adversary ===
            adversary_optimizer.zero_grad()
            
            adv_fake_output, adv_fake_proof = adversary(input_data)
            adv_scores = verifier(input_data, adv_fake_output, adv_fake_proof)
            adversary_loss = F.binary_cross_entropy(adv_scores, torch.ones_like(adv_scores))
            adversary_loss.backward()
            
            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(adversary.parameters(), config.gradient_clip_value)
            
            adversary_optimizer.step()
            
            # === STEP 5: Train Prover Proof Generator ===
            prover_optimizer.zero_grad()
            
            fresh_real_output, fresh_real_proof = operation_wrapper(input_data)
            prover_scores = verifier(input_data, fresh_real_output.detach(), fresh_real_proof)
            prover_loss = F.binary_cross_entropy(prover_scores, torch.ones_like(prover_scores))
            prover_loss.backward()
            
            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(operation_wrapper.proof_generator.parameters(), config.gradient_clip_value)
            
            prover_optimizer.step()
            
            # === STEP 6: Calculate metrics ===
            with torch.no_grad():
                # Recalculate for clean metrics
                eval_real_scores = verifier(input_data, real_output, real_proof)
                eval_fake_scores = verifier(input_data, fake_output, fake_proof)
                
                real_accept_rate = (eval_real_scores > 0.5).float().mean().item()
                fake_reject_rate = (eval_fake_scores < 0.5).float().mean().item()
                verifier_accuracy = (real_accept_rate + fake_reject_rate) / 2
                fool_rate = (adv_scores > 0.5).float().mean().item()
                
                # Store metrics
                metrics["verifier_loss"].append(verifier_loss.item())
                metrics["adversary_loss"].append(adversary_loss.item())
                metrics["prover_loss"].append(prover_loss.item())
                metrics["verifier_accuracy"].append(verifier_accuracy)
                metrics["adversary_fool_rate"].append(fool_rate)
                metrics["real_accept_rate"].append(real_accept_rate)
                metrics["fake_reject_rate"].append(fake_reject_rate)
                metrics["learning_rates"]["verifier"].append(verifier_optimizer.param_groups[0]['lr'])
                metrics["learning_rates"]["adversary"].append(adversary_optimizer.param_groups[0]['lr'])
            
            # Update progress bar
            pbar.set_postfix({
                'Acc': f"{verifier_accuracy:.4f}",
                'Target': f"{target_accuracy:.4f}",
                'Best': f"{best_accuracy:.4f}",
                'Real': f"{real_accept_rate:.3f}",
                'Fake': f"{fake_reject_rate:.3f}"
            })
            pbar.update(1)
            
            # Check for improvement
            if verifier_accuracy > best_accuracy:
                best_accuracy = verifier_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Update learning rate schedulers
            verifier_scheduler.step(verifier_accuracy)
            adversary_scheduler.step(adversary_loss.item())
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"⏹️ Early stopping after {patience} epochs without improvement")
                break
            
            # Check for target accuracy
            if verifier_accuracy >= target_accuracy:
                self.logger.info(f"🎯 Target accuracy achieved: {verifier_accuracy:.5f} >= {target_accuracy:.5f}")
                break
        
        pbar.close()
        
        total_time = time.time() - start_time
        
        # Final evaluation with larger batch
        self.logger.info("🔍 Final evaluation with larger batch...")
        with torch.no_grad():
            eval_batch_size = min(512, config.batch_size * 4)
            eval_input = torch.randn(eval_batch_size, *op_info.input_shape[1:]).to(self.device)
            
            eval_real_output, eval_real_proof = operation_wrapper(eval_input)
            eval_fake_output, eval_fake_proof = adversary(eval_input)
            
            eval_real_scores = verifier(eval_input, eval_real_output, eval_real_proof)
            eval_fake_scores = verifier(eval_input, eval_fake_output, eval_fake_proof)
            
            final_real_accept = (eval_real_scores > 0.5).float().mean().item()
            final_fake_reject = (eval_fake_scores < 0.5).float().mean().item()
            final_accuracy = (final_real_accept + final_fake_reject) / 2
            final_fool_rate = (eval_fake_scores > 0.5).float().mean().item()
        
        self.logger.info(f"✅ Training completed in {total_time:.1f}s")
        self.logger.info(f"📊 Final accuracy: {final_accuracy:.5f}")
        self.logger.info(f"📊 Real accept rate: {final_real_accept:.5f}")
        self.logger.info(f"📊 Fake reject rate: {final_fake_reject:.5f}")
        self.logger.info(f"📊 Adversary fool rate: {final_fool_rate:.5f}")
        
        return {
            "success": True,
            "final_verifier_accuracy": final_accuracy,
            "final_adversary_fool_rate": final_fool_rate,
            "final_real_accept_rate": final_real_accept,
            "final_fake_reject_rate": final_fake_reject,
            "epochs_trained": epoch + 1,
            "training_time": total_time,
            "metrics": metrics,
            "target_achieved": final_accuracy >= target_accuracy
        }
    
    def _export_nas_models(self, operation_wrapper, verifier, adversary, op_info: OpCompilationInfo, config: ArchitectureConfig):
        """Export NAS-trained models to ONNX."""
        
        import torch
        from pathlib import Path
        
        # Create output directory
        output_dir = Path("ops") / op_info.folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move models to CPU for ONNX export and validation
        operation_wrapper_cpu = operation_wrapper.cpu()
        verifier_cpu = verifier.cpu()
        adversary_cpu = adversary.cpu()
        
        # Create dummy inputs for ONNX export (ensure they're on CPU for ONNX export)
        dummy_input = torch.randn(1, *op_info.input_shape[1:])
        dummy_output = torch.randn(1, *op_info.output_shape[1:])
        
        # Calculate actual proof dimension from the configuration
        proof_dim = config.proof_dim
        
        # Validate that the operation wrapper produces the expected proof dimension
        with torch.no_grad():
            operation_wrapper_cpu.eval()
            test_output, test_proof = operation_wrapper_cpu(dummy_input)
            actual_proof_dim = test_proof.shape[-1]
            
            if actual_proof_dim != proof_dim:
                self.logger.warning(f"Prover proof dimension mismatch: expected {proof_dim}, got {actual_proof_dim}")
                proof_dim = actual_proof_dim
            
            # Also validate adversary proof dimension
            adversary_cpu.eval()
            _, adv_test_proof = adversary_cpu(dummy_input)
            adv_proof_dim = adv_test_proof.shape[-1]
            
            if adv_proof_dim != proof_dim:
                self.logger.warning(f"Adversary proof dimension mismatch: expected {proof_dim}, got {adv_proof_dim}")
                # Use the prover's proof dimension as the reference
                self.logger.info(f"Using prover proof dimension: {proof_dim}")
            
            # Validate output dimensions match expected
            expected_output_shape = op_info.output_shape[1:]  # Exclude batch dimension
            actual_output_shape = test_output.shape[1:]  # Exclude batch dimension
            
            if actual_output_shape != expected_output_shape:
                self.logger.warning(f"Output shape mismatch: expected {expected_output_shape}, got {actual_output_shape}")
                # Update op_info with actual shape
                op_info.output_shape = (1,) + actual_output_shape
                dummy_output = test_output[:1]  # Use actual output for export
            
            self.logger.info(f"Validated dimensions - Input: {op_info.input_shape}, Output: {op_info.output_shape}, Proof: {proof_dim}D")
        
        dummy_proof = torch.randn(1, proof_dim)
        
        # Export prover (operation wrapper)
        prover_path = output_dir / f"{op_info.folder_name}_prover.onnx"
        operation_wrapper_cpu.eval()
        torch.onnx.export(
            operation_wrapper_cpu,
            dummy_input,
            str(prover_path),
            export_params=True,
            opset_version=17,
            input_names=['input'],
            output_names=['output', 'proof'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
                'proof': {0: 'batch_size'}
            }
        )
        
        # Export verifier
        verifier_path = output_dir / f"{op_info.folder_name}_verifier.onnx"
        verifier_cpu.eval()
        torch.onnx.export(
            verifier_cpu,
            (dummy_input, dummy_output, dummy_proof),
            str(verifier_path),
            export_params=True,
            opset_version=17,
            input_names=['input', 'output', 'proof'],
            output_names=['score'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
                'proof': {0: 'batch_size'},
                'score': {0: 'batch_size'}
            }
        )
        
        # Export adversary
        adversary_path = output_dir / f"{op_info.folder_name}_adversary.onnx"
        adversary_cpu.eval()
        torch.onnx.export(
            adversary_cpu,
            dummy_input,
            str(adversary_path),
            export_params=True,
            opset_version=17,
            input_names=['input'],
            output_names=['fake_output', 'fake_proof'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'fake_output': {0: 'batch_size'},
                'fake_proof': {0: 'batch_size'}
            }
        )
        
        # Update op_info with paths
        op_info.prover_onnx_path = str(prover_path)
        op_info.verifier_onnx_path = str(verifier_path)
        op_info.adversary_onnx_path = str(adversary_path)
        op_info.is_compiled = True
        
        self.logger.info(f"📦 Exported NAS models to {output_dir}")
        self.logger.info(f"   Prover: {prover_path}")
        self.logger.info(f"   Verifier: {verifier_path}")
        self.logger.info(f"   Adversary: {adversary_path}")
    
    def _plot_training_metrics(self, op_info: OpCompilationInfo, metrics: Dict[str, list]):
        """Generate training plots for the compiled operation."""
        try:
            import matplotlib.pyplot as plt
            
            # Create plots directory
            plots_dir = Path("ops") / op_info.folder_name / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Metrics: {op_info.folder_name}', fontsize=16)
            
            epochs = range(len(metrics["verifier_accuracy"]))
            
            # Verifier Accuracy
            ax1.plot(epochs, metrics["verifier_accuracy"], 'b-', linewidth=2, label='Verifier Accuracy')
            ax1.axhline(y=0.99999, color='r', linestyle='--', alpha=0.7, label='Target (99.999%)')
            ax1.set_title('Verifier Accuracy Over Time')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.05)
            
            # Verifier Loss
            ax2.plot(epochs, metrics["verifier_loss"], 'g-', linewidth=2, label='Verifier Loss')
            ax2.set_title('Verifier Loss Over Time')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Adversary Loss
            ax3.plot(epochs, metrics["adversary_loss"], 'r-', linewidth=2, label='Adversary Loss')
            ax3.set_title('Adversary Loss Over Time')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Adversary Fool Rate
            ax4.plot(epochs, metrics["adversary_fool_rate"], 'm-', linewidth=2, label='Adversary Fool Rate')
            ax4.set_title('Adversary Fool Rate Over Time')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Fool Rate')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1.05)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = plots_dir / f"{op_info.folder_name}_training_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Training plots saved to {plot_path}")
            
        except ImportError:
            self.logger.warning("matplotlib not available, skipping plots")
        except Exception as e:
            self.logger.error(f"Failed to generate plots: {e}")
    
    def _generate_benchmarks(self, operation_wrapper, verifier, adversary, op_info: OpCompilationInfo, result: Dict):
        """Generate comprehensive benchmarks for the compiled operation."""
        try:
            import torch
            import time
            import json
            from pathlib import Path
            
            self.logger.info(f"🔬 Generating benchmarks for {op_info.folder_name}")
            
            # Create benchmarks directory
            benchmarks_dir = Path("ops") / op_info.folder_name / "benchmarks"
            benchmarks_dir.mkdir(parents=True, exist_ok=True)
            
            # Benchmark parameters
            test_batch_sizes = [1, 4, 16, 64]
            num_runs = 10
            
            benchmark_results = {
                "operation": op_info.folder_name,
                "input_shape": op_info.input_shape,
                "output_shape": op_info.output_shape,
                "timestamp": time.time(),
                "training_results": {
                    "final_accuracy": result["final_verifier_accuracy"],
                    "training_time": result["training_time"],
                    "epochs_trained": result["epochs_trained"],
                    "target_achieved": result["target_achieved"]
                },
                "performance_benchmarks": {},
                "accuracy_benchmarks": {}
            }
            
            # Move models back to original device for benchmarking
            operation_wrapper = operation_wrapper.to(self.device)
            verifier = verifier.to(self.device)
            adversary = adversary.to(self.device)
            
            operation_wrapper.eval()
            verifier.eval()
            adversary.eval()
            
            with torch.no_grad():
                for batch_size in test_batch_sizes:
                    self.logger.info(f"  📊 Benchmarking batch size {batch_size}")
                    
                    # Generate test data
                    test_input = torch.randn(batch_size, *op_info.input_shape[1:]).to(self.device)
                    
                    # Benchmark prover performance
                    prover_times = []
                    for _ in range(num_runs):
                        start_time = time.time()
                        real_output, real_proof = operation_wrapper(test_input)
                        torch.cuda.synchronize() if self.device == "cuda" else None
                        prover_times.append(time.time() - start_time)
                    
                    # Benchmark verifier performance
                    verifier_times = []
                    for _ in range(num_runs):
                        start_time = time.time()
                        scores = verifier(test_input, real_output, real_proof)
                        torch.cuda.synchronize() if self.device == "cuda" else None
                        verifier_times.append(time.time() - start_time)
                    
                    # Benchmark adversary performance
                    adversary_times = []
                    for _ in range(num_runs):
                        start_time = time.time()
                        fake_output, fake_proof = adversary(test_input)
                        torch.cuda.synchronize() if self.device == "cuda" else None
                        adversary_times.append(time.time() - start_time)
                    
                    # Performance metrics
                    benchmark_results["performance_benchmarks"][f"batch_{batch_size}"] = {
                        "prover": {
                            "mean_time": float(torch.tensor(prover_times).mean()),
                            "std_time": float(torch.tensor(prover_times).std()),
                            "throughput": batch_size / float(torch.tensor(prover_times).mean())
                        },
                        "verifier": {
                            "mean_time": float(torch.tensor(verifier_times).mean()),
                            "std_time": float(torch.tensor(verifier_times).std()),
                            "throughput": batch_size / float(torch.tensor(verifier_times).mean())
                        },
                        "adversary": {
                            "mean_time": float(torch.tensor(adversary_times).mean()),
                            "std_time": float(torch.tensor(adversary_times).std()),
                            "throughput": batch_size / float(torch.tensor(adversary_times).mean())
                        }
                    }
                    
                    # Accuracy benchmarks
                    large_test_input = torch.randn(256, *op_info.input_shape[1:]).to(self.device)
                    
                    # Real examples
                    real_output_large, real_proof_large = operation_wrapper(large_test_input)
                    real_scores = verifier(large_test_input, real_output_large, real_proof_large)
                    
                    # Fake examples
                    fake_output_large, fake_proof_large = adversary(large_test_input)
                    fake_scores = verifier(large_test_input, fake_output_large, fake_proof_large)
                    
                    # Mixed examples (real output + fake proof)
                    mixed_scores1 = verifier(large_test_input, real_output_large, fake_proof_large)
                    
                    # Mixed examples (fake output + real proof)
                    mixed_scores2 = verifier(large_test_input, fake_output_large, real_proof_large)
                    
                    benchmark_results["accuracy_benchmarks"][f"batch_{batch_size}"] = {
                        "real_accept_rate": float((real_scores > 0.5).float().mean()),
                        "fake_reject_rate": float((fake_scores < 0.5).float().mean()),
                        "mixed_reject_rate_1": float((mixed_scores1 < 0.5).float().mean()),
                        "mixed_reject_rate_2": float((mixed_scores2 < 0.5).float().mean()),
                        "overall_accuracy": float(((real_scores > 0.5).float().mean() + (fake_scores < 0.5).float().mean()) / 2),
                        "adversary_fool_rate": float((fake_scores > 0.5).float().mean())
                    }
            
            # Save benchmark results
            benchmark_path = benchmarks_dir / f"{op_info.folder_name}_benchmarks.json"
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            # Generate benchmark summary
            summary_path = benchmarks_dir / f"{op_info.folder_name}_benchmark_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"BENCHMARK SUMMARY: {op_info.folder_name}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Training Results:\n")
                f.write(f"  Final Accuracy: {result['final_verifier_accuracy']:.6f}\n")
                f.write(f"  Training Time: {result['training_time']:.1f}s\n")
                f.write(f"  Epochs Trained: {result['epochs_trained']}\n")
                f.write(f"  Target Achieved: {result['target_achieved']}\n\n")
                
                f.write("Performance Benchmarks (per sample):\n")
                for batch_size in test_batch_sizes:
                    perf = benchmark_results["performance_benchmarks"][f"batch_{batch_size}"]
                    f.write(f"  Batch {batch_size}:\n")
                    f.write(f"    Prover: {perf['prover']['mean_time']*1000/batch_size:.2f}ms ± {perf['prover']['std_time']*1000/batch_size:.2f}ms\n")
                    f.write(f"    Verifier: {perf['verifier']['mean_time']*1000/batch_size:.2f}ms ± {perf['verifier']['std_time']*1000/batch_size:.2f}ms\n")
                    f.write(f"    Adversary: {perf['adversary']['mean_time']*1000/batch_size:.2f}ms ± {perf['adversary']['std_time']*1000/batch_size:.2f}ms\n")
                
                f.write("\nAccuracy Benchmarks:\n")
                for batch_size in test_batch_sizes:
                    acc = benchmark_results["accuracy_benchmarks"][f"batch_{batch_size}"]
                    f.write(f"  Batch {batch_size}:\n")
                    f.write(f"    Real Accept Rate: {acc['real_accept_rate']:.6f}\n")
                    f.write(f"    Fake Reject Rate: {acc['fake_reject_rate']:.6f}\n")
                    f.write(f"    Overall Accuracy: {acc['overall_accuracy']:.6f}\n")
                    f.write(f"    Adversary Fool Rate: {acc['adversary_fool_rate']:.6f}\n")
            
            self.logger.info(f"📊 Benchmarks saved to {benchmarks_dir}")
            self.logger.info(f"  Detailed results: {benchmark_path}")
            self.logger.info(f"  Summary: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate benchmarks: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _save_compilation_metadata(self, op_info: OpCompilationInfo, config: ArchitectureConfig, result: Dict):
        """Save comprehensive compilation metadata."""
        try:
            import json
            import time
            from pathlib import Path
            
            # Create metadata directory
            metadata_dir = Path("ops") / op_info.folder_name / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            # Compilation metadata
            metadata = {
                "operation_info": {
                    "folder_name": op_info.folder_name,
                    "operation_type": getattr(op_info, 'operation_type', op_info.op_type.value),
                    "input_shape": op_info.input_shape,
                    "output_shape": op_info.output_shape,
                    "is_compiled": op_info.is_compiled
                },
                "architecture_config": config.to_dict(),
                "compilation_results": result,
                "compilation_timestamp": time.time(),
                "file_paths": {
                    "prover_onnx": getattr(op_info, 'prover_onnx_path', None),
                    "verifier_onnx": getattr(op_info, 'verifier_onnx_path', None),
                    "adversary_onnx": getattr(op_info, 'adversary_onnx_path', None)
                },
                "system_info": {
                    "device": self.device,
                    "target_accuracy": self.target_accuracy,
                    "framework_version": "1.0.0"
                }
            }
            
            # Save metadata
            metadata_path = metadata_dir / f"{op_info.folder_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"📋 Compilation metadata saved to {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save compilation metadata: {e}")

# Monkey patch the enhanced compiler into the NAS framework
NASCompilationFramework.compiler_class = NASEnabledCompiler 