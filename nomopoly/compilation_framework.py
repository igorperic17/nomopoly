"""
ONNX Compilation Framework

Main framework for scanning ONNX models, discovering operations, and compiling
them into proof-capable components. This ties together the ops registry and
operation compiler into a complete workflow.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from .ops_registry import ops_registry, OpCompilationInfo, SupportedOp
from .onnx_compiler import ONNXOperationCompiler


class ONNXCompilationFramework:
    """
    Main framework for compiling ONNX operations into proof-capable components.
    
    This class orchestrates the entire compilation workflow:
    1. Scan ONNX models to discover operations
    2. Identify uncompiled operations 
    3. Compile each operation with adversarial training
    4. Export compiled ONNX models (prover, verifier, adversary)
    5. Log progress and save metrics
    """
    
    def __init__(
        self,
        ops_dir: str = "ops",
        device: str = "mps"
    ):
        self.ops_dir = Path(ops_dir)
        self.device = device
        
        # Create operations directory
        self.ops_dir.mkdir(exist_ok=True)
        
        # Initialize compiler
        self.compiler = ONNXOperationCompiler(device=device)
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up framework-level logging."""
        log_file = self.ops_dir / "compilation_framework.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("ONNXCompilationFramework")
        
    def scan_and_register_model(self, onnx_model_path: str) -> List[OpCompilationInfo]:
        """
        Scan an ONNX model and register all supported operations.
        
        Args:
            onnx_model_path: Path to the ONNX model to scan
            
        Returns:
            List of discovered operation information objects
        """
        self.logger.info(f"🔍 Scanning ONNX model: {onnx_model_path}")
        
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        # Use the global registry to scan the model
        discovered_ops = ops_registry.scan_onnx_model(onnx_model_path)
        
        self.logger.info(f"✅ Discovered {len(discovered_ops)} operations in {onnx_model_path}")
        
        return discovered_ops
    
    def compile_uncompiled_operations(
        self,
        num_epochs: int = 200,
        batch_size: int = 32,
        proof_dim: int = 32,
        force_recompile: bool = False,
        target_accuracy: float = 0.99,
        max_epochs: int = 1000
    ) -> Dict[str, Dict]:
        """
        Compile all uncompiled operations in the registry.
        
        Args:
            num_epochs: Minimum number of training epochs per operation
            batch_size: Training batch size
            proof_dim: Dimension of proof vectors
            force_recompile: If True, recompile even already compiled operations
            target_accuracy: Target verifier accuracy (default 99%)
            max_epochs: Maximum number of epochs to prevent infinite training
            
        Returns:
            Dictionary mapping operation names to compilation results
        """
        if force_recompile:
            uncompiled_ops = list(ops_registry.discovered_ops.values())
            self.logger.info(f"🔄 Force recompiling {len(uncompiled_ops)} operations")
        else:
            uncompiled_ops = ops_registry.get_uncompiled_operations()
            self.logger.info(f"🔧 Found {len(uncompiled_ops)} uncompiled operations")
        
        if not uncompiled_ops:
            self.logger.info("✅ All operations are already compiled!")
            return {}
        
        compilation_results = {}
        total_start_time = time.time()
        
        for i, op_info in enumerate(uncompiled_ops):
            self.logger.info(f"\n📌 Compiling operation {i+1}/{len(uncompiled_ops)}: {op_info.folder_name}")
            
            start_time = time.time()
            
            try:
                # Compile the operation
                result = self.compiler.compile_operation(
                    op_info=op_info,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    proof_dim=proof_dim,
                    target_accuracy=target_accuracy,
                    max_epochs=max_epochs
                )
                
                compilation_time = time.time() - start_time
                result["compilation_time"] = compilation_time
                
                compilation_results[op_info.folder_name] = result
                
                if result["success"]:
                    self.logger.info(f"✅ Successfully compiled {op_info.folder_name} in {compilation_time:.1f}s")
                    self.logger.info(f"   Final verifier accuracy: {result['final_verifier_accuracy']:.3f}")
                    self.logger.info(f"   Final adversary fool rate: {result['final_adversary_fool_rate']:.3f}")
                else:
                    self.logger.error(f"❌ Failed to compile {op_info.folder_name}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                compilation_time = time.time() - start_time
                error_msg = f"Exception during compilation: {str(e)}"
                self.logger.error(f"❌ {op_info.folder_name}: {error_msg}")
                
                compilation_results[op_info.folder_name] = {
                    "success": False,
                    "error": error_msg,
                    "compilation_time": compilation_time
                }
        
        total_time = time.time() - total_start_time
        
        # Summary
        successful_compilations = [r for r in compilation_results.values() if r.get("success", False)]
        failed_compilations = [r for r in compilation_results.values() if not r.get("success", False)]
        
        self.logger.info(f"\n📊 Compilation Summary:")
        self.logger.info(f"   ✅ Successful: {len(successful_compilations)}")
        self.logger.info(f"   ❌ Failed: {len(failed_compilations)}")
        self.logger.info(f"   ⏱️  Total time: {total_time:.1f}s")
        
        if successful_compilations:
            avg_accuracy = sum(r["final_verifier_accuracy"] for r in successful_compilations) / len(successful_compilations)
            self.logger.info(f"   📈 Average verifier accuracy: {avg_accuracy:.3f}")
        
        return compilation_results
    
    def compile_model_operations(
        self,
        onnx_model_path: str,
        num_epochs: int = 200,
        batch_size: int = 32,
        proof_dim: int = 32,
        force_recompile: bool = False,
        target_accuracy: float = 0.99,
        max_epochs: int = 1000
    ) -> Dict[str, Dict]:
        """
        Complete workflow: scan a model and compile all its operations.
        
        Args:
            onnx_model_path: Path to ONNX model to process
            num_epochs: Minimum number of training epochs per operation
            batch_size: Training batch size
            proof_dim: Dimension of proof vectors
            force_recompile: If True, recompile even already compiled operations
            target_accuracy: Target verifier accuracy (default 99%)
            max_epochs: Maximum number of epochs to prevent infinite training
            
        Returns:
            Dictionary mapping operation names to compilation results
        """
        self.logger.info(f"🚀 Starting complete compilation workflow for: {onnx_model_path}")
        
        # Step 1: Scan and register operations
        discovered_ops = self.scan_and_register_model(onnx_model_path)
        
        if not discovered_ops:
            self.logger.warning("⚠️  No supported operations found in the model")
            return {}
        
        # Step 2: Compile uncompiled operations
        compilation_results = self.compile_uncompiled_operations(
            num_epochs=num_epochs,
            batch_size=batch_size,
            proof_dim=proof_dim,
            force_recompile=force_recompile,
            target_accuracy=target_accuracy,
            max_epochs=max_epochs
        )
        
        # Step 3: Print final registry status
        ops_registry.print_registry_status()
        
        return compilation_results
    
    def list_compiled_operations(self) -> List[OpCompilationInfo]:
        """Get list of all compiled operations."""
        return ops_registry.get_compiled_operations()
    
    def list_uncompiled_operations(self) -> List[OpCompilationInfo]:
        """Get list of all uncompiled operations."""
        return ops_registry.get_uncompiled_operations()
    
    def get_operation_info(self, operation_name: str) -> Optional[OpCompilationInfo]:
        """Get information about a specific operation."""
        return ops_registry.discovered_ops.get(operation_name)
    
    def validate_compiled_models(self) -> Dict[str, bool]:
        """
        Validate that all compiled ONNX models are valid.
        
        Returns:
            Dictionary mapping operation names to validation status
        """
        from .utils import validate_onnx_model
        
        self.logger.info("🔍 Validating compiled ONNX models...")
        
        compiled_ops = self.list_compiled_operations()
        validation_results = {}
        
        for op_info in compiled_ops:
            op_name = op_info.folder_name
            all_valid = True
            
            # Check prover
            if op_info.prover_onnx_path and os.path.exists(op_info.prover_onnx_path):
                prover_valid = validate_onnx_model(op_info.prover_onnx_path)
                if not prover_valid:
                    self.logger.warning(f"⚠️  Invalid prover model: {op_name}")
                    all_valid = False
            else:
                self.logger.warning(f"⚠️  Missing prover model: {op_name}")
                all_valid = False
            
            # Check verifier  
            if op_info.verifier_onnx_path and os.path.exists(op_info.verifier_onnx_path):
                verifier_valid = validate_onnx_model(op_info.verifier_onnx_path)
                if not verifier_valid:
                    self.logger.warning(f"⚠️  Invalid verifier model: {op_name}")
                    all_valid = False
            else:
                self.logger.warning(f"⚠️  Missing verifier model: {op_name}")
                all_valid = False
            
            # Check adversary
            if op_info.adversary_onnx_path and os.path.exists(op_info.adversary_onnx_path):
                adversary_valid = validate_onnx_model(op_info.adversary_onnx_path)
                if not adversary_valid:
                    self.logger.warning(f"⚠️  Invalid adversary model: {op_name}")
                    all_valid = False
            else:
                self.logger.warning(f"⚠️  Missing adversary model: {op_name}")
                all_valid = False
            
            validation_results[op_name] = all_valid
            
            if all_valid:
                self.logger.info(f"✅ {op_name}: All models valid")
            else:
                self.logger.error(f"❌ {op_name}: Some models invalid/missing")
        
        valid_count = sum(validation_results.values())
        total_count = len(validation_results)
        
        self.logger.info(f"📊 Validation Summary: {valid_count}/{total_count} operations have valid models")
        
        return validation_results
    
    def cleanup_incomplete_compilations(self):
        """Remove incomplete compilation artifacts."""
        self.logger.info("🧹 Cleaning up incomplete compilations...")
        
        incomplete_count = 0
        
        for op_info in ops_registry.discovered_ops.values():
            if not op_info.compilation_complete:
                # Remove partial artifacts
                for path in [op_info.prover_onnx_path, op_info.verifier_onnx_path, op_info.adversary_onnx_path]:
                    if path and os.path.exists(path):
                        os.remove(path)
                        self.logger.info(f"   Removed: {path}")
                        incomplete_count += 1
        
        if incomplete_count > 0:
            self.logger.info(f"✅ Cleaned up {incomplete_count} incomplete artifacts")
        else:
            self.logger.info("✅ No incomplete artifacts found")


# Create global compilation framework instance
compilation_framework = ONNXCompilationFramework() 