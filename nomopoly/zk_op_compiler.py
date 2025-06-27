"""
ZK Operation Compiler

A clean interface for compiling individual ONNX operations into ZK operations.
Uses the NAS compilation framework and caches results in the ops folder.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from .ops_registry import ops_registry, OpCompilationInfo, SupportedOp
from .nas_compilation_framework import NASCompilationFramework


class ZKOp:
    """
    A compiled ZK operation that can generate proofs.
    
    This represents a single ONNX operation that has been compiled to support
    zero-knowledge proofs. It contains the prover, verifier, and adversary models.
    """
    
    def __init__(self, op_info: OpCompilationInfo):
        self.op_info = op_info
        self.folder_name = op_info.folder_name
        self.op_type = op_info.op_type
        self.input_shape = op_info.input_shape
        self.output_shape = op_info.output_shape
        
        # Paths to compiled models
        self.prover_path = op_info.prover_onnx_path
        self.verifier_path = op_info.verifier_onnx_path
        self.adversary_path = op_info.adversary_onnx_path
        
        # Lazy loading of ONNX sessions
        self._prover_session = None
        self._verifier_session = None
        self._adversary_session = None
    
    @property
    def is_compiled(self) -> bool:
        """Check if this operation is fully compiled."""
        return self.op_info.compilation_complete
    
    @property
    def prover_session(self):
        """Lazy load prover ONNX session."""
        if self._prover_session is None:
            import onnxruntime as ort
            self._prover_session = ort.InferenceSession(
                self.prover_path, 
                providers=['CPUExecutionProvider']
            )
        return self._prover_session
    
    @property 
    def verifier_session(self):
        """Lazy load verifier ONNX session."""
        if self._verifier_session is None:
            import onnxruntime as ort
            self._verifier_session = ort.InferenceSession(
                self.verifier_path,
                providers=['CPUExecutionProvider'] 
            )
        return self._verifier_session
    
    @property
    def adversary_session(self):
        """Lazy load adversary ONNX session."""
        if self._adversary_session is None:
            import onnxruntime as ort
            self._adversary_session = ort.InferenceSession(
                self.adversary_path,
                providers=['CPUExecutionProvider']
            )
        return self._adversary_session
    
    def prove(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the prover to get operation output and proof.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Tuple of (operation_output, proof)
        """
        if not self.is_compiled:
            raise RuntimeError(f"Operation {self.folder_name} is not compiled")
        
        # Convert to numpy for ONNX runtime
        input_np = input_data.detach().cpu().numpy()
        
        # Run prover
        prover_outputs = self.prover_session.run(None, {
            self.prover_session.get_inputs()[0].name: input_np
        })
        
        # Convert back to tensors
        operation_output = torch.from_numpy(prover_outputs[0])
        proof = torch.from_numpy(prover_outputs[1])
        
        return operation_output, proof
    
    def verify(self, input_data: torch.Tensor, output_data: torch.Tensor, 
               proof: torch.Tensor) -> torch.Tensor:
        """
        Run the verifier to check if proof is valid.
        
        Args:
            input_data: Input tensor
            output_data: Expected output tensor  
            proof: Proof tensor
            
        Returns:
            Verification score (0-1, higher means more believable)
        """
        if not self.is_compiled:
            raise RuntimeError(f"Operation {self.folder_name} is not compiled")
        
        # Convert to numpy for ONNX runtime
        inputs_np = {
            self.verifier_session.get_inputs()[0].name: input_data.detach().cpu().numpy(),
            self.verifier_session.get_inputs()[1].name: output_data.detach().cpu().numpy(),
            self.verifier_session.get_inputs()[2].name: proof.detach().cpu().numpy()
        }
        
        # Run verifier
        verification_result = self.verifier_session.run(None, inputs_np)
        
        return torch.from_numpy(verification_result[0])
    
    def __repr__(self) -> str:
        status = "✅ compiled" if self.is_compiled else "❌ not compiled"
        return f"ZKOp({self.folder_name}, {status})"


class ZKOpCompiler:
    """
    Compiler for converting ONNX operations into ZK operations.
    
    This class provides a clean interface for:
    1. Checking if operations are already compiled (cached)
    2. Compiling new operations using NAS framework
    3. Loading compiled operations as ZKOp instances
    """
    
    def __init__(self, ops_dir: str = "ops", device: str = "mps"):
        self.ops_dir = Path(ops_dir)
        self.device = device
        self.logger = logging.getLogger("ZKOpCompiler")
        
        # Initialize the NAS framework for compilation
        self.nas_framework = NASCompilationFramework(
            ops_dir=str(ops_dir),
            device=device
        )
        
        # Cache of compiled operations
        self._compiled_ops_cache: Dict[str, ZKOp] = {}
        
        # Scan for existing compiled operations on initialization
        self._scan_for_compiled_operations()
    
    def _scan_for_compiled_operations(self):
        """Scan the ops directory for existing compiled operations and register them."""
        if not self.ops_dir.exists():
            self.logger.info(f"📁 Ops directory {self.ops_dir} does not exist")
            return
        
        self.logger.info(f"🔍 Scanning {self.ops_dir} for compiled operations...")
        
        # Scan each subdirectory in ops/
        for op_dir in self.ops_dir.iterdir():
            if not op_dir.is_dir():
                continue
            
            op_name = op_dir.name
            
            # Check if this operation has all required ONNX files
            prover_path = op_dir / f"{op_name}_prover.onnx"
            verifier_path = op_dir / f"{op_name}_verifier.onnx"
            adversary_path = op_dir / f"{op_name}_adversary.onnx"
            
            if prover_path.exists() and verifier_path.exists() and adversary_path.exists():
                # Try to infer operation details from the folder name
                op_info = self._create_op_info_from_folder(op_dir)
                if op_info:
                    # Register in the ops registry
                    ops_registry.discovered_ops[op_name] = op_info
                    self.logger.info(f"✅ Found compiled operation: {op_name}")
                else:
                    self.logger.warning(f"⚠️ Could not parse operation info for {op_name}")
            else:
                self.logger.debug(f"📝 Incomplete operation found: {op_name}")
    
    def _create_op_info_from_folder(self, op_dir: Path) -> Optional[OpCompilationInfo]:
        """Create OpCompilationInfo from a compiled operation folder."""
        from .ops_registry import OpCompilationInfo, SupportedOp
        
        op_name = op_dir.name
        
        # Try to parse operation type and shape from folder name
        # Format examples: "relu_1x256", "conv_1x3x8x8", "gemm_1x256"
        parts = op_name.split('_')
        if len(parts) < 2:
            return None
        
        op_type_str = parts[0].upper()
        shape_str = '_'.join(parts[1:])
        
        # Map operation type
        op_type_map = {
            'RELU': SupportedOp.RELU,
            'CONV': SupportedOp.CONV2D,
            'GEMM': SupportedOp.GEMM,
            'MATMUL': SupportedOp.MATMUL,
            'MAXPOOL': SupportedOp.MAXPOOL,
            'AVGPOOL': SupportedOp.AVGPOOL,
            'FLATTEN': SupportedOp.FLATTEN,
            'ADD': SupportedOp.ADD
        }
        
        op_type = op_type_map.get(op_type_str)
        if not op_type:
            return None
        
        # Parse shape (e.g., "1x256" -> [1, 256])
        try:
            shape_parts = shape_str.split('x')
            input_shape = [int(x) for x in shape_parts]
            
            # Infer output shape based on operation type
            if op_type == SupportedOp.RELU:
                output_shape = input_shape
            elif op_type == SupportedOp.FLATTEN:
                output_shape = [input_shape[0], -1]  # Flatten to 2D
            elif op_type in [SupportedOp.GEMM, SupportedOp.MATMUL]:
                output_shape = input_shape  # Assume same for now
            elif op_type in [SupportedOp.MAXPOOL, SupportedOp.AVGPOOL]:
                # Assume 2x2 pooling reduces spatial dimensions by half
                if len(input_shape) == 4:  # [N, C, H, W]
                    output_shape = [input_shape[0], input_shape[1], input_shape[2]//2, input_shape[3]//2]
                else:
                    output_shape = input_shape
            elif op_type == SupportedOp.CONV2D:
                # Assume same spatial size for simplicity
                output_shape = input_shape
            else:
                output_shape = input_shape
                
        except (ValueError, IndexError):
            return None
        
        # Create OpCompilationInfo
        op_info = OpCompilationInfo(
            op_type=op_type,
            input_shape=input_shape,
            output_shape=output_shape,
            attributes={},
            prover_onnx_path=str(op_dir / f"{op_name}_prover.onnx"),
            verifier_onnx_path=str(op_dir / f"{op_name}_verifier.onnx"),
            adversary_onnx_path=str(op_dir / f"{op_name}_adversary.onnx"),
            compilation_log_path=str(op_dir / "compilation.log"),
            is_compiled=True
        )
        
        return op_info
    
    def is_compiled(self, op_info: OpCompilationInfo) -> bool:
        """Check if an operation is already compiled."""
        return op_info.compilation_complete
    
    def get_compiled_op(self, op_info: OpCompilationInfo) -> Optional[ZKOp]:
        """
        Get a compiled ZK operation if it exists.
        
        Args:
            op_info: Operation information
            
        Returns:
            ZKOp instance if compiled, None otherwise
        """
        if not self.is_compiled(op_info):
            return None
        
        # Check cache first
        if op_info.folder_name in self._compiled_ops_cache:
            return self._compiled_ops_cache[op_info.folder_name]
        
        # Create and cache ZKOp
        zk_op = ZKOp(op_info)
        self._compiled_ops_cache[op_info.folder_name] = zk_op
        
        return zk_op
    
    def compile_operation(
        self, 
        op_info: OpCompilationInfo,
        force_recompile: bool = False,
        target_accuracy: float = 0.99999
    ) -> ZKOp:
        """
        Compile an operation into a ZK operation.
        
        Args:
            op_info: Operation information
            force_recompile: Whether to recompile if already compiled
            target_accuracy: Target accuracy for compilation
            
        Returns:
            Compiled ZKOp instance
        """
        # Check if already compiled and not forcing recompile
        if not force_recompile and self.is_compiled(op_info):
            self.logger.info(f"✅ {op_info.folder_name} already compiled, using cache")
            return self.get_compiled_op(op_info)
        
        self.logger.info(f"🔧 Compiling {op_info.folder_name} to target accuracy {target_accuracy:.5f}")
        
        # Register the operation if not already registered
        if op_info.folder_name not in ops_registry.discovered_ops:
            ops_registry.discovered_ops[op_info.folder_name] = op_info
        
        # Use NAS framework to compile
        evolution_results = self.nas_framework.evolve_all_operations_to_precision(
            force_recompile=force_recompile
        )
        
        # Check if compilation was successful
        result = evolution_results.get(op_info.folder_name)
        if not result or not result.get("success", False):
            error_msg = result.get("error", "Unknown error") if result else "No result"
            raise RuntimeError(f"Failed to compile {op_info.folder_name}: {error_msg}")
        
        # Refresh operation info to get updated paths
        updated_op_info = ops_registry.discovered_ops[op_info.folder_name]
        
        if not updated_op_info.compilation_complete:
            raise RuntimeError(f"Compilation claimed success but files not found for {op_info.folder_name}")
        
        self.logger.info(f"✅ Successfully compiled {op_info.folder_name}")
        
        # Create and cache ZKOp
        zk_op = ZKOp(updated_op_info)
        self._compiled_ops_cache[op_info.folder_name] = zk_op
        
        return zk_op
    
    def compile_multiple_operations(
        self,
        op_infos: list[OpCompilationInfo],
        force_recompile: bool = False,
        target_accuracy: float = 0.99999
    ) -> Dict[str, ZKOp]:
        """
        Compile multiple operations efficiently.
        
        Args:
            op_infos: List of operation information
            force_recompile: Whether to recompile if already compiled
            target_accuracy: Target accuracy for compilation
            
        Returns:
            Dictionary mapping operation names to ZKOp instances
        """
        self.logger.info(f"🔧 Compiling {len(op_infos)} operations")
        
        # Filter operations that need compilation
        ops_to_compile = []
        already_compiled = {}
        
        for op_info in op_infos:
            if force_recompile or not self.is_compiled(op_info):
                ops_to_compile.append(op_info)
                # Register the operation
                ops_registry.discovered_ops[op_info.folder_name] = op_info
            else:
                self.logger.info(f"✅ {op_info.folder_name} already compiled")
                already_compiled[op_info.folder_name] = self.get_compiled_op(op_info)
        
        # Compile operations that need it
        if ops_to_compile:
            self.logger.info(f"🧬 Running NAS compilation for {len(ops_to_compile)} operations")
            evolution_results = self.nas_framework.evolve_all_operations_to_precision(
                force_recompile=force_recompile
            )
            
            # Process results
            compiled_ops = {}
            for op_info in ops_to_compile:
                result = evolution_results.get(op_info.folder_name)
                if result and result.get("success", False):
                    updated_op_info = ops_registry.discovered_ops[op_info.folder_name]
                    if updated_op_info.compilation_complete:
                        zk_op = ZKOp(updated_op_info)
                        self._compiled_ops_cache[op_info.folder_name] = zk_op
                        compiled_ops[op_info.folder_name] = zk_op
                        self.logger.info(f"✅ Compiled {op_info.folder_name}")
                    else:
                        self.logger.error(f"❌ Compilation files missing for {op_info.folder_name}")
                else:
                    error_msg = result.get("error", "Unknown error") if result else "No result"
                    self.logger.error(f"❌ Failed to compile {op_info.folder_name}: {error_msg}")
            
            # Combine with already compiled
            compiled_ops.update(already_compiled)
            return compiled_ops
        else:
            self.logger.info("✅ All operations already compiled")
            return already_compiled
    
    def list_compiled_operations(self) -> Dict[str, ZKOp]:
        """
        List all compiled operations available in the ops cache.
        
        Returns:
            Dictionary mapping operation names to ZKOp instances
        """
        compiled_ops = {}
        
        for op_info in ops_registry.get_compiled_operations():
            if op_info.folder_name not in self._compiled_ops_cache:
                self._compiled_ops_cache[op_info.folder_name] = ZKOp(op_info)
            compiled_ops[op_info.folder_name] = self._compiled_ops_cache[op_info.folder_name]
        
        return compiled_ops
    
    def get_compiled_operation(self, op_name: str) -> Optional[ZKOp]:
        """Get a compiled operation by name."""
        compiled_ops = self.list_compiled_operations()
        return compiled_ops.get(op_name)
    
    def clear_cache(self):
        """Clear the compiled operations cache."""
        self._compiled_ops_cache.clear()
        self.logger.info("🗑️ Cleared compiled operations cache")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the operations cache."""
        total_discovered = len(ops_registry.discovered_ops)
        total_compiled = len(ops_registry.get_compiled_operations())
        cached_ops = len(self._compiled_ops_cache)
        
        return {
            "total_discovered": total_discovered,
            "total_compiled": total_compiled, 
            "cached_in_memory": cached_ops,
            "uncompiled": total_discovered - total_compiled
        }


# Global ZK operation compiler instance
zk_op_compiler = ZKOpCompiler() 