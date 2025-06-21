"""
Benchmarking and evaluation utilities for Zero Knowledge ML models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import time
import json
import os
from .networks import ZKProverNet, ZKVerifierNet, ZKAdversarialNet


class ZKMLBenchmark:
    """Comprehensive benchmark suite for Zero Knowledge ML models."""
    
    def __init__(
        self,
        prover: ZKProverNet,
        verifier: ZKVerifierNet,
        adversary: ZKAdversarialNet,
        device: str = "auto"
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.prover = prover.to(self.device)
        self.verifier = verifier.to(self.device)
        self.adversary = adversary.to(self.device)
        
    def evaluate_prover_accuracy(
        self,
        test_data: torch.Tensor,
        tolerance: float = 1e-6
    ) -> Dict[str, float]:
        """Evaluate the computational accuracy of the prover network."""
        self.prover.eval()
        
        with torch.no_grad():
            test_data = test_data.to(self.device)
            
            # Get prover outputs
            prover_outputs, _ = self.prover(test_data)
            
            # Compute ground truth (sum of inputs)
            ground_truth = torch.sum(test_data, dim=1, keepdim=True)
            
            # Compute absolute errors
            abs_errors = torch.abs(prover_outputs - ground_truth)
            
            # Compute metrics
            mae = torch.mean(abs_errors).item()
            mse = torch.mean((prover_outputs - ground_truth) ** 2).item()
            accuracy = (abs_errors < tolerance).float().mean().item()
            
            max_error = torch.max(abs_errors).item()
            min_error = torch.min(abs_errors).item()
            
        return {
            "mae": mae,
            "mse": mse,
            "accuracy": accuracy,
            "max_error": max_error,
            "min_error": min_error,
            "rmse": np.sqrt(mse)
        }
    
    def run_comprehensive_benchmark(
        self,
        test_data: torch.Tensor,
        save_results: bool = True,
        results_dir: str = "./benchmark_results"
    ) -> Dict[str, Any]:
        """Run a comprehensive benchmark of all components."""
        print("Running comprehensive ZK-ML benchmark...")
        
        results = {}
        
        # Prover accuracy evaluation
        print("Evaluating prover accuracy...")
        results["prover_accuracy"] = self.evaluate_prover_accuracy(test_data)
        
        # Save results if requested
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            
            # Save JSON results
            results_file = os.path.join(results_dir, "benchmark_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {results_file}")
        
        return results
