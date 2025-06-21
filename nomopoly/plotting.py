"""
Plotting and visualization utilities for ZK-ML training results.

This module provides comprehensive plotting functionality for:
- Training progress visualization
- Performance metrics analysis
- Verification accuracy breakdowns
- Score separation analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os


class ZKPlotter:
    """
    Comprehensive plotting utilities for ZK-ML training results.
    """
    
    def __init__(self, plots_dir: str = "plots"):
        self.plots_dir = plots_dir
        self._ensure_plots_directory()
    
    def _ensure_plots_directory(self):
        """Create plots directory if it doesn't exist."""
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def plot_training_progress(self, stats: Dict[str, List[float]], save_path: Optional[str] = None) -> str:
        """
        Plot comprehensive training statistics for triplet verification.
        
        Args:
            stats: Dictionary containing training statistics
            save_path: Optional path to save the plot
            
        Returns:
            Path where the plot was saved
        """
        if save_path is None:
            save_path = os.path.join(self.plots_dir, "training_progress.png")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss curves
        axes[0, 0].plot(stats["verifier_loss"], label="Verifier Loss", linewidth=2, color='blue')
        axes[0, 0].plot(stats["malicious_loss"], label="Malicious Loss", linewidth=2, color='red')
        axes[0, 0].set_title("Training Losses")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Overall verification accuracy (main metric)
        axes[0, 1].plot(stats["overall_accuracy"], linewidth=3, color='green', label="Overall Accuracy")
        axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', label='Random Chance')
        axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', label='Target: 90%')
        axes[0, 1].axhline(y=0.95, color='purple', linestyle='--', label='Excellent: 95%')
        axes[0, 1].set_title("Overall Verification Accuracy (MAIN METRIC)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Four verification types breakdown
        axes[0, 2].plot(stats["real_real_acc"], label="✅ Real+Real", linewidth=2, color='green')
        axes[0, 2].plot(stats["fake_fake_acc"], label="❌ Fake+Fake", linewidth=2, color='red')
        axes[0, 2].plot(stats["fake_real_acc"], label="❌ Fake+Real", linewidth=2, color='orange')
        axes[0, 2].plot(stats["real_wrong_acc"], label="❌ Real+Wrong", linewidth=2, color='purple')
        axes[0, 2].axhline(y=0.95, color='gray', linestyle='--', label='Target: 95%')
        axes[0, 2].set_title("Four Verification Types")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Accuracy")
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Malicious success rate (adversarial balance)
        axes[1, 0].plot(stats["malicious_success"], linewidth=3, color='red', label="Malicious Success")
        axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', label='Balanced')
        axes[1, 0].axhline(y=0.2, color='green', linestyle='--', label='Good Defense')
        axes[1, 0].axhline(y=0.8, color='orange', linestyle='--', label='Strong Attack')
        axes[1, 0].set_title("Malicious Success Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Success Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Score means for different verification types
        axes[1, 1].plot(stats["real_real_mean"], label="Real+Real Score", linewidth=2, color='green')
        axes[1, 1].plot(stats["fake_fake_mean"], label="Fake+Fake Score", linewidth=2, color='red')
        axes[1, 1].plot(stats["fake_real_mean"], label="Fake+Real Score", linewidth=2, color='orange')
        axes[1, 1].plot(stats["real_wrong_mean"], label="Real+Wrong Score", linewidth=2, color='purple')
        axes[1, 1].set_title("Verification Score Means")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Score separation (key indicator of learning)
        axes[1, 2].plot(stats["score_separation"], linewidth=3, color='purple', label="Score Separation")
        axes[1, 2].axhline(y=0, color='gray', linestyle='--', label='No Separation')
        axes[1, 2].axhline(y=0.3, color='green', linestyle='--', label='Good Separation')
        axes[1, 2].axhline(y=0.5, color='orange', linestyle='--', label='Excellent Separation')
        axes[1, 2].set_title("Score Separation (Real vs Fake)")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Separation")
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training progress plot saved to {save_path}")
        return save_path
    
    def plot_verification_breakdown(self, stats: Dict[str, List[float]], save_path: Optional[str] = None) -> str:
        """
        Create a detailed breakdown of verification performance.
        
        Args:
            stats: Dictionary containing training statistics
            save_path: Optional path to save the plot
            
        Returns:
            Path where the plot was saved
        """
        if save_path is None:
            save_path = os.path.join(self.plots_dir, "verification_breakdown.png")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Final accuracy bar chart
        final_accuracies = [
            stats["real_real_acc"][-1],
            stats["fake_fake_acc"][-1], 
            stats["fake_real_acc"][-1],
            stats["real_wrong_acc"][-1]
        ]
        labels = ['Real+Real\n(Accept)', 'Fake+Fake\n(Reject)', 'Fake+Real\n(Reject)', 'Real+Wrong\n(Reject)']
        colors = ['green', 'red', 'orange', 'purple']
        
        bars = axes[0, 0].bar(labels, final_accuracies, color=colors, alpha=0.7)
        axes[0, 0].axhline(y=0.95, color='gray', linestyle='--', label='Target: 95%')
        axes[0, 0].set_title("Final Verification Accuracy by Type")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].legend()
        
        # Add percentage labels on bars
        for bar, acc in zip(bars, final_accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Score distribution evolution
        epochs = list(range(len(stats["real_real_mean"])))
        axes[0, 1].fill_between(epochs, stats["real_real_mean"], alpha=0.3, color='green', label='Real+Real')
        axes[0, 1].plot(stats["real_real_mean"], color='green', linewidth=2)
        
        # Plot fake score ranges
        fake_scores = np.array([stats["fake_fake_mean"], stats["fake_real_mean"], stats["real_wrong_mean"]])
        fake_min = np.min(fake_scores, axis=0)
        fake_max = np.max(fake_scores, axis=0)
        fake_mean = np.mean(fake_scores, axis=0)
        
        axes[0, 1].fill_between(epochs, fake_min, fake_max, alpha=0.3, color='red', label='Fake Range')
        axes[0, 1].plot(fake_mean, color='red', linewidth=2, label='Fake Mean')
        
        axes[0, 1].set_title("Score Distribution Evolution")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Verification Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning curve comparison
        axes[1, 0].plot(stats["overall_accuracy"], linewidth=3, color='blue', label='Overall Accuracy')
        axes[1, 0].plot([1 - ms for ms in stats["malicious_success"]], linewidth=2, color='red', 
                       linestyle='--', label='Defense Success (1 - Malicious)')
        axes[1, 0].axhline(y=0.9, color='green', linestyle='--', label='Target: 90%')
        axes[1, 0].set_title("Learning Progress")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Success Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Training stability analysis
        window_size = min(10, len(stats["overall_accuracy"]) // 4)
        if window_size > 1:
            # Calculate moving average for stability
            overall_smooth = np.convolve(stats["overall_accuracy"], 
                                       np.ones(window_size)/window_size, mode='valid')
            separation_smooth = np.convolve(stats["score_separation"], 
                                          np.ones(window_size)/window_size, mode='valid')
            
            epochs_smooth = list(range(window_size-1, len(stats["overall_accuracy"])))
            axes[1, 1].plot(epochs_smooth, overall_smooth, linewidth=2, color='blue', label='Accuracy (Smoothed)')
            axes[1, 1].plot(epochs_smooth, separation_smooth, linewidth=2, color='purple', label='Separation (Smoothed)')
        else:
            axes[1, 1].plot(stats["overall_accuracy"], linewidth=2, color='blue', label='Overall Accuracy')
            axes[1, 1].plot(stats["score_separation"], linewidth=2, color='purple', label='Score Separation')
        
        axes[1, 1].set_title("Training Stability")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Metric Value")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Verification breakdown plot saved to {save_path}")
        return save_path
    
    def plot_score_analysis(self, stats: Dict[str, List[float]], save_path: Optional[str] = None) -> str:
        """
        Create detailed score analysis plots.
        
        Args:
            stats: Dictionary containing training statistics
            save_path: Optional path to save the plot
            
        Returns:
            Path where the plot was saved
        """
        if save_path is None:
            save_path = os.path.join(self.plots_dir, "score_analysis.png")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Score evolution over time
        axes[0].plot(stats["real_real_mean"], label="Real+Real", linewidth=2, color='green')
        axes[0].plot(stats["fake_fake_mean"], label="Fake+Fake", linewidth=2, color='red')
        axes[0].plot(stats["fake_real_mean"], label="Fake+Real", linewidth=2, color='orange')
        axes[0].plot(stats["real_wrong_mean"], label="Real+Wrong", linewidth=2, color='purple')
        axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Boundary')
        axes[0].set_title("Verification Score Evolution")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Mean Verification Score")
        axes[0].legend()
        axes[0].grid(True)
        
        # Score separation analysis
        axes[1].plot(stats["score_separation"], linewidth=3, color='purple', label='Score Separation')
        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3, label='No Separation')
        axes[1].axhline(y=0.3, color='orange', linestyle='--', label='Good Separation')
        axes[1].axhline(y=0.5, color='green', linestyle='--', label='Excellent Separation')
        axes[1].fill_between(range(len(stats["score_separation"])), 0, stats["score_separation"], 
                            alpha=0.3, color='purple')
        axes[1].set_title("Score Separation (Real vs Fake)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Separation")
        axes[1].legend()
        axes[1].grid(True)
        
        # Final score distribution
        final_scores = [
            stats["real_real_mean"][-1],
            stats["fake_fake_mean"][-1],
            stats["fake_real_mean"][-1], 
            stats["real_wrong_mean"][-1]
        ]
        labels = ['Real+Real', 'Fake+Fake', 'Fake+Real', 'Real+Wrong']
        colors = ['green', 'red', 'orange', 'purple']
        
        bars = axes[2].bar(labels, final_scores, color=colors, alpha=0.7)
        axes[2].axhline(y=0.5, color='gray', linestyle='--', label='Decision Boundary')
        axes[2].set_title("Final Score Distribution")
        axes[2].set_ylabel("Mean Verification Score")
        axes[2].set_ylim(0, 1.05)
        axes[2].legend()
        
        # Add score labels on bars
        for bar, score in zip(bars, final_scores):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Score analysis plot saved to {save_path}")
        return save_path
    
    def create_summary_report(self, stats: Dict[str, List[float]], save_dir: Optional[str] = None) -> List[str]:
        """
        Create a comprehensive visual report with all plots.
        
        Args:
            stats: Dictionary containing training statistics
            save_dir: Optional directory to save plots (defaults to self.plots_dir)
            
        Returns:
            List of paths to saved plots
        """
        if save_dir is None:
            save_dir = self.plots_dir
        
        print("📊 Creating comprehensive visual report...")
        
        saved_plots = []
        
        # Main training progress plot
        plot_path = self.plot_training_progress(stats, os.path.join(save_dir, "training_progress.png"))
        saved_plots.append(plot_path)
        
        # Detailed verification breakdown
        plot_path = self.plot_verification_breakdown(stats, os.path.join(save_dir, "verification_breakdown.png"))
        saved_plots.append(plot_path)
        
        # Score analysis
        plot_path = self.plot_score_analysis(stats, os.path.join(save_dir, "score_analysis.png"))
        saved_plots.append(plot_path)
        
        print(f"✅ Visual report completed! {len(saved_plots)} plots saved to {save_dir}/")
        return saved_plots


def create_training_plots(stats: Dict[str, List[float]], plots_dir: str = "plots") -> List[str]:
    """
    Convenience function to create all training plots.
    
    Args:
        stats: Training statistics dictionary
        plots_dir: Directory to save plots
        
    Returns:
        List of saved plot paths
    """
    plotter = ZKPlotter(plots_dir)
    return plotter.create_summary_report(stats) 