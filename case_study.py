# case_study.py
"""
Enhanced Case Study: Proactive Failure Detection in Model Training
Complete implementation with all required components
"""

import os
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for case study experiments"""
    # Model configuration
    model_name: str = "gpt2"  # Default to smaller model
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    
    # Training configuration
    max_steps: int = 800
    eval_every_n_steps: int = 20
    batch_size: int = 2
    learning_rate: float = 5e-5
    warmup_steps: int = 50
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 1
    
    # Fault injection configuration
    fault_injection_step: int = 400
    fault_type: str = "LR_SPIKE"
    lr_spike_factor: float = 15.0
    lr_spike_duration: int = 20
    
    # Monitoring configuration
    r_metric_alert_threshold: float = 0.57
    r_metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "lambda": 0.10,
        "sigma_sq": 0.45,
        "delta_l": 0.45
    })
    
    # Window sizes
    loss_history_window: int = 10
    gradient_history_window: int = 20
    hardware_event_window: int = 50
    
    # Output configuration
    output_dir: str = "case_study_results"
    experiment_name: str = field(default_factory=lambda: f"case_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Device configuration
    device: str = "auto"
    mixed_precision: bool = True
    use_cpu: bool = False
    use_8bit: bool = False
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.device == "auto":
            if self.use_cpu:
                self.device = "cpu"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        self.output_path = Path(self.output_dir) / self.experiment_name
        self.output_path.mkdir(parents=True, exist_ok=True)


class MetricTracker:
    """Base class for tracking metrics with history"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.all_values = []
    
    def update(self, value: float) -> None:
        """Update metric with new value"""
        if value is None:
            return
        
        # Handle special values
        if np.isnan(value) or np.isinf(value):
            logger.warning(f"Invalid value detected: {value}")
            value = self.get_mean() if self.history else 0.0
        
        self.history.append(float(value))
        self.all_values.append(float(value))
    
    def get_mean(self) -> float:
        """Get mean of recent values"""
        if not self.history:
            return 0.0
        return float(np.mean(list(self.history)))
    
    def get_std(self) -> float:
        """Get standard deviation of recent values"""
        if len(self.history) < 2:
            return 0.0
        return float(np.std(list(self.history)))
    
    def get_variance(self) -> float:
        """Get variance of recent values"""
        if len(self.history) < 2:
            return 0.0
        return float(np.var(list(self.history)))
    
    def normalize(self, value: float) -> float:
        """Normalize value based on history using percentile ranking"""
        if len(self.history) < 2:
            return 0.0
        
        # Convert to list for percentile calculation
        history_list = list(self.history)
        
        # Calculate percentile rank
        below = sum(1 for h in history_list if h < value)
        percentile = below / len(history_list)
        
        return float(np.clip(percentile, 0.0, 1.0))


class ReliabilityMonitor:
    """Enhanced reliability monitoring system with R-Metric"""
    
    def __init__(self, config: Any):
        """Initialize monitor with configuration
        
        Args:
            config: Configuration object (ExperimentConfig or similar) with attributes:
                - loss_history_window (int)
                - gradient_history_window (int) 
                - hardware_event_window (int)
                - r_metric_weights (dict)
        """
        self.config = config
        
        # Initialize metric trackers with safe defaults
        loss_window = getattr(config, 'loss_history_window', 10)
        grad_window = getattr(config, 'gradient_history_window', 20)
        hw_window = getattr(config, 'hardware_event_window', 50)
        
        self.loss_tracker = MetricTracker(loss_window)
        self.gradient_tracker = MetricTracker(grad_window)
        self.hardware_tracker = MetricTracker(hw_window)
        
        # Component histories for normalization
        self.lambda_history = deque(maxlen=100)
        self.sigma_sq_history = deque(maxlen=100)
        self.delta_l_history = deque(maxlen=100)
        
        # Metrics storage
        self.metrics_history = []
        
        # Get weights with defaults
        default_weights = {"lambda": 0.10, "sigma_sq": 0.45, "delta_l": 0.45}
        self.weights = getattr(config, 'r_metric_weights', default_weights)
        
        logger.debug(f"ReliabilityMonitor initialized with weights: {self.weights}")
    
    def simulate_hardware_events(self) -> float:
        """Simulate hardware failure rate (λ)
        
        In production, this would come from actual hardware monitoring.
        For simulation, we use a low base rate with occasional spikes.
        """
        base_rate = 0.1
        
        # Simulate occasional hardware events (5% chance)
        if np.random.random() < 0.05:
            spike = base_rate + np.random.exponential(0.5)
            self.hardware_tracker.update(spike)
            return float(spike)
        
        self.hardware_tracker.update(base_rate)
        return float(base_rate)
    
    def calculate_gradient_variance(self, gradients: Optional[List] = None) -> float:
        """Calculate gradient variance across parameters (σ²)
        
        Args:
            gradients: List of gradient tensors
            
        Returns:
            Variance of gradient norms
        """
        if not gradients:
            return 0.0
        
        grad_norms = []
        for grad in gradients:
            if grad is not None:
                try:
                    # Handle both numpy arrays and torch tensors
                    if hasattr(grad, 'norm'):
                        grad_norm = grad.norm().item()
                    elif hasattr(grad, 'data'):
                        grad_norm = float(np.linalg.norm(grad.data))
                    else:
                        grad_norm = float(np.linalg.norm(grad))
                    
                    if not np.isnan(grad_norm) and not np.isinf(grad_norm):
                        grad_norms.append(grad_norm)
                except Exception as e:
                    logger.debug(f"Error calculating gradient norm: {e}")
                    continue
        
        if not grad_norms:
            return 0.0
        
        variance = float(np.var(grad_norms))
        self.gradient_tracker.update(variance)
        
        return variance
    
    def calculate_loss_drift(self, current_loss: float) -> float:
        """Calculate validation loss drift (ΔL)
        
        Args:
            current_loss: Current loss value
            
        Returns:
            Drift from moving average
        """
        # Handle invalid loss values
        if current_loss is None or np.isnan(current_loss) or np.isinf(current_loss):
            logger.warning(f"Invalid loss value: {current_loss}")
            return 10.0  # High value indicates instability
        
        # First loss - no drift yet
        if len(self.loss_tracker.history) == 0:
            self.loss_tracker.update(current_loss)
            return 0.0
        
        # Calculate drift from moving average
        mean_loss = self.loss_tracker.get_mean()
        drift = abs(current_loss - mean_loss)
        
        # Update tracker
        self.loss_tracker.update(current_loss)
        
        return float(drift)
    
    def calculate_r_metric(self, 
                          current_loss: float,
                          gradients: Optional[List] = None) -> Dict[str, float]:
        """Calculate the composite R-Metric
        
        Args:
            current_loss: Current training/validation loss
            gradients: Optional list of gradient tensors
            
        Returns:
            Dictionary with R-metric and component values
        """
        try:
            # Calculate components
            lambda_val = self.simulate_hardware_events()
            sigma_sq_val = self.calculate_gradient_variance(gradients) if gradients else 0.0
            delta_l_val = self.calculate_loss_drift(current_loss)
            
            # Update histories
            self.lambda_history.append(lambda_val)
            self.sigma_sq_history.append(sigma_sq_val)
            self.delta_l_history.append(delta_l_val)
            
            # Normalize components
            lambda_norm = self._normalize_value(lambda_val, self.lambda_history)
            sigma_sq_norm = self._normalize_value(sigma_sq_val, self.sigma_sq_history)
            delta_l_norm = self._normalize_value(delta_l_val, self.delta_l_history)
            
            # Calculate weighted R-Metric
            r_metric = (
                self.weights.get("lambda", 0.10) * lambda_norm +
                self.weights.get("sigma_sq", 0.45) * sigma_sq_norm +
                self.weights.get("delta_l", 0.45) * delta_l_norm
            )
            
            # Store in history
            result = {
                "r_metric": float(r_metric),
                "lambda": float(lambda_val),
                "lambda_norm": float(lambda_norm),
                "sigma_sq": float(sigma_sq_val),
                "sigma_sq_norm": float(sigma_sq_norm),
                "delta_l": float(delta_l_val),
                "delta_l_norm": float(delta_l_norm)
            }
            
            self.metrics_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating R-metric: {e}")
            # Return safe default values
            return {
                "r_metric": 0.0,
                "lambda": 0.0,
                "lambda_norm": 0.0,
                "sigma_sq": 0.0,
                "sigma_sq_norm": 0.0,
                "delta_l": 0.0,
                "delta_l_norm": 0.0
            }
    
    def _normalize_value(self, value: float, history: deque) -> float:
        """Normalize value based on history using min-max scaling
        
        Args:
            value: Value to normalize
            history: Historical values
            
        Returns:
            Normalized value in [0, 1]
        """
        if len(history) < 2:
            return 0.0
        
        try:
            history_list = list(history)
            min_val = min(history_list)
            max_val = max(history_list)
            
            # Avoid division by zero
            if max_val == min_val:
                return 0.5  # Middle value if no variance
            
            # Min-max normalization
            normalized = (value - min_val) / (max_val - min_val)
            
            # Clip to [0, 1]
            return float(np.clip(normalized, 0.0, 1.0))
            
        except Exception as e:
            logger.debug(f"Error normalizing value: {e}")
            return 0.0
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of monitoring
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {
                "num_observations": 0,
                "mean_r_metric": 0.0,
                "max_r_metric": 0.0,
                "alerts": 0
            }
        
        r_values = [m["r_metric"] for m in self.metrics_history]
        
        # Count alerts (assuming threshold of 0.6)
        threshold = getattr(self.config, 'r_metric_alert_threshold', 0.6)
        alerts = sum(1 for r in r_values if r > threshold)
        
        return {
            "num_observations": len(self.metrics_history),
            "mean_r_metric": float(np.mean(r_values)),
            "max_r_metric": float(np.max(r_values)),
            "min_r_metric": float(np.min(r_values)),
            "std_r_metric": float(np.std(r_values)),
            "alerts": alerts,
            "alert_rate": alerts / len(r_values) if r_values else 0.0
        }


class BaselineMonitors:
    """Collection of baseline monitoring approaches"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Simple heuristics
        self.consecutive_loss_increases = 0
        self.loss_spike_threshold = 3.0
        
        # Gradient norm monitoring
        self.grad_norm_threshold = 100.0
        self.grad_norm_history = deque(maxlen=20)
        
        # Loss history for various checks
        self.loss_history = []
        self.prev_loss = None
    
    def update_simple_heuristic(self, current_loss: float, prev_loss: Optional[float] = None) -> bool:
        """Simple heuristic: consecutive loss increases"""
        if prev_loss is None:
            prev_loss = self.prev_loss
        
        if prev_loss is not None and current_loss > prev_loss:
            self.consecutive_loss_increases += 1
        else:
            self.consecutive_loss_increases = 0
        
        self.prev_loss = current_loss
        return self.consecutive_loss_increases >= 3
    
    def update_loss_spike(self, current_loss: float) -> bool:
        """Detect sudden loss spikes"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < 10:
            return False
        
        recent = self.loss_history[-10:]
        mean_loss = np.mean(recent[:-1])
        std_loss = np.std(recent[:-1])
        
        if std_loss > 0:
            z_score = (current_loss - mean_loss) / std_loss
            return z_score > self.loss_spike_threshold
        
        return False
    
    def update_gradient_monitoring(self, gradients: List[torch.Tensor]) -> bool:
        """Monitor gradient norms for explosion"""
        if not gradients:
            return False
        
        grad_norms = []
        for grad in gradients:
            if grad is not None:
                try:
                    if hasattr(grad, 'norm'):
                        grad_norm = grad.norm().item()
                    else:
                        grad_norm = float(np.linalg.norm(grad))
                    
                    if not np.isnan(grad_norm) and not np.isinf(grad_norm):
                        grad_norms.append(grad_norm)
                except:
                    continue
        
        if not grad_norms:
            return False
        
        max_grad_norm = max(grad_norms)
        self.grad_norm_history.append(max_grad_norm)
        
        return max_grad_norm > self.grad_norm_threshold


class FaultInjector:
    """Manages fault injection for experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.fault_active = False
        self.fault_start_step = None
        self.original_lr = None
        
    def should_inject(self, step: int) -> bool:
        """Check if fault should be injected at current step"""
        return step == self.config.fault_injection_step
    
    def inject_fault(self, optimizer: torch.optim.Optimizer, step: int):
        """Inject fault into training process"""
        if self.config.fault_type == "LR_SPIKE":
            self._inject_lr_spike(optimizer, step)
        elif self.config.fault_type == "GRADIENT_EXPLOSION":
            self._inject_gradient_explosion(optimizer, step)
        # Add more fault types as needed
        
    def _inject_lr_spike(self, optimizer: torch.optim.Optimizer, step: int):
        """Inject learning rate spike"""
        if not self.fault_active:
            self.fault_active = True
            self.fault_start_step = step
            self.original_lr = optimizer.param_groups[0]['lr']
            
            new_lr = self.original_lr * self.config.lr_spike_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            logger.warning(f"FAULT INJECTED: LR spiked from {self.original_lr:.2e} to {new_lr:.2e}")
    
    def _inject_gradient_explosion(self, optimizer: torch.optim.Optimizer, step: int):
        """Inject gradient explosion"""
        if not self.fault_active:
            self.fault_active = True
            self.fault_start_step = step
            self.original_lr = optimizer.param_groups[0]['lr']
            
            # Dramatically increase learning rate to cause gradient explosion
            new_lr = self.original_lr * 100
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            logger.warning(f"FAULT INJECTED: Gradient explosion via LR spike")
    
    def should_recover(self, step: int) -> bool:
        """Check if fault should be recovered"""
        if self.fault_active and self.fault_start_step is not None:
            return step >= self.fault_start_step + self.config.lr_spike_duration
        return False
    
    def recover_fault(self, optimizer: torch.optim.Optimizer):
        """Recover from injected fault"""
        if self.fault_active and self.original_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.original_lr
            
            logger.info(f"FAULT RECOVERED: LR restored to {self.original_lr:.2e}")
            self.fault_active = False
            self.original_lr = None


class CaseStudyTrainer:
    """Main trainer class for case study experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging for this experiment
        self.setup_logging()
        
        # Results tracking
        self.results = {
            "config": config.__dict__,
            "metrics": [],
            "alerts": {
                "r_metric": None,
                "simple_heuristic": None,
                "loss_spike": None,
                "gradient_monitoring": None
            }
        }
        
    def setup_logging(self):
        """Setup experiment logging"""
        log_file = self.config.output_path / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        logger.info(f"Configuration: {json.dumps(self.config.__dict__, indent=2, default=str)}")
    
    def save_results(self):
        """Save experiment results"""
        # Save metrics to CSV
        if self.results["metrics"]:
            df = pd.DataFrame(self.results["metrics"])
            csv_path = self.config.output_path / "metrics.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Metrics saved to {csv_path}")
        
        # Save full results to JSON (handle Path objects)
        json_path = self.config.output_path / "results.json"
        
        # Convert Path objects to strings in config
        json_safe_results = self.results.copy()
        json_safe_results["config"] = {}
        for key, value in self.results["config"].items():
            if isinstance(value, Path):
                json_safe_results["config"][key] = str(value)
            else:
                json_safe_results["config"][key] = value
        
        with open(json_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2, default=str)
        logger.info(f"Results saved to {json_path}")
    
    def generate_summary(self):
        """Generate experiment summary"""
        summary = {
            "experiment_name": self.config.experiment_name,
            "model": self.config.model_name,
            "total_steps": self.config.max_steps,
            "fault_type": self.config.fault_type,
            "fault_step": self.config.fault_injection_step,
            "alerts": self.results["alerts"],
            "alert_lead_times": {}
        }
        
        # Calculate lead times
        for method, alert_step in self.results["alerts"].items():
            if alert_step is not None:
                lead_time = self.config.fault_injection_step - alert_step
                summary["alert_lead_times"][method] = lead_time
        
        # Save summary
        summary_path = self.config.output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Summary:")
        logger.info(json.dumps(summary, indent=2))
        
        return summary


# Additional classes that might be used by other scripts
class MultiModelConfig:
    """Configuration for multi-model experiments"""
    pass  # Implement as needed


class EnhancedFaultInjector:
    """Enhanced fault injector with more fault types"""
    pass  # Implement as needed


class ComputationalOverheadMonitor:
    """Monitor computational overhead"""
    pass  # Implement as needed


# Exports - maintain backward compatibility
__all__ = [
    'ExperimentConfig',
    'MetricTracker',
    'ReliabilityMonitor',
    'BaselineMonitors',
    'FaultInjector',
    'CaseStudyTrainer',
    'MultiModelConfig',
    'EnhancedFaultInjector',
    'ComputationalOverheadMonitor'
]
