# case_study.py
"""
Updated Case Study Module for R-Metric
Production-ready implementation with proper error handling
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


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
            config: Configuration object with required attributes:
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
        
        logger.info(f"ReliabilityMonitor initialized with weights: {self.weights}")
    
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
                    logger.warning(f"Error calculating gradient norm: {e}")
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
            logger.warning(f"Error normalizing value: {e}")
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


# Backward compatibility exports
__all__ = ['ReliabilityMonitor', 'MetricTracker']