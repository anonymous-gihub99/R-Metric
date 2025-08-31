#!/usr/bin/env python3
"""
Debugging Version of Scaling Validation Script for R-Metric
This version includes enhanced error reporting and step-by-step diagnostics
"""

import os
import sys
import json
import time
import logging
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm

# Configure enhanced logging
logging.basicConfig(
    level=logging.DEBUG,  # More verbose logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scaling_validation_debug.log')
    ]
)
logger = logging.getLogger(__name__)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special types"""
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if torch.is_tensor(obj):
            return obj.cpu().numpy().tolist()
        return super().default(obj)


@dataclass
class ScalingConfig:
    """Configuration for scaling experiments with debug settings"""
    # Model configuration
    model_name: str = "gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    
    # Training configuration - even more conservative for debugging
    batch_size: int = 1
    max_steps: int = 50  # Very short for debugging
    eval_every_n_steps: int = 5  # More frequent evaluation
    learning_rate: float = 1e-5  # Lower learning rate
    warmup_steps: int = 5
    max_seq_length: int = 64  # Even shorter sequences
    gradient_accumulation_steps: int = 2  # Smaller accumulation
    
    # Fault injection - delayed for debugging
    fault_injection_step: int = 30
    fault_type: str = "LR_SPIKE"
    fault_severity: float = 5.0  # Less severe
    
    # R-Metric configuration
    r_metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "lambda": 0.10,
        "sigma_sq": 0.45,
        "delta_l": 0.70
    })
    r_metric_alert_threshold: float = 0.6
    
    # Window sizes for ReliabilityMonitor
    loss_history_window: int = 10
    gradient_history_window: int = 20
    hardware_event_window: int = 50
    
    # Output configuration
    output_dir: str = "scaling_validation_debug"
    experiment_name: str = field(default_factory=lambda: f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Device configuration
    device: str = "auto"
    mixed_precision: bool = False
    use_cpu: bool = False
    
    def __post_init__(self):
        """Initialize paths and device with validation"""
        try:
            self.output_path = Path(self.output_dir) / self.experiment_name
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            if self.device == "auto":
                if self.use_cpu:
                    self.device = "cpu"
                else:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Config initialized: device={self.device}, output={self.output_path}")
        except Exception as e:
            logger.error(f"Config initialization failed: {e}")
            raise
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with proper serialization"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict


class DebugReliabilityMonitor:
    """Enhanced reliability monitor with debugging"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.loss_history = []
        self.gradient_norms = []
        self.r_metric_history = []
        logger.debug("ReliabilityMonitor initialized")
        
    def calculate_r_metric(self, loss: float, gradients: Optional[List] = None) -> Dict[str, float]:
        """Calculate R-metric with detailed logging"""
        try:
            logger.debug(f"Calculating R-metric for loss={loss:.6f}")
            
            # Validate loss
            if not isinstance(loss, (int, float)) or np.isnan(loss) or np.isinf(loss):
                logger.warning(f"Invalid loss value: {loss}")
                return self._default_r_metric()
            
            self.loss_history.append(loss)
            
            # Calculate components with validation
            lambda_val = np.random.uniform(0.05, 0.15)  # Simulated hardware events
            
            # Gradient variance with safety checks
            sigma_sq_val = 0.0
            if gradients:
                try:
                    grad_norms = []
                    for g in gradients:
                        if g is not None and torch.is_tensor(g):
                            norm = g.norm().item()
                            if not (np.isnan(norm) or np.isinf(norm)):
                                grad_norms.append(norm)
                    
                    if grad_norms:
                        sigma_sq_val = np.var(grad_norms)
                        self.gradient_norms.extend(grad_norms)
                        logger.debug(f"Gradient norms: {len(grad_norms)} valid, var={sigma_sq_val:.6f}")
                except Exception as e:
                    logger.warning(f"Gradient processing error: {e}")
                    sigma_sq_val = 0.0
            
            # Loss drift with safety checks
            delta_l_val = 0.0
            if len(self.loss_history) > 1:
                try:
                    recent_losses = self.loss_history[-5:]
                    mean_loss = np.mean(recent_losses)
                    if not (np.isnan(mean_loss) or np.isinf(mean_loss)):
                        delta_l_val = abs(loss - mean_loss)
                    logger.debug(f"Loss drift: current={loss:.6f}, mean={mean_loss:.6f}, delta={delta_l_val:.6f}")
                except Exception as e:
                    logger.warning(f"Loss drift calculation error: {e}")
                    delta_l_val = 0.0
            
            # Normalize components with safety bounds
            lambda_norm = min(max(lambda_val * 10, 0.0), 1.0)
            sigma_sq_norm = min(max(sigma_sq_val / 100, 0.0), 1.0) if sigma_sq_val > 0 else 0.0
            delta_l_norm = min(max(delta_l_val / 2, 0.0), 1.0) if delta_l_val > 0 else 0.0
            
            # Calculate R-metric
            weights = self.config.r_metric_weights
            r_metric = (
                weights["lambda"] * lambda_norm +
                weights["sigma_sq"] * sigma_sq_norm +
                weights["delta_l"] * delta_l_norm
            )
            
            # Validate R-metric
            if np.isnan(r_metric) or np.isinf(r_metric):
                logger.warning("Invalid R-metric calculated, using default")
                r_metric = 0.0
            
            r_metric = max(0.0, min(1.0, r_metric))  # Clamp to [0,1]
            self.r_metric_history.append(r_metric)
            
            result = {
                "r_metric": float(r_metric),
                "lambda": float(lambda_val),
                "lambda_norm": float(lambda_norm),
                "sigma_sq": float(sigma_sq_val),
                "sigma_sq_norm": float(sigma_sq_norm),
                "delta_l": float(delta_l_val),
                "delta_l_norm": float(delta_l_norm)
            }
            
            logger.debug(f"R-metric calculated: {r_metric:.6f}")
            return result
            
        except Exception as e:
            logger.error(f"R-metric calculation failed: {e}")
            logger.error(traceback.format_exc())
            return self._default_r_metric()
    
    def _default_r_metric(self) -> Dict[str, float]:
        """Return default R-metric values"""
        return {
            "r_metric": 0.0,
            "lambda": 0.0,
            "lambda_norm": 0.0,
            "sigma_sq": 0.0,
            "sigma_sq_norm": 0.0,
            "delta_l": 0.0,
            "delta_l_norm": 0.0
        }


class DebugScalingValidator:
    """Enhanced validator with step-by-step debugging"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results = []
        
        logger.info(f"Initialized DebugScalingValidator")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output path: {config.output_path}")
        
        # Test basic torch operations
        self._test_torch_setup()
        
    def _test_torch_setup(self):
        """Test basic PyTorch functionality"""
        try:
            logger.info("Testing PyTorch setup...")
            
            # Test device
            test_tensor = torch.randn(2, 2).to(self.device)
            logger.info(f"✅ Device test passed: {test_tensor.device}")
            
            # Test memory
            if torch.cuda.is_available():
                memory_free = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"✅ GPU memory available: {memory_free:.1f} GB")
            
            del test_tensor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"❌ PyTorch setup test failed: {e}")
            raise
        
    def detect_gpu_configuration(self) -> Dict[str, Any]:
        """Detect current GPU setup with enhanced logging"""
        try:
            if not torch.cuda.is_available():
                logger.info("No CUDA available, using CPU")
                return {
                    "type": "CPU",
                    "count": 0,
                    "names": [],
                    "total_memory_gb": 0
                }
            
            gpu_count = torch.cuda.device_count()
            gpu_names = []
            gpu_memory = []
            
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                gpu_names.append(name)
                gpu_memory.append(memory)
                logger.info(f"GPU {i}: {name} ({memory:.1f} GB)")
            
            # Determine configuration type
            if gpu_count == 1:
                if "L4" in gpu_names[0]:
                    config_type = "1xL4"
                elif "T4" in gpu_names[0]:
                    config_type = "1xT4"
                else:
                    config_type = f"1x{gpu_names[0].split()[0]}"
            else:
                config_type = f"{gpu_count}xGPU"
            
            return {
                "type": config_type,
                "count": gpu_count,
                "names": gpu_names,
                "memory_gb": gpu_memory,
                "total_memory_gb": sum(gpu_memory)
            }
            
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            return {"type": "ERROR", "count": 0, "names": [], "total_memory_gb": 0}
    
    def load_model_and_data(self) -> Tuple:
        """Load model and dataset with enhanced error handling"""
        model = None
        tokenizer = None
        dataloader = None
        
        try:
            logger.info("="*50)
            logger.info("LOADING MODEL AND DATA")
            logger.info("="*50)
            
            # Step 1: Load tokenizer
            logger.info(f"Step 1: Loading tokenizer: {self.config.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=None,
                trust_remote_code=False
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            logger.info(f"✅ Tokenizer loaded successfully")
            logger.info(f"   Vocab size: {tokenizer.vocab_size}")
            logger.info(f"   Pad token: {tokenizer.pad_token}")
            
            # Step 2: Load model
            logger.info(f"Step 2: Loading model: {self.config.model_name}")
            
            # More conservative model loading
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                device_map=None,  # Manual device placement
                trust_remote_code=False,
                use_cache=False,
                low_cpu_mem_usage=True
            )
            
            # Manual device placement
            logger.info(f"Moving model to device: {self.device}")
            model = model.to(self.device)
            model.train()
            
            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Parameters: {param_count:.1f}M")
            logger.info(f"   Device: {next(model.parameters()).device}")
            
            # Step 3: Load dataset
            logger.info(f"Step 3: Loading dataset: {self.config.dataset_name}")
            
            try:
                dataset = load_dataset(
                    self.config.dataset_name,
                    self.config.dataset_config,
                    split='train',
                    streaming=False,
                    trust_remote_code=False
                )
                logger.info(f"✅ Dataset loaded: {len(dataset)} samples")
            except Exception as e:
                logger.error(f"❌ Dataset loading failed: {e}")
                # Try alternative dataset
                logger.info("Trying alternative dataset...")
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')
                logger.info(f"✅ Alternative dataset loaded: {len(dataset)} samples")
            
            # Limit dataset size for debugging
            dataset_size = min(100, len(dataset))
            dataset = dataset.select(range(dataset_size))
            logger.info(f"Dataset limited to {dataset_size} samples for debugging")
            
            # Step 4: Tokenize dataset
            logger.info("Step 4: Tokenizing dataset")
            
            def tokenize_function(examples):
                try:
                    # Handle both single text and batch
                    texts = examples['text'] if isinstance(examples['text'], list) else [examples['text']]
                    
                    # Filter out empty or None texts
                    texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 0]
                    
                    if not texts:
                        logger.warning("No valid texts found in batch")
                        return {
                            'input_ids': torch.zeros(1, self.config.max_seq_length, dtype=torch.long),
                            'attention_mask': torch.zeros(1, self.config.max_seq_length, dtype=torch.long)
                        }
                    
                    return tokenizer(
                        texts,
                        truncation=True,
                        padding='max_length',
                        max_length=self.config.max_seq_length,
                        return_tensors='pt'
                    )
                except Exception as e:
                    logger.error(f"Tokenization error: {e}")
                    return {
                        'input_ids': torch.zeros(1, self.config.max_seq_length, dtype=torch.long),
                        'attention_mask': torch.zeros(1, self.config.max_seq_length, dtype=torch.long)
                    }
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                batch_size=10  # Smaller batches for tokenization
            )
            
            logger.info(f"✅ Tokenization complete: {len(tokenized_dataset)} samples")
            
            # Step 5: Create dataloader
            logger.info("Step 5: Creating dataloader")
            
            tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            
            dataloader = DataLoader(
                tokenized_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,  # No multiprocessing for debugging
                pin_memory=False,  # Disable pin memory
                drop_last=True
            )
            
            logger.info(f"✅ DataLoader created: {len(dataloader)} batches")
            
            # Test one batch
            logger.info("Step 6: Testing first batch")
            test_batch = next(iter(dataloader))
            logger.info(f"✅ First batch loaded successfully")
            logger.info(f"   Batch keys: {list(test_batch.keys())}")
            logger.info(f"   Input shape: {test_batch['input_ids'].shape}")
            logger.info(f"   Attention shape: {test_batch['attention_mask'].shape}")
            
            return model, tokenizer, dataloader
            
        except Exception as e:
            logger.error(f"❌ Model/data loading failed: {e}")
            logger.error(traceback.format_exc())
            
            # Cleanup on failure
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            raise
    
    def run_experiment(self, weight_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Run experiment with extensive debugging"""
        experiment_start_time = time.time()
        
        try:
            logger.info("\n" + "="*60)
            logger.info("STARTING EXPERIMENT")
            logger.info("="*60)
            
            if weight_config:
                self.config.r_metric_weights = weight_config
                logger.info(f"Using weights: λ={weight_config['lambda']:.3f}, "
                          f"σ²={weight_config['sigma_sq']:.3f}, "
                          f"ΔL={weight_config['delta_l']:.3f}")
            
            # Load model and data
            logger.info("Loading model and data...")
            model, tokenizer, dataloader = self.load_model_and_data()
            
            # Setup optimizer
            logger.info("Setting up optimizer...")
            optimizer = AdamW(
                model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps
            )
            
            logger.info(f"✅ Optimizer setup complete")
            logger.info(f"   Learning rate: {self.config.learning_rate}")
            logger.info(f"   Warmup steps: {self.config.warmup_steps}")
            
            # Initialize monitoring
            monitor = DebugReliabilityMonitor(self.config)
            
            # Training metrics
            metrics_history = []
            first_alert_step = None
            fault_injected = False
            crash_step = None
            
            logger.info("\n" + "="*60)
            logger.info("STARTING TRAINING LOOP")
            logger.info("="*60)
            
            # Training loop with enhanced error handling
            data_iter = iter(dataloader)
            
            for step in tqdm(range(self.config.max_steps), desc="Training"):
                step_start_time = time.time()
                
                try:
                    logger.debug(f"--- Step {step} ---")
                    
                    # Get batch with retry logic
                    batch = None
                    for attempt in range(3):
                        try:
                            batch = next(data_iter)
                            break
                        except StopIteration:
                            logger.debug(f"Restarting data iterator (attempt {attempt+1})")
                            data_iter = iter(dataloader)
                            batch = next(data_iter)
                            break
                        except Exception as e:
                            logger.warning(f"Batch loading attempt {attempt+1} failed: {e}")
                            if attempt == 2:
                                raise
                    
                    if batch is None:
                        logger.error("Failed to load batch after 3 attempts")
                        crash_step = step
                        break
                    
                    # Move to device with error handling
                    try:
                        batch = {k: v.to(self.device, non_blocking=False) for k, v in batch.items()}
                        logger.debug(f"Batch moved to {self.device}")
                    except Exception as e:
                        logger.error(f"Failed to move batch to device: {e}")
                        crash_step = step
                        break
                    
                    # Inject fault
                    if step == self.config.fault_injection_step and not fault_injected:
                        logger.info(f"🔥 INJECTING FAULT at step {step}")
                        original_lr = optimizer.param_groups[0]['lr']
                        new_lr = original_lr * self.config.fault_severity
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        logger.info(f"Learning rate: {original_lr:.6f} -> {new_lr:.6f}")
                        fault_injected = True
                    
                    # Forward pass with detailed error handling
                    try:
                        logger.debug("Forward pass...")
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['input_ids']
                        )
                        loss = outputs.loss
                        logger.debug(f"Forward pass complete, loss: {loss.item():.6f}")
                        
                    except Exception as e:
                        logger.error(f"Forward pass failed: {e}")
                        crash_step = step
                        break
                    
                    # Check for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(f"❌ NaN/Inf loss detected at step {step}: {loss.item()}")
                        crash_step = step
                        break
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass
                    try:
                        logger.debug("Backward pass...")
                        scaled_loss.backward()
                        logger.debug("Backward pass complete")
                    except Exception as e:
                        logger.error(f"Backward pass failed: {e}")
                        crash_step = step
                        break
                    
                    # Gradient accumulation
                    gradients = None
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        try:
                            logger.debug("Gradient step...")
                            
                            # Get gradients for monitoring
                            gradients = []
                            total_norm = 0.0
                            for p in model.parameters():
                                if p.grad is not None:
                                    gradients.append(p.grad.clone())
                                    total_norm += p.grad.norm().item() ** 2
                            total_norm = total_norm ** 0.5
                            
                            logger.debug(f"Total gradient norm: {total_norm:.6f}")
                            
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            # Optimizer step
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            
                            logger.debug("Gradient step complete")
                            
                        except Exception as e:
                            logger.error(f"Gradient step failed: {e}")
                            crash_step = step
                            break
                    
                    # Monitoring every N steps
                    if step % self.config.eval_every_n_steps == 0:
                        try:
                            logger.debug("Calculating R-metric...")
                            
                            # Calculate R-metric
                            r_metric_results = monitor.calculate_r_metric(
                                loss.item(),
                                gradients
                            )
                            
                            # Check for alert
                            if r_metric_results['r_metric'] > self.config.r_metric_alert_threshold:
                                if first_alert_step is None:
                                    first_alert_step = step
                                    logger.info(f"🚨 R-Metric ALERT at step {step}: {r_metric_results['r_metric']:.3f}")
                            
                            # Store metrics
                            metrics_history.append({
                                'step': step,
                                'loss': float(loss.item()),
                                'learning_rate': float(scheduler.get_last_lr()[0]),
                                'step_time': time.time() - step_start_time,
                                **r_metric_results
                            })
                            
                            # Log progress
                            logger.info(f"Step {step:3d}: Loss={loss.item():.4f}, "
                                      f"R-Metric={r_metric_results['r_metric']:.3f}, "
                                      f"LR={scheduler.get_last_lr()[0]:.2e}")
                            
                        except Exception as e:
                            logger.error(f"Monitoring failed at step {step}: {e}")
                            # Continue training even if monitoring fails
                            metrics_history.append({
                                'step': step,
                                'loss': float(loss.item()),
                                'error': str(e)
                            })
                    
                except Exception as e:
                    logger.error(f"❌ Training step {step} failed: {e}")
                    logger.error(traceback.format_exc())
                    crash_step = step
                    break
            
            # Calculate final results
            logger.info("\n" + "="*60)
            logger.info("EXPERIMENT COMPLETE")
            logger.info("="*60)
            
            lead_time = None
            if first_alert_step is not None and fault_injected:
                lead_time = self.config.fault_injection_step - first_alert_step
                logger.info(f"✅ Alert detected with {lead_time} steps lead time")
            elif first_alert_step is not None:
                logger.info(f"Alert detected at step {first_alert_step} (no fault injected yet)")
            else:
                logger.info("No alerts detected")
            
            if crash_step is not None:
                logger.info(f"❌ Training crashed at step {crash_step}")
            
            # Get GPU info
            gpu_config = self.detect_gpu_configuration()
            
            result = {
                'gpu_config': gpu_config['type'],
                'gpu_count': gpu_config['count'],
                'weight_lambda': float(self.config.r_metric_weights['lambda']),
                'weight_sigma_sq': float(self.config.r_metric_weights['sigma_sq']),
                'weight_delta_l': float(self.config.r_metric_weights['delta_l']),
                'first_alert_step': first_alert_step,
                'fault_injection_step': self.config.fault_injection_step if fault_injected else None,
                'crash_step': crash_step,
                'lead_time_steps': lead_time,
                'total_steps_completed': step + 1,
                'fault_injected': fault_injected,
                'metrics_history': metrics_history,
                'experiment_duration': time.time() - experiment_start_time,
                'timestamp': datetime.now().isoformat(),
                'success': crash_step is None
            }
            
            logger.info(f"Experiment summary:")
            logger.info(f"  - Total steps: {step + 1}/{self.config.max_steps}")
            logger.info(f"  - Success: {result['success']}")
            logger.info(f"  - Duration: {result['experiment_duration']:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ EXPERIMENT FAILED: {e}")
            logger.error(traceback.format_exc())
            
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat(),
                'experiment_duration': time.time() - experiment_start_time,
                'success': False
            }
        
        finally:
            # Cleanup
            logger.info("Cleaning up...")
            try:
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("✅ Cleanup complete")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
    
    def run_single_test(self) -> Dict:
        """Run a single test experiment"""
        logger.info("Running single test experiment...")
        return self.run_experiment()
    
    def run_weight_variations(self) -> List[Dict]:
        """Test multiple weight configurations with debugging"""
        weight_configs = [
            {"lambda": 0.10, "sigma_sq": 0.45, "delta_l": 0.70},  # Default
            {"lambda": 0.15, "sigma_sq": 0.40, "delta_l": 0.45},  # Variant 1  
            {"lambda": 0.10, "sigma_sq": 0.40, "delta_l": 0.50},  # Variant 2
        ]
        
        all_results = []
        
        for i, weights in enumerate(weight_configs):
            logger.info(f"\n" + "="*80)
            logger.info(f"EXPERIMENT {i+1}/{len(weight_configs)}")
            logger.info("="*80)
            
            try:
                result = self.run_experiment(weights)
                result['experiment_id'] = i
                all_results.append(result)
                
                # Log experiment result
                if result.get('success', False):
                    logger.info(f"✅ Experiment {i+1} completed successfully")
                else:
                    logger.error(f"❌ Experiment {i+1} failed")
                    if 'error' in result:
                        logger.error(f"   Error: {result['error']}")
                
                # Save intermediate results
                self.save_results(all_results)
                
                # Brief pause between experiments
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Experiment {i+1} crashed: {e}")
                error_result = {
                    'experiment_id': i,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'timestamp': datetime.now().isoformat(),
                    'success': False
                }
                all_results.append(error_result)
        
        return all_results
    
    def save_results(self, results: List[Dict]):
        """Save results with enhanced error handling"""
        try:
            logger.info("Saving results...")
            
            # Ensure output directory exists
            self.config.output_path.mkdir(parents=True, exist_ok=True)
            
            # Save detailed JSON
            json_path = self.config.output_path / "scaling_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, cls=JSONEncoder)
            logger.info(f"✅ Detailed results saved to {json_path}")
            
            # Save configuration
            config_path = self.config.output_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2, cls=JSONEncoder)
            logger.info(f"✅ Configuration saved to {config_path}")
            
            # Save summary CSV
            summary_data = []
            for r in results:
                if 'error' not in r:
                    summary_data.append({
                        'experiment_id': r.get('experiment_id', 0),
                        'gpu_config': r['gpu_config'],
                        'weight_lambda': r['weight_lambda'],
                        'weight_sigma_sq': r['weight_sigma_sq'], 
                        'weight_delta_l': r['weight_delta_l'],
                        'first_alert_step': r['first_alert_step'],
                        'fault_injection_step': r['fault_injection_step'],
                        'crash_step': r['crash_step'],
                        'lead_time_steps': r['lead_time_steps'],
                        'total_steps': r['total_steps_completed'],
                        'success': r['success'],
                        'duration': r['experiment_duration']
                    })
                else:
                    summary_data.append({
                        'experiment_id': r.get('experiment_id', 0),
                        'error': r['error'],
                        'success': False
                    })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                csv_path = self.config.output_path / "scaling_summary.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"✅ Summary saved to {csv_path}")
            
            # Save error log if any failures
            failed_experiments = [r for r in results if not r.get('success', False)]
            if failed_experiments:
                error_path = self.config.output_path / "errors.json"
                with open(error_path, 'w') as f:
                    json.dump(failed_experiments, f, indent=2, cls=JSONEncoder)
                logger.info(f"⚠️  Error details saved to {error_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            logger.error(traceback.format_exc())


def run_diagnostics():
    """Run system diagnostics before main experiment"""
    logger.info("\n" + "="*60)
    logger.info("SYSTEM DIAGNOSTICS")
    logger.info("="*60)
    
    try:
        # Python version
        logger.info(f"Python version: {sys.version}")
        
        # PyTorch version
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # CUDA info
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)")
        else:
            logger.info("CUDA not available")
        
        # Memory info
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"System RAM: {memory.total/1e9:.1f} GB ({memory.percent}% used)")
        
        # Transformers version
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Datasets version  
        import datasets
        logger.info(f"Datasets version: {datasets.__version__}")
        
        logger.info("✅ Diagnostics complete")
        
    except Exception as e:
        logger.error(f"❌ Diagnostics failed: {e}")


def test_minimal_experiment():
    """Run a minimal test to isolate issues"""
    logger.info("\n" + "="*60)
    logger.info("MINIMAL TEST EXPERIMENT")
    logger.info("="*60)
    
    try:
        # Very minimal config
        config = ScalingConfig()
        config.max_steps = 10
        config.batch_size = 1
        config.max_seq_length = 32
        config.eval_every_n_steps = 2
        config.fault_injection_step = 999  # No fault injection
        config.experiment_name = f"minimal_test_{datetime.now().strftime('%H%M%S')}"
        
        logger.info("Minimal config created")
        
        # Test validator creation
        validator = DebugScalingValidator(config)
        logger.info("Validator created")
        
        # Run single experiment
        result = validator.run_single_test()
        
        if result.get('success', False):
            logger.info("✅ Minimal test PASSED")
            return True
        else:
            logger.error("❌ Minimal test FAILED")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Minimal test crashed: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """Enhanced main function with diagnostics"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Scaling Validation for R-Metric")
    parser.add_argument('--quick', action='store_true', help='Run quick test (50 steps)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--model', type=str, default='gpt2', help='Model to use')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--steps', type=int, default=100, help='Max training steps')
    parser.add_argument('--minimal', action='store_true', help='Run minimal test only')
    parser.add_argument('--diagnostics', action='store_true', help='Run diagnostics only')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔧 DEBUG SCALING VALIDATION EXPERIMENT")
    print("="*80)
    
    # Run diagnostics first
    if args.diagnostics or not args.minimal:
        run_diagnostics()
    
    if args.diagnostics:
        return
    
    # Run minimal test if requested
    if args.minimal:
        success = test_minimal_experiment()
        if success:
            print("\n✅ Minimal test completed successfully!")
            print("Try running the full experiment now.")
        else:
            print("\n❌ Minimal test failed. Check the logs for details.")
        return
    
    # Create configuration
    config = ScalingConfig()
    
    # Apply arguments
    if args.quick:
        config.max_steps = 50
        config.fault_injection_step = 25
        config.eval_every_n_steps = 5
    else:
        config.max_steps = args.steps
        config.fault_injection_step = args.steps // 2
    
    config.use_cpu = args.cpu
    config.model_name = args.model
    config.batch_size = args.batch_size
    
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Steps: {config.max_steps}")
    print(f"Output: {config.output_path}")
    print("="*80)
    
    try:
        # Run validation
        validator = DebugScalingValidator(config)
        
        # Detect GPU
        gpu_info = validator.detect_gpu_configuration()
        print(f"\n🖥️  GPU Configuration: {gpu_info['type']}")
        if gpu_info['count'] > 0:
            for i, (name, mem) in enumerate(zip(gpu_info['names'], gpu_info['memory_gb'])):
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        
        # Run experiments
        print("\n🚀 Starting experiments...")
        results = validator.run_weight_variations()
        
        # Analyze results
        print("\n" + "="*80)
        print("📊 EXPERIMENT ANALYSIS")
        print("="*80)
        
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        print(f"Total experiments: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            lead_times = [r['lead_time_steps'] for r in successful if r['lead_time_steps'] is not None]
            if lead_times:
                print(f"\n✅ Detection Results:")
                print(f"  Mean Lead Time: {np.mean(lead_times):.1f} steps")
                print(f"  Best Lead Time: {max(lead_times)} steps")
                print(f"  Detection Rate: {len(lead_times)}/{len(successful)} experiments")
            else:
                print("\n⚠️  No fault detections (this may be expected)")
        
        if failed:
            print(f"\n❌ Failed Experiments:")
            for i, result in enumerate(failed):
                error_msg = result.get('error', 'Unknown error')
                print(f"  Experiment {result.get('experiment_id', i)}: {error_msg[:100]}...")
        
        print(f"\n📁 Results saved to: {config.output_path}")
        return config.output_path
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        logger.error(traceback.format_exc())
        print(f"\n💥 Fatal error occurred: {e}")
        print("Check the log file for detailed error information.")
        return None


if __name__ == "__main__":
    try:
        # Add memory debugging
        import gc
        gc.set_debug(gc.DEBUG_STATS)
        
        # Run main
        output_path = main()
        
        if output_path:
            print(f"\n🎉 Experiment completed! Check results at: {output_path}")
            sys.exit(0)
        else:
            print("\n💀 Experiment failed completely")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        print("\n⏹️  Experiment interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        logger.error(traceback.format_exc())
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)
