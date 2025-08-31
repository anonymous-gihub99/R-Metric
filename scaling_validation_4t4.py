#!/usr/bin/env python3
"""
Production-ready Scaling Validation Script for R-Metric
Author: Lead ML Engineer
Version: 2.0
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scaling_validation.log')
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
    """Configuration for scaling experiments with all required fields"""
    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"  # Start with smaller model for testing
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    
    # Training configuration
    batch_size: int = 1  # Per GPU batch size
    max_steps: int = 100  # Short for testing
    eval_every_n_steps: int = 10
    learning_rate: float = 5e-5
    warmup_steps: int = 10
    max_seq_length: int = 128  # Shorter sequences
    gradient_accumulation_steps: int = 4
    
    # Fault injection
    fault_injection_step: int = 50
    fault_type: str = "LR_SPIKE"  # Simpler fault type
    fault_severity: float = 10.0  # Less severe
    
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
    output_dir: str = "scaling_validation_results"
    experiment_name: str = field(default_factory=lambda: f"scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Device configuration
    device: str = "auto"
    mixed_precision: bool = False  # Disable for stability
    use_cpu: bool = False
    use_ddp: bool = True  # Enable DDP for multi-GPU
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    
    def __post_init__(self):
        """Initialize paths and device"""
        self.output_path = Path(self.output_dir) / self.experiment_name
        if self.local_rank <= 0:  # Only create on main process
            self.output_path.mkdir(parents=True, exist_ok=True)
        
        if self.device == "auto":
            if self.use_cpu:
                self.device = "cpu"
            else:
                if self.local_rank >= 0:
                    self.device = f"cuda:{self.local_rank}"
                else:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with proper serialization"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict


class SimpleReliabilityMonitor:
    """Simplified reliability monitor for testing"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.loss_history = []
        self.gradient_norms = []
        self.r_metric_history = []
        
    def calculate_r_metric(self, loss: float, gradients: Optional[List] = None) -> Dict[str, float]:
        """Calculate simplified R-metric"""
        self.loss_history.append(loss)
        
        # Calculate components
        lambda_val = np.random.uniform(0.05, 0.15)  # Simulated hardware events
        
        # Gradient variance
        if gradients:
            grad_norms = [g.norm().item() for g in gradients if g is not None]
            sigma_sq_val = np.var(grad_norms) if grad_norms else 0.0
            self.gradient_norms.extend(grad_norms)
        else:
            sigma_sq_val = 0.0
        
        # Loss drift
        if len(self.loss_history) > 1:
            delta_l_val = abs(loss - np.mean(self.loss_history[-5:]))
        else:
            delta_l_val = 0.0
        
        # Normalize components (simplified)
        lambda_norm = min(lambda_val * 10, 1.0)
        sigma_sq_norm = min(sigma_sq_val / 100, 1.0) if sigma_sq_val > 0 else 0.0
        delta_l_norm = min(delta_l_val / 2, 1.0) if delta_l_val > 0 else 0.0
        
        # Calculate R-metric
        weights = self.config.r_metric_weights
        r_metric = (
            weights["lambda"] * lambda_norm +
            weights["sigma_sq"] * sigma_sq_norm +
            weights["delta_l"] * delta_l_norm
        )
        
        self.r_metric_history.append(r_metric)
        
        return {
            "r_metric": float(r_metric),
            "lambda": float(lambda_val),
            "lambda_norm": float(lambda_norm),
            "sigma_sq": float(sigma_sq_val),
            "sigma_sq_norm": float(sigma_sq_norm),
            "delta_l": float(delta_l_val),
            "delta_l_norm": float(delta_l_norm)
        }


def setup_distributed():
    """Setup distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = -1
        world_size = 1
        local_rank = -1
    
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


class ScalingValidator:
    """Main validation class with robust error handling"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        
        # Setup distributed training
        if torch.cuda.device_count() > 1 and not self.config.use_cpu:
            rank, world_size, local_rank = setup_distributed()
            self.config.local_rank = local_rank
            self.config.world_size = world_size
            self.rank = rank
            self.is_main_process = (local_rank <= 0)
            
            if local_rank >= 0:
                self.device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
            self.rank = 0
            self.is_main_process = True
            self.config.world_size = 1
        
        self.results = []
        
        if self.is_main_process:
            logger.info(f"Initialized ScalingValidator")
            logger.info(f"Device: {self.device}")
            logger.info(f"World size: {self.config.world_size}")
            logger.info(f"Output path: {config.output_path}")
        
    def detect_gpu_configuration(self) -> Dict[str, Any]:
        """Detect current GPU setup"""
        if not torch.cuda.is_available():
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
            gpu_names.append(torch.cuda.get_device_name(i))
            gpu_memory.append(torch.cuda.get_device_properties(i).total_memory / 1e9)
        
        # Determine configuration type
        if gpu_count == 1:
            if "L4" in gpu_names[0]:
                config_type = "1xL4"
            elif "T4" in gpu_names[0]:
                config_type = "1xT4"
            else:
                config_type = f"1x{gpu_names[0].split()[0]}"
        elif gpu_count == 2:
            config_type = "2xT4" if all("T4" in n for n in gpu_names) else f"{gpu_count}xGPU"
        elif gpu_count == 4:
            config_type = "4xT4" if all("T4" in n for n in gpu_names) else f"{gpu_count}xGPU"
        else:
            config_type = f"{gpu_count}xGPU"
        
        return {
            "type": config_type,
            "count": gpu_count,
            "names": gpu_names,
            "memory_gb": gpu_memory,
            "total_memory_gb": sum(gpu_memory)
        }
    
    def load_model_and_data(self) -> Tuple:
        """Load model and dataset with error handling"""
        try:
            if self.is_main_process:
                logger.info(f"Loading model: {self.config.model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with device_map for multi-GPU memory distribution
            if self.config.world_size > 1 and self.config.use_ddp:
                # For DDP, each process loads the full model on its GPU
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float32,
                    use_cache=False
                )
                model = model.to(self.device)
                
                # Wrap with DDP
                model = DDP(
                    model, 
                    device_ids=[self.config.local_rank] if self.config.local_rank >= 0 else None,
                    output_device=self.config.local_rank if self.config.local_rank >= 0 else None,
                    find_unused_parameters=False
                )
                
                if self.is_main_process:
                    logger.info(f"Model wrapped with DDP across {self.config.world_size} GPUs")
                    # Account for DDP wrapper when counting parameters
                    param_count = sum(p.numel() for p in model.module.parameters())/1e6
                    logger.info(f"Model parameters: {param_count:.1f}M")
            else:
                # Single GPU or CPU
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float32,
                    use_cache=False
                )
                model = model.to(self.device)
                
                if torch.cuda.device_count() > 1 and not self.config.use_ddp:
                    # Use DataParallel as fallback
                    model = nn.DataParallel(model)
                    if self.is_main_process:
                        logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
                
                if self.is_main_process:
                    param_count = sum(p.numel() for p in model.parameters())/1e6
                    logger.info(f"Model loaded successfully. Parameters: {param_count:.1f}M")
            
            model.train()
            
            # Load dataset
            if self.is_main_process:
                logger.info(f"Loading dataset: {self.config.dataset_name}")
            
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split='train'
            )
            
            # Limit dataset size
            dataset = dataset.select(range(min(1000, len(dataset))))
            
            # Tokenize
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_seq_length,
                    return_tensors='pt'
                )
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            tokenized_dataset.set_format(type='torch')
            
            # Create dataloader with DistributedSampler for DDP
            if self.config.world_size > 1 and self.config.use_ddp:
                sampler = DistributedSampler(
                    tokenized_dataset,
                    num_replicas=self.config.world_size,
                    rank=self.rank,
                    shuffle=True
                )
                dataloader = DataLoader(
                    tokenized_dataset,
                    batch_size=self.config.batch_size,
                    sampler=sampler,
                    num_workers=2,
                    pin_memory=True
                )
            else:
                dataloader = DataLoader(
                    tokenized_dataset,
                    batch_size=self.config.batch_size * max(1, torch.cuda.device_count()),
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True if torch.cuda.is_available() else False
                )
            
            if self.is_main_process:
                logger.info(f"Dataset loaded. Size: {len(tokenized_dataset)}")
                effective_batch_size = self.config.batch_size * self.config.world_size
                logger.info(f"Effective batch size: {effective_batch_size}")
            
            return model, tokenizer, dataloader
            
        except Exception as e:
            logger.error(f"Error loading model/data: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_experiment(self, weight_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Run a single experiment with robust error handling"""
        try:
            if weight_config:
                self.config.r_metric_weights = weight_config
                if self.is_main_process:
                    logger.info(f"Testing weights: λ={weight_config['lambda']:.2f}, "
                              f"σ²={weight_config['sigma_sq']:.2f}, "
                              f"ΔL={weight_config['delta_l']:.2f}")
            
            # Load model and data
            model, tokenizer, dataloader = self.load_model_and_data()
            
            # Setup optimizer
            optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps
            )
            
            # Initialize monitoring (only on main process)
            if self.is_main_process:
                monitor = SimpleReliabilityMonitor(self.config)
                metrics_history = []
                first_alert_step = None
                crash_step = None
            
            fault_injected = False
            
            # Training loop
            data_iter = iter(dataloader)
            
            for step in tqdm(range(self.config.max_steps), desc="Training", disable=not self.is_main_process):
                try:
                    # Set epoch for DistributedSampler
                    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
                        dataloader.sampler.set_epoch(step)
                    
                    # Get batch
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
                    
                    # Move to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Inject fault (synchronized across all processes)
                    if step == self.config.fault_injection_step and not fault_injected:
                        if self.is_main_process:
                            logger.info(f"Injecting fault at step {step}")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= self.config.fault_severity
                        fault_injected = True
                    
                    # Forward pass
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids']
                    )
                    loss = outputs.loss
                    
                    # Check for NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        if self.is_main_process:
                            logger.warning(f"NaN/Inf loss at step {step}")
                            crash_step = step
                        break
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient accumulation
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        # Get gradients for monitoring (only on main process)
                        if self.is_main_process:
                            if isinstance(model, DDP):
                                gradients = [p.grad.clone() for p in model.module.parameters() if p.grad is not None]
                            else:
                                gradients = [p.grad.clone() for p in model.parameters() if p.grad is not None]
                        else:
                            gradients = None
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Optimizer step
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    else:
                        gradients = None
                    
                    # Monitoring every N steps (only on main process)
                    if self.is_main_process and step % self.config.eval_every_n_steps == 0:
                        # Calculate R-metric
                        r_metric_results = monitor.calculate_r_metric(
                            loss.item() * self.config.gradient_accumulation_steps,
                            gradients
                        )
                        
                        # Check for alert
                        if r_metric_results['r_metric'] > self.config.r_metric_alert_threshold:
                            if first_alert_step is None:
                                first_alert_step = step
                                logger.info(f"R-Metric alert at step {step}: {r_metric_results['r_metric']:.3f}")
                        
                        # Store metrics
                        metrics_history.append({
                            'step': step,
                            'loss': float(loss.item() * self.config.gradient_accumulation_steps),
                            **r_metric_results
                        })
                        
                        # Log progress
                        if step % 50 == 0:
                            logger.info(f"Step {step}: Loss={metrics_history[-1]['loss']:.4f}, "
                                      f"R-Metric={metrics_history[-1]['r_metric']:.3f}")
                    
                except Exception as e:
                    if self.is_main_process:
                        logger.error(f"Error at step {step}: {e}")
                        crash_step = step
                    break
            
            # Calculate results (only on main process)
            if self.is_main_process:
                lead_time = None
                if first_alert_step is not None and fault_injected:
                    lead_time = self.config.fault_injection_step - first_alert_step
                
                # Get GPU info
                gpu_config = self.detect_gpu_configuration()
                
                result = {
                    'gpu_config': gpu_config['type'],
                    'gpu_count': gpu_config['count'],
                    'weight_lambda': float(self.config.r_metric_weights['lambda']),
                    'weight_sigma_sq': float(self.config.r_metric_weights['sigma_sq']),
                    'weight_delta_l': float(self.config.r_metric_weights['delta_l']),
                    'first_alert_step': first_alert_step,
                    'fault_injection_step': self.config.fault_injection_step,
                    'crash_step': crash_step,
                    'lead_time_steps': lead_time,
                    'metrics_history': metrics_history,
                    'timestamp': datetime.now().isoformat(),
                    'world_size': self.config.world_size,
                    'distributed': self.config.use_ddp and self.config.world_size > 1
                }
            else:
                result = {}
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
            # Synchronize before returning
            if dist.is_initialized():
                dist.barrier()
            
            return result
            
        except Exception as e:
            if self.is_main_process:
                logger.error(f"Experiment failed: {e}")
                logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_weight_variations(self) -> List[Dict]:
        """Test multiple weight configurations"""
        weight_configs = [
            {"lambda": 0.10, "sigma_sq": 0.45, "delta_l": 0.70},  # Default
            {"lambda": 0.15, "sigma_sq": 0.55, "delta_l": 0.70},  # Variant 1
            {"lambda": 0.10, "sigma_sq": 0.52, "delta_l": 0.70},  # Variant 2
        ]
        
        all_results = []
        
        for i, weights in enumerate(weight_configs):
            if self.is_main_process:
                logger.info(f"\nRunning experiment {i+1}/{len(weight_configs)}")
            
            result = self.run_experiment(weights)
            
            if self.is_main_process and result:
                result['experiment_id'] = i
                all_results.append(result)
                # Save intermediate results
                self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: List[Dict]):
        """Save results with proper JSON serialization (only on main process)"""
        if not self.is_main_process:
            return
        
        try:
            # Save JSON
            json_path = self.config.output_path / "scaling_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, cls=JSONEncoder)
            logger.info(f"Results saved to {json_path}")
            
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
                        'lead_time_steps': r['lead_time_steps'],
                        'world_size': r.get('world_size', 1),
                        'distributed': r.get('distributed', False)
                    })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                csv_path = self.config.output_path / "scaling_summary.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Summary saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main execution with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scaling Validation for R-Metric")
    parser.add_argument('--quick', action='store_true', help='Run quick test (100 steps)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-8B', help='Model to use')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--steps', type=int, default=200, help='Max training steps')
    parser.add_argument('--no-ddp', action='store_true', help='Disable DDP, use DataParallel instead')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ScalingConfig()
    
    # Apply arguments
    if args.quick:
        config.max_steps = 100
        config.fault_injection_step = 50
        config.eval_every_n_steps = 10
    else:
        config.max_steps = args.steps
        config.fault_injection_step = args.steps // 2
    
    config.use_cpu = args.cpu
    config.model_name = args.model
    config.batch_size = args.batch_size
    config.use_ddp = not args.no_ddp
    config.local_rank = args.local_rank
    
    # Setup for torchrun/DDP
    if 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])
    
    # Only print on main process
    is_main = (config.local_rank <= 0)
    
    if is_main:
        print("\n" + "="*60)
        print("SCALING VALIDATION EXPERIMENT")
        print("="*60)
        print(f"Model: {config.model_name}")
        print(f"Device: {config.device}")
        print(f"Steps: {config.max_steps}")
        print(f"Batch size per GPU: {config.batch_size}")
        print(f"DDP Enabled: {config.use_ddp}")
        print(f"Output: {config.output_path}")
        print("="*60)
    
    # Run validation
    validator = ScalingValidator(config)
    
    # Detect GPU (only on main process)
    if is_main:
        gpu_info = validator.detect_gpu_configuration()
        print(f"\nGPU Configuration: {gpu_info['type']}")
        if gpu_info['count'] > 0:
            for i, (name, mem) in enumerate(zip(gpu_info['names'], gpu_info['memory_gb'])):
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
            print(f"Total GPU Memory: {gpu_info['total_memory_gb']:.1f} GB")
            effective_batch = config.batch_size * config.world_size
            print(f"Effective batch size: {effective_batch}")
    
    # Run experiments
    if is_main:
        print("\nStarting experiments...")
    
    results = validator.run_weight_variations()
    
    # Analyze results (only on main process)
    if is_main:
        successful = [r for r in results if 'error' not in r]
        if successful:
            lead_times = [r['lead_time_steps'] for r in successful if r['lead_time_steps'] is not None]
            if lead_times:
                print(f"\n✅ Experiments Complete!")
                print(f"Mean Lead Time: {np.mean(lead_times):.1f} steps")
                print(f"Best Lead Time: {max(lead_times):.1f} steps")
            else:
                print("\n⚠️ No successful detections")
        else:
            print("\n❌ All experiments failed")
        
        print(f"\nResults saved to: {config.output_path}")
    
    # Clean up distributed
    cleanup_distributed()
    
    return config.output_path


if __name__ == "__main__":
    try:
        output_path = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger