import time
import random
import math
import numpy as np
import json
import os
from collections import deque

# --- Configuration ---
# This dictionary defines a single experimental run. 
# It's configured to run a Llama 3 simulation with a new MoE-specific fault.
CONFIG = {
    'experiment_id': 'run_001_llama3_router_imbalance',
    'model_type': 'Llama-3-8B',  # Modern Architectures: 'Llama-3-8B', 'Mistral-7B', 'GPT-4-MoE'
    'dataset': 'SlimPajama-subset',
    'num_workers': 8,
    'total_training_steps': 5000,
    'eval_every_n_steps': 100,

    # --- Fault Injection ---
    'fault_injection': {
        'type': 'ROUTER_IMBALANCE', # Modern Faults: 'ROUTER_IMBALANCE', 'EXPERT_FAILURE', 'GQA_MISMATCH', 'RMS_NORM_ERROR'
        'trigger_step': 2500,
        'params': {
            'imbalance_factor': 0.8,  # For ROUTER_IMBALANCE: 80% of tokens go to one expert
            'expert_to_kill': 2,       # For EXPERT_FAILURE
        }
    },

    # --- R Metric Weights (from your paper) ---
    'r_metric_weights': {
        'w1_lambda': 0.40,
        'w2_sigma_sq': 0.35,
        'w3_delta_l': 0.25,
    },

    # --- Thresholds for baselines and R metric ---
    'alert_thresholds': {
        'r_metric': 0.65,
        'loss_spike': 3.0,
        'grad_norm': 100.0
    }
}

# --- Core Classes ---

class ReliabilityMonitor:
    """
    Calculates the components of the R metric (λ, σ², ΔL) and the metric itself.
    """
    def __init__(self, config):
        self.config = config
        self.weights = config['r_metric_weights']
        self.hardware_events = deque(maxlen=50)
        self.loss_history = deque(maxlen=100)
        self.lambda_history = deque(maxlen=1000)
        self.sigma_sq_history = deque(maxlen=1000)
        self.delta_l_history = deque(maxlen=1000)

    def _normalize(self, value, history):
        """Applies min-max normalization over a sliding window."""
        if not history or len(history) < 2:
            return 0.0
        min_val, max_val = min(history), max(history)
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def log_hardware_event(self):
        """Simulates logging a hardware event."""
        self.hardware_events.append(time.time())

    def calculate_lambda(self):
        """Calculates λ based on the frequency of recent hardware events."""
        if len(self.hardware_events) < 2:
            return 0.0
        time_span_seconds = self.hardware_events[-1] - self.hardware_events[0]
        if time_span_seconds == 0:
            return 0.0
        failures_per_second = len(self.hardware_events) / time_span_seconds
        lambda_val = failures_per_second * 3600
        self.lambda_history.append(lambda_val)
        return lambda_val

    def calculate_sigma_sq(self, all_worker_grad_norms):
        """Calculates σ² (gradient variance) across all workers."""
        if not all_worker_grad_norms:
            return 0.0
        sigma_sq_val = np.var(all_worker_grad_norms)
        self.sigma_sq_history.append(sigma_sq_val)
        return sigma_sq_val

    def calculate_delta_l(self, current_val_loss):
        """Calculates ΔL (validation loss drift) against a moving average."""
        if np.isnan(current_val_loss):
            return 10.0 # Return a large, stable value for NaN to ensure a high metric score.
        if not self.loss_history:
            self.loss_history.append(current_val_loss)
            return 0.0
        moving_avg = np.mean(self.loss_history)
        delta_l_val = current_val_loss - moving_avg
        self.loss_history.append(current_val_loss)
        self.delta_l_history.append(delta_l_val)
        return delta_l_val

    def calculate_r_metric(self, lambda_val, sigma_sq_val, delta_l_val):
        """Calculates the final R metric using normalized, weighted components."""
        lambda_norm = self._normalize(lambda_val, self.lambda_history)
        sigma_sq_norm = self._normalize(sigma_sq_val, self.sigma_sq_history)
        delta_l_norm = self._normalize(delta_l_val, self.delta_l_history)
        r_metric = (self.weights['w1_lambda'] * lambda_norm +
                    self.weights['w2_sigma_sq'] * sigma_sq_norm +
                    self.weights['w3_delta_l'] * delta_l_norm)
        return {
            'r_metric': r_metric,
            'lambda_norm': lambda_norm,
            'sigma_sq_norm': sigma_sq_norm,
            'delta_l_norm': delta_l_norm
        }

class FaultInjector:
    """Applies different fault types to the training simulation."""
    def __init__(self, config):
        self.config = config['fault_injection']
        self.triggered = False

    def inject(self, step, simulator_state):
        """Checks if a fault should be triggered and applies it."""
        if self.triggered or step != self.config['trigger_step']:
            return simulator_state, None
        self.triggered = True
        fault_type = self.config['type']
        params = self.config['params']
        log_message = f"Injecting fault: {fault_type}"

        # MoE-Specific Faults
        if fault_type == 'ROUTER_IMBALANCE':
            simulator_state['router_imbalance_factor'] = params['imbalance_factor']
            log_message += f" - Imbalance factor set to {params['imbalance_factor']}."
        elif fault_type == 'EXPERT_FAILURE':
            simulator_state['failed_experts'].add(params['expert_to_kill'])
            log_message += f" - Expert {params['expert_to_kill']} has failed."
        # Modern Architecture Faults
        elif fault_type == 'GQA_MISMATCH':
            simulator_state['force_gqa_mismatch'] = True
            log_message += " - Forcing a Grouped-Query Attention mismatch."
        elif fault_type == 'RMS_NORM_ERROR':
            simulator_state['force_rms_norm_error'] = True
            log_message += " - Forcing an RMS Normalization error."
            
        return simulator_state, log_message

class TrainingSimulator:
    """Main class to run a single training simulation experiment."""
    def __init__(self, config):
        self.config = config
        self.monitor = ReliabilityMonitor(config)
        self.injector = FaultInjector(config)
        self.logger = self._setup_logger()
        self.state = {
            'step': 0, 'training_loss': 5.0, 'validation_loss': 5.0,
            'learning_rate': 1e-4, 'active_workers': list(range(config['num_workers'])),
            'router_imbalance_factor': 0.0, 'failed_experts': set(),
            'force_gqa_mismatch': False, 'force_rms_norm_error': False,
            'is_nan': False, 'is_crashed': False
        }
        self.alerts = {}

    def _setup_logger(self):
        """Sets up a structured JSON logger."""
        log_dir = "experiment_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.config['experiment_id']}.jsonl")
        return open(log_path, 'w')

    def _log(self, data):
        """Writes a JSON line to the log file."""
        self.logger.write(json.dumps(data) + '\n')

    def _get_simulated_grad_norms(self):
        """Simulates collecting gradient norms from each worker."""
        if not self.state['active_workers']:
            return []
        base_norm = 10.0 * math.exp(-self.state['step'] / 2000) + 1.0
        norms = [random.uniform(base_norm * 0.9, base_norm * 1.1) for _ in self.state['active_workers']]

        # Simulate effects of modern architecture faults
        if self.state['force_rms_norm_error']:
            norms[random.randint(0, len(norms)-1)] *= 25  # RMS Norm errors can cause sharp spikes
            self.state['force_rms_norm_error'] = False
        if self.state['router_imbalance_factor'] > 0:
            # Imbalance increases variance as some experts get over/under-utilized
            for i in range(len(norms)):
                norms[i] *= (1 + random.uniform(-0.5, 1.5) * self.state['router_imbalance_factor'])
        return norms

    def _update_training_state(self):
        """Simulates a single training step."""
        base_loss_decay = math.exp(-self.state['step'] / 3000)
        self.state['training_loss'] = 3.0 * base_loss_decay + random.uniform(0.1, 0.2)
        if self.state['failed_experts']:
            # Performance degrades if experts fail
            self.state['training_loss'] *= (1 + 0.5 * len(self.state['failed_experts']))
        if self.state['force_gqa_mismatch']:
            # GQA mismatch can lead to representational collapse and higher loss
            self.state['training_loss'] *= 1.8
            self.state['force_gqa_mismatch'] = False

        if self.state['training_loss'] > 50 or np.isnan(self.state['training_loss']):
            self.state['is_nan'] = True
        if not self.state['active_workers']:
            self.state['is_crashed'] = True

    def run(self):
        """Runs the entire training simulation loop."""
        print(f"--- Starting Experiment: {self.config['experiment_id']} ---")
        self._log({'event': 'EXPERIMENT_START', 'config': self.config})

        for step in range(1, self.config['total_training_steps'] + 1):
            self.state['step'] = step
            self.state, fault_log = self.injector.inject(step, self.state)
            if fault_log:
                self._log({'step': step, 'event': 'FAULT_INJECTED', 'details': fault_log})
            self._update_training_state()

            if step % self.config['eval_every_n_steps'] == 0:
                self.state['validation_loss'] = self.state['training_loss'] + random.uniform(0.05, 0.1)
                grad_norms = self._get_simulated_grad_norms()
                lambda_val = self.monitor.calculate_lambda()
                sigma_sq_val = self.monitor.calculate_sigma_sq(grad_norms)
                delta_l_val = self.monitor.calculate_delta_l(self.state['validation_loss'])
                r_metric_data = self.monitor.calculate_r_metric(lambda_val, sigma_sq_val, delta_l_val)
                self._check_alerts(r_metric_data['r_metric'], np.mean(grad_norms) if grad_norms else 0)
                log_data = {'step': step, 'event': 'METRICS', **self.state, **r_metric_data, 'alerts': self.alerts}
                self._log(log_data)
            
            if self.state['is_nan'] or self.state['is_crashed']:
                end_reason = "NaN Loss" if self.state['is_nan'] else "Worker Crash"
                self._log({'step': step, 'event': 'EXPERIMENT_FAILURE', 'reason': end_reason})
                print(f"--- Experiment Failed at step {step}: {end_reason} ---")
                break
        
        if not (self.state['is_nan'] or self.state['is_crashed']):
            self._log({'step': step, 'event': 'EXPERIMENT_SUCCESS'})
            print(f"--- Experiment Completed Successfully ---")
        self.logger.close()

    def _check_alerts(self, r_metric, grad_norm):
        """Checks metric values against thresholds."""
        thresholds = self.config['alert_thresholds']
        if 'r_metric' not in self.alerts and r_metric > thresholds['r_metric']:
            self.alerts['r_metric'] = self.state['step']
        if 'grad_norm' not in self.alerts and grad_norm > thresholds['grad_norm']:
            self.alerts['grad_norm'] = self.state['step']
        loss_hist = list(self.monitor.loss_history)
        if len(loss_hist) > 10:
            mean_loss, std_loss = np.mean(loss_hist), np.std(loss_hist)
            if std_loss > 0 and (self.state['validation_loss'] - mean_loss) / std_loss > thresholds['loss_spike']:
                if 'loss_spike' not in self.alerts:
                    self.alerts['loss_spike'] = self.state['step']

if __name__ == '__main__':
    # Run the default Llama 3 experiment
    simulator = TrainingSimulator(CONFIG)
    simulator.run()

    # Example of another experiment: GPT-4 (MoE) with a different fault
    gpt4_moe_config = CONFIG.copy()
    gpt4_moe_config['experiment_id'] = 'run_002_gpt4_moe_expert_failure'
    gpt4_moe_config['model_type'] = 'GPT-4-MoE'
    gpt4_moe_config['fault_injection'] = {
        'type': 'EXPERT_FAILURE',
        'trigger_step': 3000,
        'params': {'expert_to_kill': 5}
    }
    simulator_2 = TrainingSimulator(gpt4_moe_config)
    simulator_2.run()