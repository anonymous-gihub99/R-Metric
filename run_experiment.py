import itertools
import copy
import os
from real_simulator import TrainingSimulator, CONFIG as BASE_CONFIG

# --- Step 1: Define the Experiment Matrix ---

MODELS = [
    'LLaMA-2-70B',
    'GPT-4',
    'MoE-Expert-64',
    'T5-XXL',          # still useful baseline for seq2seq
    'BERT-Large'       # legacy baseline for comparison
]

REPETITIONS = [1, 2, 3, 4, 5, 6, 7, 8]

SEVERITY_FAULTS = {
    'LR_SPIKE': {
        'low': {'lr_multiplier': 5.0},
        'medium': {'lr_multiplier': 15.0},
        'high': {'lr_multiplier': 25.0}
    },
    'NETWORK_DEGRADATION': {
        'low': {'delay_ms': 50},
        'medium': {'delay_ms': 150},
        'high': {'delay_ms': 250}
    },
    'DATA_CORRUPTION': {
        'low': {'corruption_rate': 0.02},
        'medium': {'corruption_rate': 0.05},
        'high': {'corruption_rate': 0.10}
    },
    # Modern training pain points
    'EXPERT_ROUTING_FAILURE': {   # Specific to MoE
        'low': {'drop_rate': 0.01},
        'medium': {'drop_rate': 0.05},
        'high': {'drop_rate': 0.15}
    },
    'KV_CACHE_STALL': {          # Transformer-specific runtime issue
        'low': {'stall_ms': 20},
        'medium': {'stall_ms': 75},
        'high': {'stall_ms': 200}
    }
}

SIMPLE_FAULTS = [
    'NODE_FAILURE',
    'GRADIENT_EXPLOSION',
    'OOM_ERROR',             # Out of Memory (very common in large models)
    'DEADLOCK'               # Distributed training hang
]

NO_FAULT = 'NONE'

# *** KEY CHANGE: Lower the alert threshold to improve sensitivity (recall) ***
BASE_CONFIG['alert_thresholds']['r_metric'] = 0.56

def main():
    """
    Main function to orchestrate and run all experiments.
    """
    print("--- Starting Final Tuned Experimental Suite (Modern Architectures) ---")
    experiment_counter = 0
    all_configs = []

    # Parametric severity-based faults
    for model, (fault_type, severities), rep in itertools.product(
        MODELS, SEVERITY_FAULTS.items(), REPETITIONS
    ):
        for severity, params in severities.items():
            experiment_id = f"run_{experiment_counter:03d}_{model.lower().replace(' ', '_')}_{fault_type.lower()}_{severity}_rep{rep}"
            config = copy.deepcopy(BASE_CONFIG)
            config['experiment_id'] = experiment_id
            config['model_type'] = model
            config['fault_injection']['type'] = fault_type
            config['fault_injection']['params'].update(params)
            all_configs.append(config)
            experiment_counter += 1

    # Simple single faults
    for model, fault_type, rep in itertools.product(MODELS, SIMPLE_FAULTS, REPETITIONS):
        experiment_id = f"run_{experiment_counter:03d}_{model.lower().replace(' ', '_')}_{fault_type.lower()}_rep{rep}"
        config = copy.deepcopy(BASE_CONFIG)
        config['experiment_id'] = experiment_id
        config['model_type'] = model
        config['fault_injection']['type'] = fault_type
        all_configs.append(config)
        experiment_counter += 1

    # Control (no faults)
    for model, rep in itertools.product(MODELS, REPETITIONS):
        experiment_id = f"run_{experiment_counter:03d}_{model.lower().replace(' ', '_')}_control_rep{rep}"
        config = copy.deepcopy(BASE_CONFIG)
        config['experiment_id'] = experiment_id
        config['model_type'] = model
        config['fault_injection']['type'] = NO_FAULT
        all_configs.append(config)
        experiment_counter += 1

    print(f"Generated a total of {len(all_configs)} experiment configurations.")
    for i, config in enumerate(all_configs):
        print(f"\n--- Running Experiment {i+1}/{len(all_configs)}: {config['experiment_id']} ---")
        try:
            simulator = TrainingSimulator(config)
            simulator.run()
        except Exception as e:
            print(f"!!! Experiment {config['experiment_id']} failed with an error: {e} !!!")

    print("\n--- Full Experimental Suite Completed ---")

if __name__ == '__main__':
    main()
