#!/usr/bin/env python3
"""
Test script to verify scaling validation setup
Run this FIRST to ensure everything works
"""

import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test all required imports"""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    required_modules = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'tqdm': 'TQDM'
    }
    
    all_good = True
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"✅ {name:<20} - OK")
        except ImportError as e:
            print(f"❌ {name:<20} - MISSING ({e})")
            all_good = False
    
    return all_good


def test_gpu():
    """Test GPU availability"""
    print("\n" + "="*60)
    print("TESTING GPU")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA Available: {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {name} ({memory:.1f} GB)")
        
        # Test memory allocation
        try:
            device = torch.device("cuda:0")
            test_tensor = torch.randn(1000, 1000).to(device)
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ GPU memory allocation: OK")
            return True
        except Exception as e:
            print(f"⚠️ GPU memory allocation failed: {e}")
            return False
    else:
        print("⚠️ No GPU available - will use CPU (slower)")
        return False


def test_model_loading():
    """Test model loading"""
    print("\n" + "="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "gpt2"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Test tokenization
        text = "This is a test."
        tokens = tokenizer(text, return_tensors='pt')
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(**tokens, labels=tokens['input_ids'])
            loss = outputs.loss
        
        print(f"✅ Model loading: OK")
        print(f"   Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        print(f"   Test loss: {loss.item():.4f}")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_dataset():
    """Test dataset loading"""
    print("\n" + "="*60)
    print("TESTING DATASET")
    print("="*60)
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        print(f"✅ Dataset loading: OK")
        print(f"   Dataset size: {len(dataset)} samples")
        print(f"   Sample text: {dataset[0]['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False


def test_local_imports():
    """Test local module imports"""
    print("\n" + "="*60)
    print("TESTING LOCAL MODULES")
    print("="*60)
    
    modules_to_test = [
        ('case_study', 'ReliabilityMonitor'),
        ('enhanced_baseline_comparison', 'FixedIsolationForestBaseline'),
    ]
    
    all_good = True
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                print(f"✅ {module_name}.{class_name} - OK")
            else:
                print(f"⚠️ {module_name} loaded but {class_name} not found")
                all_good = False
        except ImportError as e:
            print(f"❌ {module_name} - FAILED ({e})")
            all_good = False
    
    return all_good


def run_minimal_test():
    """Run a minimal training test"""
    print("\n" + "="*60)
    print("RUNNING MINIMAL TRAINING TEST")
    print("="*60)
    
    try:
        from fixed_scaling_validation import ScalingConfig, ScalingValidator
        
        # Create minimal config
        config = ScalingConfig()
        config.max_steps = 20
        config.eval_every_n_steps = 5
        config.fault_injection_step = 10
        config.batch_size = 1
        config.max_seq_length = 64
        
        print(f"Configuration:")
        print(f"  Model: {config.model_name}")
        print(f"  Device: {config.device}")
        print(f"  Steps: {config.max_steps}")
        
        # Run minimal experiment
        validator = ScalingValidator(config)
        result = validator.run_experiment()
        
        if 'error' not in result:
            print("✅ Minimal training test: PASSED")
            print(f"   First alert: Step {result.get('first_alert_step', 'None')}")
            print(f"   Lead time: {result.get('lead_time_steps', 'N/A')} steps")
            return True
        else:
            print(f"❌ Minimal training test: FAILED")
            print(f"   Error: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Minimal training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SCALING VALIDATION SETUP TEST")
    print("="*60)
    
    results = {
        "Imports": test_imports(),
        "GPU": test_gpu(),
        "Model Loading": test_model_loading(),
        "Dataset": test_dataset(),
        "Local Modules": test_local_imports(),
        "Training": run_minimal_test()
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<20} : {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Ready to run scaling validation.")
        print("\nNext steps:")
        print("1. Quick test:  python fixed_scaling_validation.py --quick")
        print("2. Full run:    python fixed_scaling_validation.py --steps 500")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before running experiments.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install transformers datasets torch")
        print("- Check GPU drivers: nvidia-smi")
        print("- Ensure case_study.py is in the same directory")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())