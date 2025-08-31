#!/usr/bin/env python3
"""
Quick diagnostic script to identify the root cause of scaling validation failures
"""
import torch
import sys
import traceback
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        from tqdm import tqdm
        print("✅ tqdm")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_torch_gpu():
    """Test PyTorch GPU functionality"""
    print("\n🔍 Testing PyTorch GPU...")
    
    try:
        import torch
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)")
            
            # Test tensor operations
            x = torch.randn(2, 2).cuda()
            y = x * 2
            print(f"✅ GPU tensor test passed: {y.sum().item():.2f}")
            
            del x, y
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test basic model loading"""
    print("\n🔍 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "gpt2"
        print(f"Loading {model_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
        print("✅ Model loaded")
        
        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Test tokenization
        text = "Hello world"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Test inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Model inference test passed")
        print(f"   Output shape: {outputs.logits.shape}")
        
        # Cleanup
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test dataset loading"""
    print("\n🔍 Testing dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Try to load wikitext
        print("Loading wikitext dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Test a few samples
        sample = dataset[0]
        print(f"✅ Sample access: {len(sample['text'])} chars")
        
        # Test subset
        subset = dataset.select(range(10))
        print(f"✅ Subset creation: {len(subset)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        traceback.print_exc()
        return False

def test_training_step():
    """Test a single training step"""
    print("\n🔍 Testing training step...")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from datasets import load_dataset
        from torch.utils.data import DataLoader
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load minimal components
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32
        ).to(device)
        model.train()
        
        # Create simple dataset
        texts = ["Hello world", "This is a test", "PyTorch training"]
        
        # Tokenize
        inputs = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=32,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Single forward pass
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'], 
            labels=inputs['input_ids']
        )
        
        loss = outputs.loss
        print(f"✅ Forward pass: loss = {loss.item():.4f}")
        
        # Single backward pass
        loss.backward()
        print("✅ Backward pass completed")
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"✅ Gradient norm: {grad_norm:.4f}")
        
        # Cleanup
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("✅ Single training step test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Training step test failed: {e}")
        traceback.print_exc()
        return False

def analyze_original_error():
    """Analyze the original error from the output"""
    print("\n🔍 Analyzing original error...")
    
    # Look for common failure patterns
    print("Common failure causes:")
    print("1. ❌ All experiments failed - suggests early failure in setup")
    print("2. Model loading issues (memory, CUDA, compatibility)")
    print("3. Dataset loading/tokenization issues")
    print("4. Training loop exceptions")
    print("5. R-metric calculation errors")
    
    print("\nRecommended debugging steps:")
    print("1. Run: python debug_script.py --minimal")
    print("2. Run: python debug_script.py --diagnostics")
    print("3. Check log files for detailed errors")
    print("4. Try CPU-only: python debug_script.py --cpu")

def main():
    """Run all diagnostic tests"""
    print("🚀 Starting comprehensive diagnostics...\n")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    # Test 2: GPU
    results['gpu'] = test_torch_gpu()
    
    # Test 3: Model loading
    results['model_loading'] = test_model_loading()
    
    # Test 4: Dataset loading
    results['dataset_loading'] = test_dataset_loading()
    
    # Test 5: Training step
    results['training_step'] = test_training_step()
    
    # Summary
    print("\n" + "="*60)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The original script should work. Try running it again.")
        print("If it still fails, run with --minimal flag first.")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("Fix the failing components before running the main script.")
        
    print("\nNext steps:")
    print("1. If diagnostics pass: run the debug version with --minimal")
    print("2. If minimal test passes: run with --quick")
    print("3. If quick test passes: run full experiments")
    
    analyze_original_error()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 Diagnostic script crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
