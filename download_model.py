#!/usr/bin/env python3
"""
Script untuk mendownload model Llama 3
Alternative untuk yang tidak menggunakan Jupyter Notebook

Usage:
    python download_model.py --model llama3-8b-instruct
    python download_model.py --model llama3-8b --method ollama
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_directories():
    """Setup directories untuk model"""
    model_dir = "./models"
    llama3_dir = os.path.join(model_dir, "llama3")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(llama3_dir, exist_ok=True)
    
    return model_dir, llama3_dir

def download_with_transformers(model_name, cache_dir):
    """Download model menggunakan transformers"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print(f"Downloading {model_name} with transformers...")
        
        # Download tokenizer
        print("1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded")
        
        # Download model
        print("2. Downloading model (this may take a long time)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✓ Model downloaded")
        
        # Quick test
        print("3. Testing model...")
        test_input = tokenizer("Hello", return_tensors="pt")
        with torch.no_grad():
            output = model.generate(test_input.input_ids, max_length=20)
        print("✓ Model test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def download_with_ollama(model_name):
    """Download model menggunakan Ollama"""
    try:
        print("Installing Ollama...")
        
        # Install Ollama
        result = os.system("curl -fsSL https://ollama.ai/install.sh | sh")
        if result != 0:
            print("❌ Failed to install Ollama")
            return False
        
        print("✓ Ollama installed")
        
        # Pull model
        print("Pulling Llama 3 model...")
        result = os.system("ollama pull llama3")
        
        if result == 0:
            print("✓ Model downloaded with Ollama")
            return True
        else:
            print("❌ Failed to download model")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Llama 3 model")
    parser.add_argument(
        "--model", 
        default="llama3-8b-instruct",
        choices=["llama3-8b", "llama3-8b-instruct", "llama3-70b", "llama3-70b-instruct"],
        help="Model to download"
    )
    parser.add_argument(
        "--method",
        default="transformers",
        choices=["transformers", "ollama"],
        help="Download method"
    )
    
    args = parser.parse_args()
    
    # Model mapping
    models = {
        "llama3-8b": "meta-llama/Meta-Llama-3-8B",
        "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3-70b": "meta-llama/Meta-Llama-3-70B",
        "llama3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct"
    }
    
    print(f"=== Llama 3 Model Downloader ===")
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Full model name: {models.get(args.model, 'Unknown')}")
    print("=" * 40)
    
    # Setup directories
    model_dir, llama3_dir = setup_directories()
    print(f"Model directory: {model_dir}")
    
    # Check requirements
    try:
        if args.method == "transformers":
            import torch
            import transformers
            print(f"PyTorch version: {torch.__version__}")
            print(f"Transformers version: {transformers.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
        
        print("\n" + "=" * 40)
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {str(e)}")
        print("Install with: pip install -r requirements.txt")
        return 1
    
    # Download model
    success = False
    if args.method == "transformers":
        success = download_with_transformers(models[args.model], llama3_dir)
    elif args.method == "ollama":
        success = download_with_ollama(args.model)
    
    if success:
        # Save config
        config = {
            "model": args.model,
            "method": args.method,
            "model_name": models.get(args.model, ""),
            "download_dir": model_dir,
            "success": True
        }
        
        config_file = os.path.join(model_dir, "download_config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Model downloaded successfully!")
        print(f"Configuration saved to: {config_file}")
        return 0
    else:
        print(f"\n❌ Failed to download model")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection")
        print("2. Check if you have enough disk space (15GB+)")
        print("3. For Llama models, you might need Hugging Face access")
        print("4. Try using --method ollama as alternative")
        return 1

if __name__ == "__main__":
    sys.exit(main())