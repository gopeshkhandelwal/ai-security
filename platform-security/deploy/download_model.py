#!/usr/bin/env python3
"""
Pathfinder Model Downloader
Downloads models to quarantine directory for security scanning.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

def download_model(model_id: str, output_dir: str, token: str = None) -> dict:
    """Download model from Hugging Face to specified directory."""
    from huggingface_hub import snapshot_download
    
    model_name = model_id.split("/")[-1]
    local_dir = Path(output_dir) / model_name
    
    print(f"[INFO] Downloading {model_id}")
    print(f"[INFO] Destination: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            token=token if token else None,
            local_dir_use_symlinks=False
        )
        
        result = {
            "success": True,
            "model_id": model_id,
            "model_name": model_name,
            "local_path": str(local_dir),
            "timestamp": datetime.utcnow().isoformat()
        }
        print(f"[SUCCESS] Download complete: {local_dir}")
        
    except Exception as e:
        result = {
            "success": False,
            "model_id": model_id,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Download model to quarantine")
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., meta-llama/Llama-3.1-8B)")
    parser.add_argument("--output-dir", "-o", default="/llm/models/quarantine", 
                        help="Output directory")
    parser.add_argument("--token", "-t", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    result = download_model(args.model_id, args.output_dir, args.token)
    
    if args.json:
        print(json.dumps(result, indent=2))
    
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
