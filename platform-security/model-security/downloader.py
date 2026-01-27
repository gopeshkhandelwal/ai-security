#!/usr/bin/env python3
"""
Model Downloader

Downloads AI models from Hugging Face to a quarantine directory
for security scanning before promotion to verified storage.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("security.downloader")


@dataclass
class DownloadResult:
    """Result of model download"""
    success: bool
    model_id: str
    model_name: str
    local_path: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "local_path": self.local_path,
            "error": self.error,
            "timestamp": self.timestamp
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ModelDownloader:
    """
    Downloads models from Hugging Face to quarantine.
    
    Models are downloaded to a quarantine directory where they can be
    scanned before being promoted to verified storage.
    """
    
    DEFAULT_QUARANTINE = "/llm/models/quarantine"
    
    def __init__(
        self,
        quarantine_dir: str = None,
        token: str = None
    ):
        """
        Initialize downloader.
        
        Args:
            quarantine_dir: Directory for downloaded models
            token: Hugging Face authentication token
        """
        self.quarantine_dir = Path(quarantine_dir or self.DEFAULT_QUARANTINE)
        # Only use token if it's non-empty
        env_token = os.environ.get("HF_TOKEN", "")
        self.token = token if token else (env_token if env_token else None)
    
    def download(
        self,
        model_id: str,
        revision: str = None,
        allow_patterns: list = None,
        ignore_patterns: list = None
    ) -> DownloadResult:
        """
        Download a model to quarantine.
        
        Args:
            model_id: Hugging Face model ID (e.g., meta-llama/Llama-3.2-1B)
            revision: Specific revision/branch to download
            allow_patterns: Only download files matching patterns
            ignore_patterns: Skip files matching patterns
            
        Returns:
            DownloadResult with download status
        """
        from huggingface_hub import snapshot_download
        
        model_name = model_id.split("/")[-1]
        local_dir = self.quarantine_dir / model_name
        
        logger.info(f"Downloading {model_id} to quarantine")
        logger.info(f"Destination: {local_dir}")
        
        try:
            # Ensure quarantine directory exists
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                revision=revision,
                token=self.token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_dir_use_symlinks=False
            )
            
            result = DownloadResult(
                success=True,
                model_id=model_id,
                model_name=model_name,
                local_path=str(local_dir)
            )
            logger.info(f"Download complete: {local_dir}")
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            result = DownloadResult(
                success=False,
                model_id=model_id,
                model_name=model_name,
                error=str(e)
            )
        
        return result
    
    def download_safetensors_only(self, model_id: str) -> DownloadResult:
        """
        Download only safetensors files (skip pickle-based formats).
        
        Args:
            model_id: Hugging Face model ID
            
        Returns:
            DownloadResult
        """
        return self.download(
            model_id,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.md"],
            ignore_patterns=["*.pt", "*.pth", "*.bin", "*.pkl", "*.pickle"]
        )


def main():
    """CLI entry point"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download model to quarantine for security scanning"
    )
    parser.add_argument("model_id", 
                        help="Hugging Face model ID (e.g., meta-llama/Llama-3.1-8B)")
    parser.add_argument("--output-dir", "-o", 
                        default=ModelDownloader.DEFAULT_QUARANTINE,
                        help="Quarantine directory")
    parser.add_argument("--token", "-t", 
                        default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token")
    parser.add_argument("--safetensors-only", action="store_true",
                        help="Download only safetensors files")
    parser.add_argument("--json", action="store_true", 
                        help="Output as JSON")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(
        quarantine_dir=args.output_dir,
        token=args.token
    )
    
    if args.safetensors_only:
        result = downloader.download_safetensors_only(args.model_id)
    else:
        result = downloader.download(args.model_id)
    
    if args.json:
        print(result.to_json())
    else:
        if result.success:
            print(f"[SUCCESS] Downloaded to: {result.local_path}")
        else:
            print(f"[ERROR] {result.error}", file=sys.stderr)
    
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
