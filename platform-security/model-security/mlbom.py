#!/usr/bin/env python3
"""
MLBOM Generator

Generates Machine Learning Bill of Materials for AI models.
Provides integrity tracking, provenance information, and security metadata.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger("security.mlbom")


@dataclass
class FileEntry:
    """File entry in MLBOM"""
    path: str
    size: int
    sha256: str


@dataclass
class MLBOM:
    """Machine Learning Bill of Materials"""
    model_id: str
    model_path: str
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    version: str = "1.0"
    files: List[FileEntry] = field(default_factory=list)
    total_size: int = 0
    scan_passed: bool = True
    trust_remote_code: bool = False
    primary_format: str = "unknown"
    
    def to_dict(self) -> dict:
        return {
            "mlbom_version": self.version,
            "generated_at": self.generated_at,
            "model": {
                "id": self.model_id,
                "name": self.model_id.split("/")[-1],
                "local_path": self.model_path,
                "total_size_bytes": self.total_size,
                "file_count": len(self.files)
            },
            "security": {
                "scan_passed": self.scan_passed,
                "scan_timestamp": self.generated_at,
                "trust_remote_code": self.trust_remote_code,
                "primary_format": self.primary_format
            },
            "provenance": {
                "source": "huggingface.co",
                "downloaded_by": os.environ.get("USER", "unknown"),
                "host": os.uname().nodename
            },
            "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} 
                      for f in self.files[:50]]
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, output_path: Optional[str] = None) -> str:
        """Save MLBOM to file"""
        if output_path is None:
            output_path = str(Path(self.model_path) / "mlbom.json")
        Path(output_path).write_text(self.to_json())
        return output_path


class MLBOMGenerator:
    """
    Generator for Machine Learning Bill of Materials.
    
    Creates MLBOM documents that track model artifacts, integrity hashes,
    and security scan results.
    """
    
    # Maximum file size for hashing (100MB)
    MAX_HASH_SIZE = 100_000_000
    
    def __init__(self, include_large_files: bool = False):
        """
        Initialize generator.
        
        Args:
            include_large_files: Hash files larger than 100MB (slow)
        """
        self.include_large_files = include_large_files
    
    def generate(
        self,
        model_path: str,
        model_id: str,
        scan_passed: bool = True,
        trust_remote_code: bool = False
    ) -> MLBOM:
        """
        Generate MLBOM for a model.
        
        Args:
            model_path: Path to model directory
            model_id: Model identifier (e.g., meta-llama/Llama-3.2-1B)
            scan_passed: Whether security scan passed
            trust_remote_code: Whether model requires trust_remote_code
            
        Returns:
            MLBOM object
        """
        model_path = Path(model_path)
        logger.info(f"Generating MLBOM for {model_id}")
        
        files = []
        total_size = 0
        formats = set()
        
        for f in sorted(model_path.rglob("*")):
            if f.is_file():
                size = f.stat().st_size
                total_size += size
                
                # Hash file (skip large files unless configured)
                if size < self.MAX_HASH_SIZE or self.include_large_files:
                    file_hash = self._sha256_file(f)
                else:
                    file_hash = "skipped_large_file"
                
                files.append(FileEntry(
                    path=str(f.relative_to(model_path)),
                    size=size,
                    sha256=file_hash
                ))
                
                # Track format
                if "." in f.name:
                    formats.add(f.suffix.lstrip("."))
        
        # Determine primary format
        primary_format = "safetensors" if "safetensors" in formats else "unknown"
        
        mlbom = MLBOM(
            model_id=model_id,
            model_path=str(model_path),
            files=files,
            total_size=total_size,
            scan_passed=scan_passed,
            trust_remote_code=trust_remote_code,
            primary_format=primary_format
        )
        
        logger.info(f"MLBOM generated: {len(files)} files, {total_size} bytes")
        return mlbom
    
    def _sha256_file(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate MLBOM for AI model"
    )
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--model-id", required=True, 
                        help="Model ID (e.g., meta-llama/Llama-3.1-8B)")
    parser.add_argument("--scan-passed", action="store_true", default=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--output", "-o", help="Output file path")
    
    args = parser.parse_args()
    
    generator = MLBOMGenerator()
    mlbom = generator.generate(
        args.model_path,
        args.model_id,
        args.scan_passed,
        args.trust_remote_code
    )
    
    output_path = mlbom.save(args.output)
    print(f"[SUCCESS] MLBOM generated: {output_path}")
    print(mlbom.to_json())


if __name__ == "__main__":
    main()
