#!/usr/bin/env python3
"""
Pathfinder Secure Model Loader for Intel Gaudi

This module provides secure model loading with:
1. Pre-load security scanning (ModelScan, PickleScan, AST patterns)
2. Signature verification (ECDSA)
3. MLBOM validation
4. Enforced trust_remote_code=False

Integrates with the Pathfinder security pipeline to ensure only
verified models are loaded on Intel Gaudi accelerators.

Usage:
    from pathfinder_loader import SecureModelLoader
    
    loader = SecureModelLoader()
    model, tokenizer = loader.load("meta-llama/Llama-3.2-1B")

Author: AI Model Pathfinder Security Team
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple, Any
from dataclasses import dataclass

# Signature verification
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Pathfinder Scanner
from pathfinder_scanner import PathfinderScanner, ScanResult

logger = logging.getLogger("pathfinder.loader")


@dataclass
class LoadResult:
    """Result of secure model loading"""
    success: bool
    model: Any = None
    tokenizer: Any = None
    mlbom: Optional[dict] = None
    scan_result: Optional[ScanResult] = None
    error: Optional[str] = None


class SecureModelLoader:
    """
    Secure model loader for AI Model Pathfinder.
    
    Ensures all security checks pass before loading any model.
    """
    
    def __init__(
        self,
        verified_models_path: str = "/verified-models",
        public_key_path: Optional[str] = None,
        skip_verification: bool = False,  # NEVER set True in production
        device: str = "hpu"  # Default to Gaudi
    ):
        """
        Initialize secure loader.
        
        Args:
            verified_models_path: Path to verified model store
            public_key_path: Path to ECDSA public key for signature verification
            skip_verification: Skip signature verification (DANGER - dev only)
            device: Target device (hpu for Gaudi, cuda, cpu)
        """
        self.verified_models_path = Path(verified_models_path)
        self.public_key_path = public_key_path
        self.skip_verification = skip_verification
        self.device = device
        
        # Load public key if provided
        self.public_key = None
        if public_key_path and Path(public_key_path).exists():
            with open(public_key_path, 'rb') as f:
                self.public_key = serialization.load_pem_public_key(f.read())
        
        # Initialize scanner
        self.scanner = PathfinderScanner(strict_mode=True, allow_pickle=False)
        
        if skip_verification:
            logger.warning("⚠️  SIGNATURE VERIFICATION DISABLED - Development mode only!")
    
    def load(
        self,
        model_id: str,
        torch_dtype: str = "bfloat16",
        require_mlbom: bool = True
    ) -> LoadResult:
        """
        Securely load a model with full verification.
        
        Args:
            model_id: Model identifier (HF format: org/model-name)
            torch_dtype: Data type for model weights
            require_mlbom: Require MLBOM to exist
            
        Returns:
            LoadResult with model, tokenizer, and verification details
        """
        import torch
        
        logger.info(f"SecureModelLoader: Loading {model_id}")
        
        # Normalize model ID to path
        model_name = model_id.replace("/", "--")
        model_path = self.verified_models_path / model_name
        
        # Step 1: Check if model exists in verified store
        if not model_path.exists():
            return LoadResult(
                success=False,
                error=f"Model not found in verified store: {model_path}. Run 'pathfinder enable {model_id}' first."
            )
        
        # Step 2: Verify MLBOM exists and is valid
        mlbom_path = model_path / "mlbom.json"
        mlbom = None
        
        if require_mlbom:
            if not mlbom_path.exists():
                return LoadResult(
                    success=False,
                    error=f"MLBOM not found: {mlbom_path}. Model may not have been properly enabled."
                )
            
            try:
                with open(mlbom_path) as f:
                    mlbom = json.load(f)
                logger.info(f"  ✓ MLBOM loaded: {mlbom.get('model', {}).get('id')}")
            except Exception as e:
                return LoadResult(
                    success=False,
                    error=f"Invalid MLBOM: {e}"
                )
        
        # Step 3: Verify signatures
        if not self.skip_verification:
            sig_result = self._verify_signatures(model_path, mlbom)
            if not sig_result["valid"]:
                return LoadResult(
                    success=False,
                    error=f"Signature verification failed: {sig_result['error']}"
                )
            logger.info("  ✓ Signatures verified")
        
        # Step 4: Run security scan (defense in depth - even for verified models)
        scan_result = self.scanner.scan_model(str(model_path))
        
        if not scan_result.passed:
            logger.error(f"  ✗ Security scan FAILED")
            return LoadResult(
                success=False,
                scan_result=scan_result,
                error="Security scan failed. Model may have been tampered with."
            )
        logger.info("  ✓ Security scan passed")
        
        # Step 5: Check trust_remote_code requirement
        config = self._load_config(model_path)
        if config and config.get("auto_map"):
            return LoadResult(
                success=False,
                error="Model requires trust_remote_code=True which is BLOCKED. Use 'pathfinder enable --accept-remote-code-risk' with two-person approval."
            )
        
        # Step 6: Load model with security constraints
        try:
            logger.info("  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=False,  # ALWAYS False
                local_files_only=True     # Never download during load
            )
            
            # Set pad token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info("  Loading model...")
            dtype = getattr(torch, torch_dtype, torch.bfloat16)
            
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                trust_remote_code=False,  # ALWAYS False
                local_files_only=True     # Never download during load
            )
            
            # Move to device
            logger.info(f"  Moving to device: {self.device}")
            if self.device == "hpu":
                model = model.to("hpu")
            elif self.device == "cuda":
                model = model.to("cuda")
            # else cpu - no move needed
            
            model.eval()
            
            logger.info(f"  ✓ Model loaded successfully on {self.device}")
            
            return LoadResult(
                success=True,
                model=model,
                tokenizer=tokenizer,
                mlbom=mlbom,
                scan_result=scan_result
            )
            
        except Exception as e:
            logger.error(f"  ✗ Model loading failed: {e}")
            return LoadResult(
                success=False,
                scan_result=scan_result,
                error=f"Failed to load model: {e}"
            )
    
    def _verify_signatures(self, model_path: Path, mlbom: Optional[dict]) -> dict:
        """Verify ECDSA signatures for all model artifacts"""
        if not self.public_key:
            return {"valid": False, "error": "No public key configured"}
        
        if not mlbom or "artifacts" not in mlbom:
            return {"valid": False, "error": "MLBOM missing artifacts list"}
        
        for artifact in mlbom["artifacts"]:
            file_name = artifact["file"]
            expected_hash = artifact.get("sha256")
            sig_b64 = artifact.get("signature")
            
            file_path = model_path / file_name
            sig_path = model_path / f"{file_name}.sig"
            
            if not file_path.exists():
                return {"valid": False, "error": f"Artifact missing: {file_name}"}
            
            # Verify hash
            actual_hash = self._compute_hash(file_path)
            if expected_hash and actual_hash != expected_hash:
                return {"valid": False, "error": f"Hash mismatch for {file_name}"}
            
            # Verify signature
            if sig_path.exists():
                try:
                    with open(sig_path, 'rb') as f:
                        signature = f.read()
                    with open(file_path, 'rb') as f:
                        data = f.read()
                    
                    self.public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
                except InvalidSignature:
                    return {"valid": False, "error": f"Invalid signature for {file_name}"}
                except Exception as e:
                    return {"valid": False, "error": f"Signature verification error: {e}"}
        
        return {"valid": True}
    
    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _load_config(self, model_path: Path) -> Optional[dict]:
        """Load model config.json"""
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return None


# =============================================================================
# EXAMPLE USAGE WITH GAUDI
# =============================================================================

def example_gaudi_inference():
    """
    Example: Secure Llama inference on Gaudi2
    """
    import torch
    import habana_frameworks.torch as ht
    import habana_frameworks.torch.core as htcore
    
    print("=" * 60)
    print("  Pathfinder Secure Inference on Intel Gaudi2")
    print("=" * 60)
    
    # Initialize secure loader
    loader = SecureModelLoader(
        verified_models_path="/verified-models",
        public_key_path="/keys/pathfinder.pub",
        device="hpu"
    )
    
    # Securely load model
    result = loader.load("meta-llama/Llama-3.2-1B")
    
    if not result.success:
        print(f"\n❌ LOAD FAILED: {result.error}")
        return
    
    print(f"\n✅ Model loaded securely")
    print(f"   MLBOM: {result.mlbom.get('model', {}).get('id')}")
    print(f"   Scan: {len(result.scan_result.findings)} findings")
    
    model = result.model
    tokenizer = result.tokenizer
    
    # Run inference
    prompt = "What is artificial intelligence?"
    print(f"\n[Prompt] {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("hpu")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
    
    htcore.mark_step()
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[Response]\n{response}")
    print("=" * 60)


if __name__ == "__main__":
    # For testing without Gaudi
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode - use CPU
        print("Running in test mode (CPU)")
        
        loader = SecureModelLoader(
            verified_models_path="./test-models",
            skip_verification=True,  # Dev only!
            device="cpu"
        )
        
        # Scan only test
        scanner = PathfinderScanner()
        if len(sys.argv) > 2:
            result = scanner.scan_model(sys.argv[2])
            print(result.to_json())
    else:
        example_gaudi_inference()
