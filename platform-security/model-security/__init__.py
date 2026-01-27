"""
Model Security Module

Core security components for AI model verification, scanning, and loading:
- ModelSecurityScanner: Multi-tool security scanning
- SecureModelLoader: Verified model loading with signature checks
- MLBOMGenerator: Machine Learning Bill of Materials generation
- ModelDownloader: Secure model download to quarantine
"""

from .scanner import ModelSecurityScanner, ScanResult, Finding, Severity
from .loader import SecureModelLoader, LoadResult
from .mlbom import MLBOMGenerator, MLBOM, FileEntry
from .downloader import ModelDownloader, DownloadResult

__all__ = [
    # Scanner
    "ModelSecurityScanner",
    "ScanResult",
    "Finding",
    "Severity",
    # Loader
    "SecureModelLoader",
    "LoadResult",
    # MLBOM
    "MLBOMGenerator",
    "MLBOM",
    "FileEntry",
    # Downloader
    "ModelDownloader",
    "DownloadResult",
]

__version__ = "1.0.0"
