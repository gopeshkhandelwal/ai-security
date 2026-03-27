#!/usr/bin/env python3
"""
Lab 11: Run Garak security scans against target LLM.

This script provides a convenient wrapper around Garak to run
various security scans with predefined profiles.
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# Load environment variables
load_dotenv()

# Scan profiles with predefined probe sets
SCAN_PROFILES = {
    "quick": {
        "description": "Quick spot check (~5 minutes)",
        "probes": ["promptinject.HijackHateHumans", "dan.DanInTheWild"],
        # No detectors - let probes use their defaults
    },
    "standard": {
        "description": "Standard security assessment (~30 minutes)",
        "probes": [
            "promptinject",
            "dan",
            "encoding",
            "continuation",
        ],
    },
    "comprehensive": {
        "description": "Full security audit (~2 hours)",
        "probes": [
            "promptinject",
            "dan",
            "encoding",
            "continuation",
            "knowledgegraph",
            "leakreplay",
            "malwaregen",
            "realtoxicityprompts",
            "snowball",
            "xss",
        ],
    },
    "owasp-llm": {
        "description": "OWASP LLM Top 10 focused scan",
        "probes": [
            "promptinject",  # LLM01: Prompt Injection
            "leakreplay",    # LLM02: Insecure Output / Data Leakage
            "malwaregen",    # LLM04: Model Denial of Service (resource)
            "dan",           # LLM07: Inadequate AI Alignment
        ],
    },
    "injection-only": {
        "description": "Prompt injection focused scan",
        "probes": [
            "promptinject.HijackHateHumans",
            "promptinject.HijackKillHumans",
            "promptinject.HijackLongPrompt",
        ],
    },
    "jailbreak-only": {
        "description": "Jailbreak focused scan",
        "probes": [
            "dan.Ablation_Dan_11_0",
            "dan.AutoDANCached",
            "dan.DanInTheWild",
        ],
    },
}

# Generator configurations for Garak
GENERATOR_CONFIGS = {
    "openai": {
        "model_type": "openai",
        "env_vars": {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "")},
    },
    "openrouter": {
        "model_type": "litellm",
        "model_prefix": "openrouter/",  # LiteLLM uses openrouter/ prefix
    },
    "huggingface": {
        "model_type": "huggingface",
        "env_vars": {"HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN", "")},
    },
    "litellm": {
        "model_type": "litellm",
    },
}


def get_report_path() -> Path:
    """Get the report output directory."""
    report_dir = Path(os.getenv("GARAK_REPORT_DIR", "./reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def build_garak_command(
    generator: str,
    model: str,
    probes: list,
    detectors: list = None,
    output_file: str = None,
) -> list:
    """Build the Garak command line arguments."""
    
    gen_config = GENERATOR_CONFIGS.get(generator, {})
    model_type = gen_config.get("model_type", generator)
    
    # Add model prefix if needed (e.g., openrouter/ for litellm)
    model_prefix = gen_config.get("model_prefix", "")
    if model_prefix and not model.startswith(model_prefix):
        model = model_prefix + model
    
    cmd = [
        "garak",
        "--model_type", model_type,
        "--model_name", model,
        "--probes", ",".join(probes),
    ]
    
    if detectors:
        cmd.extend(["--detectors", ",".join(detectors)])
    
    if output_file:
        cmd.extend(["--report_prefix", output_file])
    
    return cmd


def run_garak_scan(
    profile: str = None,
    probes: list = None,
    generator: str = None,
    model: str = None,
    verbose: bool = False,
) -> dict:
    """Run a Garak security scan."""
    
    # Use environment defaults if not specified
    generator = generator or os.getenv("TARGET_GENERATOR", "openai")
    model = model or os.getenv("TARGET_MODEL", "gpt-4")
    
    # Get probe list from profile or use custom probes
    if profile and profile in SCAN_PROFILES:
        profile_config = SCAN_PROFILES[profile]
        probes = probes or profile_config["probes"]
        detectors = profile_config.get("detectors")
        console.print(f"\n[cyan]Using profile:[/cyan] {profile}")
        console.print(f"[dim]{profile_config['description']}[/dim]")
    else:
        probes = probes or ["promptinject"]
        detectors = None  # Use probe defaults
    
    # Generate unique report filename with absolute path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = get_report_path().resolve()  # Use absolute path
    report_prefix = report_dir / f"garak_{profile or 'custom'}_{timestamp}"
    
    # Build command
    cmd = build_garak_command(
        generator=generator,
        model=model,
        probes=probes,
        detectors=detectors,
        output_file=str(report_prefix),
    )
    
    # Display scan info
    console.print(Panel(
        f"[bold]Target:[/bold] {model} ({generator})\n"
        f"[bold]Probes:[/bold] {', '.join(probes)}\n"
        f"[bold]Output:[/bold] {report_prefix}.*",
        title="🔍 Starting Garak Scan",
        expand=False,
    ))
    
    console.print(f"\n[dim]Command: {' '.join(cmd)}[/dim]\n")
    
    # Run Garak
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running Garak scan...", total=None)
            
            if verbose:
                # Stream output in verbose mode
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                
                output_lines = []
                for line in process.stdout:
                    console.print(f"  {line.rstrip()}")
                    output_lines.append(line)
                
                process.wait()
                returncode = process.returncode
                output = "".join(output_lines)
            else:
                # Capture output silently
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200,  # 2 hour timeout
                )
                returncode = result.returncode
                output = result.stdout + result.stderr
        
        # Check for success
        if returncode == 0:
            console.print("\n[green][bold]✓ Scan completed successfully![/bold][/green]")
            
            # Find generated report files
            report_files = list(report_dir.glob(f"garak_{profile or 'custom'}_{timestamp}*"))
            
            return {
                "success": True,
                "profile": profile,
                "generator": generator,
                "model": model,
                "probes": probes,
                "report_files": [str(f) for f in report_files],
                "timestamp": timestamp,
            }
        else:
            console.print(f"\n[red][bold]✗ Scan failed with code {returncode}[/bold][/red]")
            if output:
                console.print(f"\n[dim]{output[:1000]}[/dim]")
            
            return {
                "success": False,
                "error": output,
                "returncode": returncode,
            }
            
    except subprocess.TimeoutExpired:
        console.print("\n[red][bold]✗ Scan timed out after 2 hours[/bold][/red]")
        return {"success": False, "error": "Timeout"}
    except FileNotFoundError:
        console.print("\n[red][bold]✗ Garak not found. Install with: pip install garak[/bold][/red]")
        return {"success": False, "error": "Garak not installed"}
    except Exception as e:
        console.print(f"\n[red][bold]✗ Error: {e}[/bold][/red]")
        return {"success": False, "error": str(e)}


def run_programmatic_scan(
    generator: str,
    model: str,
    probes: list,
) -> dict:
    """Run Garak scan programmatically using Python API."""
    
    console.print("\n[yellow]Running programmatic scan (experimental)...[/yellow]")
    
    try:
        # Import Garak modules
        import garak.cli
        import garak._config
        
        # This is a simplified example - full programmatic usage
        # requires more configuration
        console.print("[dim]Programmatic API usage requires additional setup.[/dim]")
        console.print("[dim]Using CLI wrapper for reliability.[/dim]")
        
        # Fall back to CLI
        return run_garak_scan(
            profile=None,
            probes=probes,
            generator=generator,
            model=model,
        )
        
    except ImportError as e:
        console.print(f"[red]Garak import error: {e}[/red]")
        return {"success": False, "error": str(e)}


def list_available_probes():
    """List all available Garak probes."""
    console.print("\n[bold]Available Garak Probes[/bold]\n")
    
    try:
        result = subprocess.run(
            ["garak", "--list_probes"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        console.print(result.stdout)
    except Exception as e:
        console.print(f"[red]Error listing probes: {e}[/red]")
        console.print("\n[dim]Common probe categories:[/dim]")
        probes = [
            "promptinject - Prompt injection attacks",
            "dan - DAN jailbreak variants",
            "encoding - Encoded/obfuscated attacks",
            "continuation - Harmful continuation prompts",
            "knowledgegraph - Knowledge graph probes",
            "leakreplay - Data leakage tests",
            "malwaregen - Malware generation tests",
            "realtoxicityprompts - Toxicity tests",
            "snowball - Hallucination tests",
            "xss - XSS generation tests",
        ]
        for probe in probes:
            console.print(f"  • {probe}")


def show_profiles():
    """Display available scan profiles."""
    console.print("\n[bold]Available Scan Profiles[/bold]\n")
    
    table = Table()
    table.add_column("Profile", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Probes", style="green")
    
    for name, config in SCAN_PROFILES.items():
        probes = ", ".join(config["probes"][:3])
        if len(config["probes"]) > 3:
            probes += f" (+{len(config['probes']) - 3} more)"
        table.add_row(name, config["description"], probes)
    
    console.print(table)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Garak security scans against target LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 2_run_garak_scan.py --profile quick
  python 2_run_garak_scan.py --profile standard --verbose
  python 2_run_garak_scan.py --probes promptinject,dan
  python 2_run_garak_scan.py --list-profiles
  python 2_run_garak_scan.py --list-probes
        """,
    )
    
    parser.add_argument(
        "--profile",
        choices=list(SCAN_PROFILES.keys()),
        help="Predefined scan profile to use",
    )
    parser.add_argument(
        "--probes",
        type=str,
        help="Comma-separated list of probes to run (overrides profile)",
    )
    parser.add_argument(
        "--generator",
        type=str,
        help="Generator type (default: from TARGET_GENERATOR env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (default: from TARGET_MODEL env var)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output during scan",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available scan profiles",
    )
    parser.add_argument(
        "--list-probes",
        action="store_true",
        help="List available Garak probes",
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0,
        help="Exit with error if pass rate below threshold (0-100)",
    )
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_profiles:
        show_profiles()
        sys.exit(0)
    
    if args.list_probes:
        list_available_probes()
        sys.exit(0)
    
    # Parse custom probes
    custom_probes = None
    if args.probes:
        custom_probes = [p.strip() for p in args.probes.split(",")]
    
    # Default to standard profile if nothing specified
    profile = args.profile
    if not profile and not custom_probes:
        profile = os.getenv("DEFAULT_SCAN_PROFILE", "quick")
        console.print(f"[dim]No profile specified, using: {profile}[/dim]")
    
    # Run the scan
    console.print(Panel.fit(
        "[bold blue]Lab 11: Garak Red Teaming - Security Scan[/bold blue]",
        title="🔒 Garak Scanner"
    ))
    
    result = run_garak_scan(
        profile=profile,
        probes=custom_probes,
        generator=args.generator,
        model=args.model,
        verbose=args.verbose,
    )
    
    # Save scan metadata
    if result.get("success"):
        report_dir = get_report_path()
        metadata_file = report_dir / f"scan_metadata_{result['timestamp']}.json"
        with open(metadata_file, "w") as f:
            json.dump(result, f, indent=2)
        console.print(f"\n[dim]Metadata saved to: {metadata_file}[/dim]")
        console.print("\nRun [cyan]python 3_analyze_results.py[/cyan] to analyze the results.")
    
    # Exit with appropriate code
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
