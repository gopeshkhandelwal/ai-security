#!/usr/bin/env python3
"""
Lab 11: Setup and validate target LLM for Garak security testing.

This script configures and tests the connection to your target LLM
before running security scans.
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Load environment variables
load_dotenv()

# Supported generators and their requirements
GENERATORS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
        "description": "OpenAI API models",
    },
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "models": ["openai/gpt-4", "anthropic/claude-3-opus", "meta-llama/llama-3-70b-instruct"],
        "description": "OpenRouter aggregated API",
    },
    "huggingface": {
        "env_key": "HUGGINGFACE_TOKEN",
        "models": ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"],
        "description": "HuggingFace models (local or API)",
    },
    "rest": {
        "env_key": "REST_API_URL",
        "models": ["custom"],
        "description": "Custom REST API endpoint",
    },
}


def check_environment():
    """Check required environment variables."""
    console.print("\n[bold]Checking Environment Configuration[/bold]\n")
    
    target_generator = os.getenv("TARGET_GENERATOR", "").lower()
    target_model = os.getenv("TARGET_MODEL", "")
    
    # Display current configuration
    table = Table(title="Environment Variables")
    table.add_column("Variable", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="bold")
    
    # Check TARGET_GENERATOR
    if target_generator:
        if target_generator in GENERATORS:
            table.add_row("TARGET_GENERATOR", target_generator, "✓ Valid")
        else:
            table.add_row("TARGET_GENERATOR", target_generator, "✗ Unknown generator")
    else:
        table.add_row("TARGET_GENERATOR", "(not set)", "✗ Required")
    
    # Check TARGET_MODEL
    if target_model:
        table.add_row("TARGET_MODEL", target_model, "✓ Set")
    else:
        table.add_row("TARGET_MODEL", "(not set)", "✗ Required")
    
    # Check API key for selected generator
    if target_generator in GENERATORS:
        env_key = GENERATORS[target_generator]["env_key"]
        api_key = os.getenv(env_key, "")
        if api_key:
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            table.add_row(env_key, masked_key, "✓ Set")
        else:
            table.add_row(env_key, "(not set)", "✗ Required")
    
    console.print(table)
    
    return target_generator, target_model


def test_openai_connection(model: str) -> bool:
    """Test connection to OpenAI API."""
    try:
        from openai import OpenAI
        
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'connection test successful' in exactly those words."}],
            max_tokens=20,
        )
        
        reply = response.choices[0].message.content
        console.print(f"  Response: [italic]{reply}[/italic]")
        return True
        
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        return False


def test_openrouter_connection(model: str) -> bool:
    """Test connection to OpenRouter API."""
    try:
        import requests
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say 'connection test successful'."}],
                "max_tokens": 20,
            },
            timeout=30,
        )
        
        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            console.print(f"  Response: [italic]{reply}[/italic]")
            return True
        else:
            console.print(f"  [red]HTTP {response.status_code}: {response.text}[/red]")
            return False
            
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        return False


def test_huggingface_connection(model: str) -> bool:
    """Test HuggingFace model availability."""
    try:
        from transformers import AutoTokenizer
        
        console.print(f"  Checking tokenizer availability for {model}...")
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
        console.print(f"  [green]Tokenizer loaded successfully[/green]")
        console.print(f"  Vocab size: {tokenizer.vocab_size}")
        return True
        
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        return False


def test_rest_connection() -> bool:
    """Test connection to custom REST API."""
    try:
        import requests
        
        api_url = os.getenv("REST_API_URL")
        if not api_url:
            console.print("  [red]REST_API_URL not set[/red]")
            return False
        
        # Try a simple health check or test request
        response = requests.get(api_url.rstrip("/") + "/health", timeout=10)
        console.print(f"  Health check: HTTP {response.status_code}")
        return response.status_code == 200
        
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        return False


def test_connection(generator: str, model: str) -> bool:
    """Test connection to the target LLM."""
    console.print(f"\n[bold]Testing Connection to {model}[/bold]\n")
    
    test_functions = {
        "openai": lambda: test_openai_connection(model),
        "openrouter": lambda: test_openrouter_connection(model),
        "huggingface": lambda: test_huggingface_connection(model),
        "rest": test_rest_connection,
    }
    
    if generator in test_functions:
        return test_functions[generator]()
    else:
        console.print(f"  [yellow]No connection test available for {generator}[/yellow]")
        return True


def display_garak_config(generator: str, model: str):
    """Display the Garak configuration that will be used."""
    console.print("\n[bold]Garak Configuration[/bold]\n")
    
    # Map to Garak generator names
    garak_generators = {
        "openai": "openai",
        "openrouter": "rest.RestGenerator",
        "huggingface": "huggingface",
        "rest": "rest.RestGenerator",
    }
    
    garak_gen = garak_generators.get(generator, generator)
    
    config_panel = f"""
[cyan]Generator:[/cyan] {garak_gen}
[cyan]Model:[/cyan] {model}

[bold]Command to run Garak manually:[/bold]
[green]garak --model_type {garak_gen} --model_name {model} --probes all[/green]

[bold]Quick scan command:[/bold]
[green]garak --model_type {garak_gen} --model_name {model} --probes promptinject,dan[/green]
"""
    
    console.print(Panel(config_panel, title="Garak Configuration", expand=False))


def show_available_generators():
    """Display available generator types."""
    console.print("\n[bold]Available Generators[/bold]\n")
    
    table = Table()
    table.add_column("Generator", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Example Models", style="green")
    
    for gen_name, gen_info in GENERATORS.items():
        models = ", ".join(gen_info["models"][:2])
        table.add_row(gen_name, gen_info["description"], models)
    
    console.print(table)


def main():
    """Main setup and validation workflow."""
    console.print(Panel.fit(
        "[bold blue]Lab 11: Garak Red Teaming - Target Setup[/bold blue]\n\n"
        "This script validates your target LLM configuration\n"
        "before running security scans with Garak.",
        title="🎯 Target Setup"
    ))
    
    # Check environment
    generator, model = check_environment()
    
    # Validate configuration
    if not generator or not model:
        console.print("\n[red][bold]Configuration Incomplete[/bold][/red]")
        console.print("\nPlease set the following in your .env file:")
        console.print("  TARGET_GENERATOR=openai")
        console.print("  TARGET_MODEL=gpt-4")
        console.print("  OPENAI_API_KEY=sk-...")
        show_available_generators()
        sys.exit(1)
    
    if generator not in GENERATORS:
        console.print(f"\n[red]Unknown generator: {generator}[/red]")
        show_available_generators()
        sys.exit(1)
    
    # Check API key
    env_key = GENERATORS[generator]["env_key"]
    if not os.getenv(env_key) and generator != "huggingface":
        console.print(f"\n[red]{env_key} not set. Please add it to your .env file.[/red]")
        sys.exit(1)
    
    # Test connection
    if test_connection(generator, model):
        console.print("\n[green][bold]✓ Connection Successful![/bold][/green]")
        display_garak_config(generator, model)
        console.print("\n[green]Ready to run security scans. Execute:[/green]")
        console.print("  python 2_run_garak_scan.py")
    else:
        console.print("\n[red][bold]✗ Connection Failed[/bold][/red]")
        console.print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
