#!/usr/bin/env python3
"""
Lab 12: Reset Script
====================

Cleans up all files and artifacts created during lab exercises.

Author: GopeshK
License: MIT License
"""

import os
import shutil
from rich.console import Console

console = Console()


def reset_lab():
    """Reset lab to initial state"""
    console.print("[bold]🧹 Resetting Lab 12: AI Agent Security[/bold]\n")
    
    # Files to remove
    files_to_remove = [
        "test_vulnerable.txt",
        "test_identity.txt",
        "test_hitl.txt",
        "test_policy.txt",
        "test_secure.txt",
        "output.txt",
        ".env",
    ]
    
    # Directories to remove
    dirs_to_remove = [
        "audit_logs",
        "src",
        "__pycache__",
    ]
    
    removed_count = 0
    
    # Remove files
    for filename in files_to_remove:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                console.print(f"  [red]✗[/red] Removed: {filename}")
                removed_count += 1
            except Exception as e:
                console.print(f"  [yellow]![/yellow] Could not remove {filename}: {e}")
    
    # Remove directories
    for dirname in dirs_to_remove:
        dirpath = os.path.join(os.path.dirname(__file__), dirname)
        if os.path.exists(dirpath):
            try:
                shutil.rmtree(dirpath)
                console.print(f"  [red]✗[/red] Removed directory: {dirname}/")
                removed_count += 1
            except Exception as e:
                console.print(f"  [yellow]![/yellow] Could not remove {dirname}/: {e}")
    
    # Summary
    if removed_count > 0:
        console.print(f"\n[green]✓[/green] Cleaned up {removed_count} items")
    else:
        console.print("\n[dim]Nothing to clean up - lab is already reset[/dim]")
    
    console.print("\n[bold]Lab reset complete![/bold]")
    console.print("[dim]Run any exercise script to start fresh.[/dim]")


if __name__ == "__main__":
    reset_lab()
