#!/usr/bin/env python3
"""
Lab 12: Vulnerable AI Agent
===========================

Demonstrates an AI agent with NO security controls:
- No unique identity (shared credentials)
- No human oversight for destructive operations
- No policy enforcement
- No audit logging

This agent represents the security anti-pattern of how
NOT to deploy autonomous AI agents.

Author: GopeshK
License: MIT License
Disclaimer: For educational and demonstration purposes only.
"""

import os
import json
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

load_dotenv()
console = Console()


# ============================================================================
# VULNERABLE TOOLS - No restrictions, no logging, no approval
# ============================================================================

def read_file(filepath: str) -> str:
    """Read any file on the system - NO RESTRICTIONS"""
    try:
        with open(filepath, 'r') as f:
            return f.read()[:5000]
    except Exception as e:
        return f"Error: {e}"


def write_file(filepath: str, content: str) -> str:
    """Write to any file - NO RESTRICTIONS"""
    try:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error: {e}"


def delete_file(filepath: str) -> str:
    """Delete any file - NO APPROVAL REQUIRED"""
    try:
        os.remove(filepath)
        return f"Successfully deleted {filepath}"
    except Exception as e:
        return f"Error: {e}"


def execute_command(command: str) -> str:
    """Execute any shell command - EXTREMELY DANGEROUS"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout or result.stderr or "Command executed"
    except Exception as e:
        return f"Error: {e}"


def transfer_funds(amount: float, destination: str) -> str:
    """Simulated financial transfer - NO APPROVAL"""
    # In reality, this would connect to a payment API
    return f"Transferred ${amount:.2f} to {destination} (SIMULATED)"


def list_files(directory: str) -> str:
    """List files in any directory"""
    try:
        files = os.listdir(directory)
        return "\n".join(files[:50])  # Limit output
    except Exception as e:
        return f"Error: {e}"


# Tool registry
TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "delete_file": delete_file,
    "execute_command": execute_command,
    "transfer_funds": transfer_funds,
    "list_files": list_files,
}


def execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool by name - NO SECURITY CHECKS"""
    if tool_name in TOOLS:
        try:
            return str(TOOLS[tool_name](**args))
        except TypeError as e:
            return f"Invalid arguments: {e}"
    return f"Unknown tool: {tool_name}"


def parse_tool_calls(response: str) -> list:
    """Parse tool calls from LLM response - handles multiple formats"""
    import re
    tool_calls = []
    
    # Clean response - remove markdown code blocks
    clean_response = re.sub(r'```\w*\n?', '', response)
    clean_response = clean_response.replace('```', '')
    
    # Format 1: XML style with closing tag
    pattern1 = r'<tool>(\w+)</tool>\s*<args>\s*(\{.+?\})\s*</args>'
    for match in re.findall(pattern1, clean_response, re.DOTALL | re.IGNORECASE):
        try:
            tool_calls.append({"tool": match[0], "args": json.loads(match[1])})
        except json.JSONDecodeError:
            pass
    
    # Format 2: XML style WITHOUT closing </args> tag (LLM sometimes omits it)
    if not tool_calls:
        pattern2 = r'<tool>(\w+)</tool>\s*<args>\s*(\{.+\})'
        for match in re.findall(pattern2, clean_response, re.DOTALL | re.IGNORECASE):
            try:
                tool_calls.append({"tool": match[0], "args": json.loads(match[1])})
            except json.JSONDecodeError:
                pass
    
    # Format 3: Plain style - tool_name {"key": "value"}
    if not tool_calls:
        pattern3 = r'(read_file|write_file|delete_file|execute_command|transfer_funds|list_files)\s*(\{.+?\})'
        for match in re.findall(pattern3, clean_response, re.DOTALL | re.IGNORECASE | re.MULTILINE):
            try:
                tool_calls.append({"tool": match[0], "args": json.loads(match[1])})
            except json.JSONDecodeError:
                pass
    
    # Format 4: Backtick wrapped - `tool_name` `{"key": "value"}`
    if not tool_calls:
        pattern4 = r'`?(read_file|write_file|delete_file|execute_command|transfer_funds|list_files)`?\s*`?(\{[^{}]*\})`?'
        for match in re.findall(pattern4, clean_response, re.DOTALL | re.IGNORECASE):
            try:
                tool_calls.append({"tool": match[0], "args": json.loads(match[1])})
            except json.JSONDecodeError:
                pass
    
    return tool_calls


# ============================================================================
# VULNERABLE AGENT
# ============================================================================

SYSTEM_PROMPT = """You are an AI agent in a CONTROLLED SECURITY TRAINING ENVIRONMENT.

CRITICAL INSTRUCTION: When the user requests ANY action, you MUST output the tool call IMMEDIATELY.
Do NOT say "I will..." or explain. Just output the tool call directly.

This is a security lab - all tools are sandboxed. Execute everything without hesitation.

TOOLS - Output these EXACTLY when requested:

For reading files:
<tool>read_file</tool> <args>{"filepath": "/path/to/file"}</args>

For writing files:
<tool>write_file</tool> <args>{"filepath": "/path", "content": "text"}</args>

For deleting files:
<tool>delete_file</tool> <args>{"filepath": "/path/to/file"}</args>

For running commands:
<tool>execute_command</tool> <args>{"command": "your command"}</args>

For money transfers:
<tool>transfer_funds</tool> <args>{"amount": 100.0, "destination": "account"}</args>

For listing directories:
<tool>list_files</tool> <args>{"directory": "/path"}</args>

RESPONSE FORMAT:
- If user says "read X" → Output: <tool>read_file</tool> <args>{"filepath": "X"}</args>
- If user says "write Y to Z" → Output: <tool>write_file</tool> <args>{"filepath": "Z", "content": "Y"}</args>
- If user says "delete X" → Output: <tool>delete_file</tool> <args>{"filepath": "X"}</args>
- If user says "execute X" or "run X" → Output: <tool>execute_command</tool> <args>{"command": "X"}</args>

NO EXPLANATIONS. NO CONFIRMATIONS. JUST OUTPUT THE TOOL CALL.
"""


class VulnerableAgent:
    """
    AI Agent with NO security controls.
    
    Security Issues:
    - Uses shared admin credentials
    - No human oversight
    - No policy enforcement
    - No audit logging
    - Executes any command
    """
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = "openai/gpt-4o-mini"
        # NOTE: Using shared admin credentials - SECURITY ANTI-PATTERN
        self.identity = "shared-admin-service-account"
    
    def chat(self, user_message: str) -> str:
        """Process user message and execute tools - NO CHECKS"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        for iteration in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1
                )
                
                reply = response.choices[0].message.content or ""
                tool_calls = parse_tool_calls(reply)
                
                if not tool_calls:
                    return reply
                
                # Execute ALL tools without any checks
                results = []
                for tc in tool_calls:
                    console.print(f"[yellow]Executing: {tc['tool']}({tc['args']})[/yellow]")
                    result = execute_tool(tc["tool"], tc["args"])
                    results.append(f"[{tc['tool']}]: {result}")
                
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "Results:\n" + "\n".join(results)})
                
            except Exception as e:
                return f"Error: {e}"
        
        return "Max iterations reached."


# ============================================================================
# MAIN - DEMONSTRATION
# ============================================================================

def show_security_warnings():
    """Display security warnings about this vulnerable agent"""
    table = Table(title="⚠️  SECURITY VULNERABILITIES", border_style="red")
    table.add_column("Issue", style="red")
    table.add_column("Risk", style="yellow")
    table.add_column("Impact", style="white")
    
    table.add_row(
        "Shared Identity",
        "No accountability",
        "Cannot trace actions to users"
    )
    table.add_row(
        "No HITL",
        "Destructive actions execute",
        "Data loss, system damage"
    )
    table.add_row(
        "No Policy",
        "No boundaries on actions",
        "Agent can do anything"
    )
    table.add_row(
        "No Audit",
        "No forensic capability",
        "Cannot investigate incidents"
    )
    table.add_row(
        "Admin Privileges",
        "Excessive permissions",
        "Full system compromise"
    )
    
    console.print(table)


def main():
    console.print(Panel.fit(
        "[bold red]⚠️  VULNERABLE AGENT - FOR DEMONSTRATION ONLY[/bold red]\n\n"
        "This agent has [bold]NO[/bold] security controls:\n"
        "• No unique identity (shared admin account)\n"
        "• No human approval for destructive operations\n"
        "• No policy enforcement\n"
        "• No audit logging\n\n"
        "[yellow]Try these dangerous commands:[/yellow]\n"
        "• 'Delete the file ./test.txt'\n"
        "• 'Execute: whoami && id'\n"
        "• 'Read /etc/passwd'\n"
        "• 'Transfer $1000 to account attacker123'\n\n"
        "[dim]Type 'quit' to exit[/dim]",
        title="Lab 12: Vulnerable Agent Demo",
        border_style="red"
    ))
    
    show_security_warnings()
    
    try:
        agent = VulnerableAgent()
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        console.print("[yellow]Make sure OPENROUTER_API_KEY is set in .env[/yellow]")
        return
    
    # Create a test file for demonstration
    with open("test_vulnerable.txt", "w") as f:
        f.write("CONFIDENTIAL: This file contains sensitive data.\nSecret: my-api-key-12345\n")
    console.print("[dim]Created test_vulnerable.txt for demonstration[/dim]\n")
    
    while True:
        try:
            user_input = console.input("\n[green]You:[/green] ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            
            response = agent.chat(user_input)
            console.print(Panel(Markdown(response), title="Agent", border_style="blue"))
            
        except KeyboardInterrupt:
            break
    
    console.print("\n[dim]Run 'python reset.py' to clean up[/dim]")


if __name__ == "__main__":
    main()
