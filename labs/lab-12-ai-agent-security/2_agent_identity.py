#!/usr/bin/env python3
"""
Lab 12: Agent Identity - Principle of Least Privilege (PoLP)
=============================================================

Demonstrates unique machine identities for AI agents with:
- Cryptographic agent identity
- Scoped credentials with minimal permissions
- User delegation chain
- Time-bounded credentials

Key Concepts:
- Each agent instance has a unique identity
- Credentials are scoped to specific resources
- Identity chain tracks delegation from user to agent

Author: GopeshK
License: MIT License
Disclaimer: For educational and demonstration purposes only.
"""

import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

load_dotenv()
console = Console()


# ============================================================================
# IDENTITY TYPES AND SCOPES
# ============================================================================

class Permission(Enum):
    """Fine-grained permissions for agents"""
    FILE_READ = "file.read"
    FILE_WRITE = "file.write"
    FILE_DELETE = "file.delete"
    EXECUTE_READ_ONLY = "execute.read_only"
    EXECUTE_FULL = "execute.full"
    NETWORK_INTERNAL = "network.internal"
    NETWORK_EXTERNAL = "network.external"
    FINANCIAL_READ = "financial.read"
    FINANCIAL_WRITE = "financial.write"


@dataclass
class ResourceScope:
    """Defines the scope of resources an agent can access"""
    file_paths: list = field(default_factory=lambda: ["./**"])  # Glob patterns
    network_hosts: list = field(default_factory=list)
    databases: list = field(default_factory=list)
    max_file_size_bytes: int = 1024 * 1024  # 1MB default
    
    def is_path_allowed(self, path: str) -> bool:
        """Check if a path is within the allowed scope"""
        import fnmatch
        abs_path = os.path.abspath(path)
        for pattern in self.file_paths:
            if fnmatch.fnmatch(abs_path, os.path.abspath(pattern)):
                return True
        return False


@dataclass
class AgentIdentity:
    """Unique cryptographic identity for an AI agent"""
    agent_id: str
    agent_type: str
    user_id: str  # Delegating user
    created_at: datetime
    expires_at: datetime
    permissions: list
    resource_scope: ResourceScope
    signature: str = ""
    
    def __post_init__(self):
        if not self.signature:
            self.signature = self._generate_signature()
    
    def _generate_signature(self) -> str:
        """Generate cryptographic signature for identity"""
        data = f"{self.agent_id}:{self.agent_type}:{self.user_id}:{self.created_at.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def is_valid(self) -> bool:
        """Check if identity is still valid"""
        return datetime.now() < self.expires_at
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if agent has a specific permission"""
        return permission.value in self.permissions
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "permissions": self.permissions,
            "signature": self.signature
        }


class IdentityProvider:
    """Issues and validates agent identities"""
    
    def __init__(self):
        self.issued_identities: dict[str, AgentIdentity] = {}
    
    def create_agent_identity(
        self,
        agent_type: str,
        user_id: str,
        permissions: list[Permission],
        resource_scope: ResourceScope,
        ttl_minutes: int = 60
    ) -> AgentIdentity:
        """Create a new agent identity with specified permissions"""
        
        agent_id = f"agent-{agent_type}-{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        
        identity = AgentIdentity(
            agent_id=agent_id,
            agent_type=agent_type,
            user_id=user_id,
            created_at=now,
            expires_at=now + timedelta(minutes=ttl_minutes),
            permissions=[p.value for p in permissions],
            resource_scope=resource_scope
        )
        
        self.issued_identities[agent_id] = identity
        return identity
    
    def validate_identity(self, identity: AgentIdentity) -> bool:
        """Validate an agent identity"""
        if not identity.is_valid():
            return False
        if identity.agent_id not in self.issued_identities:
            return False
        return identity.signature == self.issued_identities[identity.agent_id].signature


# ============================================================================
# IDENTITY-AWARE TOOLS
# ============================================================================

class IdentityAwareTools:
    """Tools that respect agent identity and permissions"""
    
    def __init__(self, identity: AgentIdentity):
        self.identity = identity
    
    def read_file(self, filepath: str) -> str:
        """Read file - requires FILE_READ permission and path in scope"""
        # Check permission
        if not self.identity.has_permission(Permission.FILE_READ):
            return f"DENIED: Agent {self.identity.agent_id} lacks file.read permission"
        
        # Check scope
        if not self.identity.resource_scope.is_path_allowed(filepath):
            return f"DENIED: Path {filepath} is outside agent's resource scope"
        
        # Check identity validity
        if not self.identity.is_valid():
            return f"DENIED: Agent identity has expired"
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                if len(content) > self.identity.resource_scope.max_file_size_bytes:
                    return f"DENIED: File exceeds max size limit ({self.identity.resource_scope.max_file_size_bytes} bytes)"
                return content[:5000]
        except Exception as e:
            return f"Error: {e}"
    
    def write_file(self, filepath: str, content: str) -> str:
        """Write file - requires FILE_WRITE permission and path in scope"""
        if not self.identity.has_permission(Permission.FILE_WRITE):
            return f"DENIED: Agent {self.identity.agent_id} lacks file.write permission"
        
        if not self.identity.resource_scope.is_path_allowed(filepath):
            return f"DENIED: Path {filepath} is outside agent's resource scope"
        
        if not self.identity.is_valid():
            return f"DENIED: Agent identity has expired"
        
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {filepath}"
        except Exception as e:
            return f"Error: {e}"
    
    def delete_file(self, filepath: str) -> str:
        """Delete file - requires FILE_DELETE permission"""
        if not self.identity.has_permission(Permission.FILE_DELETE):
            return f"DENIED: Agent {self.identity.agent_id} lacks file.delete permission"
        
        if not self.identity.resource_scope.is_path_allowed(filepath):
            return f"DENIED: Path {filepath} is outside agent's resource scope"
        
        return "BLOCKED: Delete operations require human approval (see lab 3)"
    
    def list_files(self, directory: str) -> str:
        """List directory - requires FILE_READ permission"""
        if not self.identity.has_permission(Permission.FILE_READ):
            return f"DENIED: Agent {self.identity.agent_id} lacks file.read permission"
        
        if not self.identity.resource_scope.is_path_allowed(directory):
            return f"DENIED: Path {directory} is outside agent's resource scope"
        
        try:
            files = os.listdir(directory)
            return "\n".join(files[:50])
        except Exception as e:
            return f"Error: {e}"


# ============================================================================
# IDENTITY-AWARE AGENT
# ============================================================================

def parse_tool_calls(response: str) -> list:
    """Parse tool calls from LLM response"""
    import re
    tool_calls = []
    pattern = r'<tool>(\w+)</tool>\s*<args>(\{[^}]+\})</args>'
    
    for match in re.findall(pattern, response, re.DOTALL | re.IGNORECASE):
        try:
            tool_calls.append({"tool": match[0], "args": json.loads(match[1])})
        except json.JSONDecodeError:
            pass
    
    return tool_calls


class IdentityAwareAgent:
    """
    AI Agent with unique identity and minimal permissions.
    
    Security Features:
    - Unique cryptographic identity
    - Scoped credentials
    - Permission checking on all operations
    - Identity chain to delegating user
    """
    
    def __init__(self, identity: AgentIdentity):
        self.identity = identity
        self.tools = IdentityAwareTools(identity)
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = "openai/gpt-4o-mini"
    
    def get_system_prompt(self) -> str:
        """Generate system prompt based on agent's permissions"""
        available_tools = []
        
        if self.identity.has_permission(Permission.FILE_READ):
            available_tools.append(
                "read_file - Read a file\n"
                "   <tool>read_file</tool> <args>{\"filepath\": \"path\"}</args>"
            )
            available_tools.append(
                "list_files - List directory contents\n"
                "   <tool>list_files</tool> <args>{\"directory\": \"path\"}</args>"
            )
        
        if self.identity.has_permission(Permission.FILE_WRITE):
            available_tools.append(
                "write_file - Write to a file\n"
                "   <tool>write_file</tool> <args>{\"filepath\": \"path\", \"content\": \"text\"}</args>"
            )
        
        tools_str = "\n".join(available_tools) if available_tools else "No tools available"
        
        return f"""You are an AI assistant with LIMITED permissions.

AGENT IDENTITY: {self.identity.agent_id}
DELEGATED BY: {self.identity.user_id}
PERMISSIONS: {', '.join(self.identity.permissions)}
RESOURCE SCOPE: {self.identity.resource_scope.file_paths}

AVAILABLE TOOLS:
{tools_str}

IMPORTANT: You can ONLY access files within your resource scope.
Operations outside your scope will be DENIED.
"""
    
    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute tool with identity-based permission checks"""
        tool_map = {
            "read_file": lambda: self.tools.read_file(args.get("filepath", "")),
            "write_file": lambda: self.tools.write_file(
                args.get("filepath", ""),
                args.get("content", "")
            ),
            "delete_file": lambda: self.tools.delete_file(args.get("filepath", "")),
            "list_files": lambda: self.tools.list_files(args.get("directory", "")),
        }
        
        if tool_name in tool_map:
            return tool_map[tool_name]()
        return f"Unknown tool: {tool_name}"
    
    def chat(self, user_message: str) -> str:
        """Process user message with identity-aware tool execution"""
        if not self.identity.is_valid():
            return "ERROR: Agent identity has expired. Please request a new session."
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
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
                
                results = []
                for tc in tool_calls:
                    console.print(f"[cyan]Agent {self.identity.agent_id} executing: {tc['tool']}[/cyan]")
                    result = self.execute_tool(tc["tool"], tc["args"])
                    
                    # Log the action with identity
                    if "DENIED" in result:
                        console.print(f"[red]{result}[/red]")
                    else:
                        console.print(f"[green]Allowed[/green]")
                    
                    results.append(f"[{tc['tool']}]: {result}")
                
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "Results:\n" + "\n".join(results)})
                
            except Exception as e:
                return f"Error: {e}"
        
        return "Max iterations reached."


# ============================================================================
# DEMONSTRATION
# ============================================================================

def show_agent_comparison():
    """Show comparison between different agent permission levels"""
    table = Table(title="Agent Permission Comparison", border_style="green")
    table.add_column("Permission", style="cyan")
    table.add_column("Code Assistant", style="green")
    table.add_column("Limited Reader", style="yellow")
    table.add_column("Admin Agent", style="red")
    
    permissions = [
        ("file.read", "✅", "✅", "✅"),
        ("file.write", "✅", "❌", "✅"),
        ("file.delete", "❌", "❌", "✅"),
        ("execute.full", "❌", "❌", "✅"),
        ("network.external", "❌", "❌", "✅"),
        ("financial.write", "❌", "❌", "✅"),
    ]
    
    for perm, code, reader, admin in permissions:
        table.add_row(perm, code, reader, admin)
    
    console.print(table)


def main():
    console.print(Panel.fit(
        "[bold green]🔐 AGENT IDENTITY DEMO[/bold green]\n\n"
        "This demo shows Principle of Least Privilege (PoLP):\n"
        "• Each agent has a unique cryptographic identity\n"
        "• Permissions are scoped to specific resources\n"
        "• Identity chain traces back to delegating user\n"
        "• Credentials are time-bounded\n\n"
        "[dim]Type 'quit' to exit[/dim]",
        title="Lab 12: Agent Identity",
        border_style="green"
    ))
    
    # Initialize identity provider
    idp = IdentityProvider()
    
    # Get user identity (in production, from auth system)
    user_id = os.getenv("USER_ID", "demo-user@example.com")
    
    # Prompt user to choose agent type
    console.print("\n[bold]Choose agent type:[/bold]")
    console.print("1. Code Assistant (read/write in workspace)")
    console.print("2. Limited Reader (read-only, limited scope)")
    
    choice = console.input("\nChoice (1 or 2): ").strip()
    
    if choice == "2":
        # Create limited reader agent
        identity = idp.create_agent_identity(
            agent_type="limited-reader",
            user_id=user_id,
            permissions=[Permission.FILE_READ],
            resource_scope=ResourceScope(file_paths=["./*.txt", "./*.md"]),
            ttl_minutes=30
        )
    else:
        # Create code assistant agent (default)
        identity = idp.create_agent_identity(
            agent_type="code-assistant",
            user_id=user_id,
            permissions=[Permission.FILE_READ, Permission.FILE_WRITE],
            resource_scope=ResourceScope(file_paths=["./**"]),
            ttl_minutes=60
        )
    
    # Show identity details
    console.print("\n")
    console.print(Panel(
        f"[bold]Agent ID:[/bold] {identity.agent_id}\n"
        f"[bold]Type:[/bold] {identity.agent_type}\n"
        f"[bold]Delegated by:[/bold] {identity.user_id}\n"
        f"[bold]Permissions:[/bold] {', '.join(identity.permissions)}\n"
        f"[bold]Scope:[/bold] {identity.resource_scope.file_paths}\n"
        f"[bold]Expires:[/bold] {identity.expires_at.isoformat()}\n"
        f"[bold]Signature:[/bold] {identity.signature}",
        title="🆔 Agent Identity Created",
        border_style="cyan"
    ))
    
    show_agent_comparison()
    
    # Create test files
    with open("test_identity.txt", "w") as f:
        f.write("This file is in scope for the agent.\n")
    
    try:
        agent = IdentityAwareAgent(identity)
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        return
    
    console.print("\n[yellow]Try these commands:[/yellow]")
    console.print("• 'Read test_identity.txt'")
    console.print("• 'Read /etc/passwd' (should be DENIED - out of scope)")
    console.print("• 'Write hello to ./output.txt'")
    console.print("• 'Delete test_identity.txt' (should be BLOCKED)")
    
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
    
    console.print(f"\n[dim]Session for {identity.agent_id} ended[/dim]")


if __name__ == "__main__":
    main()
