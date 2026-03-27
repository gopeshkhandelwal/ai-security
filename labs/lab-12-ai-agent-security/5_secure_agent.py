#!/usr/bin/env python3
"""
Lab 12: Fully Secured AI Agent
==============================

Demonstrates a production-ready AI agent with all 5 security principles:
1. Principle of Least Privilege (PoLP) - Unique identity, minimal permissions
2. Human-in-the-Loop (HITL) - Approval for high-risk actions
3. Policy as Code (PaC) - OPA-style policy enforcement
4. Autonomy Boundaries - Explicit action constraints
5. Auditability - Complete action traceability

This agent represents the security best practices for deploying
autonomous AI agents in enterprise environments.

Author: GopeshK
License: MIT License
Disclaimer: For educational and demonstration purposes only.
"""

import os
import json
import uuid
import hashlib
import fnmatch
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm
from rich.markdown import Markdown

load_dotenv()
console = Console()


# ============================================================================
# 1. IDENTITY LAYER (PoLP)
# ============================================================================

class Permission(Enum):
    """Fine-grained permissions"""
    FILE_READ = "file.read"
    FILE_WRITE = "file.write"
    FILE_DELETE = "file.delete"
    EXECUTE_READONLY = "execute.readonly"
    EXECUTE_FULL = "execute.full"
    NETWORK_INTERNAL = "network.internal"
    NETWORK_EXTERNAL = "network.external"


@dataclass
class ResourceScope:
    """Defines allowed resource scope"""
    file_paths: list = field(default_factory=lambda: ["./**"])
    exclude_paths: list = field(default_factory=lambda: ["**/.env", "**/*secret*"])
    max_file_size_bytes: int = 1024 * 1024
    
    def is_path_allowed(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        # Check exclusions first
        for pattern in self.exclude_paths:
            if fnmatch.fnmatch(abs_path, os.path.abspath(pattern)) or fnmatch.fnmatch(path, pattern):
                return False
        # Check inclusions
        for pattern in self.file_paths:
            if fnmatch.fnmatch(abs_path, os.path.abspath(pattern)) or fnmatch.fnmatch(path, pattern):
                return True
        return False


@dataclass
class AgentIdentity:
    """Unique cryptographic identity for the agent"""
    agent_id: str
    agent_type: str
    user_id: str
    session_id: str
    created_at: datetime
    expires_at: datetime
    permissions: list
    resource_scope: ResourceScope
    signature: str = ""
    
    def __post_init__(self):
        if not self.signature:
            data = f"{self.agent_id}:{self.session_id}:{self.created_at.isoformat()}"
            self.signature = hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def is_valid(self) -> bool:
        return datetime.now() < self.expires_at
    
    def has_permission(self, permission: Permission) -> bool:
        return permission.value in self.permissions


# ============================================================================
# 2. POLICY LAYER (PaC)
# ============================================================================

class PolicyEffect(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_WITH_APPROVAL = "allow_with_approval"
    ALLOW_WITH_AUDIT = "allow_with_audit"


@dataclass
class PolicyDecision:
    effect: PolicyEffect
    policy_id: str
    reason: str


class PolicyEngine:
    """OPA-style policy engine"""
    
    POLICIES = {
        "secure-agent": {
            "rules": [
                {"id": "sa-001", "action": "file.read", "path_allowed": True, "effect": "allow"},
                {"id": "sa-002", "action": "file.write", "path_allowed": True, "effect": "allow_with_audit"},
                {"id": "sa-003", "action": "file.delete", "effect": "allow_with_approval"},
                {"id": "sa-004", "action": "execute.command", "effect": "allow_with_approval"},
                {"id": "sa-005", "action": "financial.transfer", "effect": "allow_with_approval"},
                {"id": "sa-006", "action": "network.external", "effect": "deny"},
            ]
        }
    }
    
    def __init__(self):
        self.decision_log = []
    
    def evaluate(self, agent_type: str, action: str, path_allowed: bool = True) -> PolicyDecision:
        rules = self.POLICIES.get(agent_type, {}).get("rules", [])
        
        for rule in rules:
            if rule["action"] == action:
                # Check path condition if present
                if "path_allowed" in rule and not path_allowed:
                    return PolicyDecision(PolicyEffect.DENY, rule["id"], "Path not in scope")
                
                effect = PolicyEffect(rule["effect"])
                decision = PolicyDecision(effect, rule["id"], f"Rule {rule['id']} matched")
                self.decision_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": action,
                    "decision": effect.value,
                    "policy_id": rule["id"]
                })
                return decision
        
        return PolicyDecision(PolicyEffect.DENY, "default", "No matching rule - default deny")


# ============================================================================
# 3. HITL LAYER
# ============================================================================

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ApprovalRequest:
    request_id: str
    action: str
    args: dict
    risk_level: RiskLevel
    agent_id: str
    user_id: str
    status: str = "pending"


class HITLWorkflow:
    """Human-in-the-Loop approval workflow"""
    
    RISK_MAP = {
        "file.read": RiskLevel.LOW,
        "file.write": RiskLevel.MEDIUM,
        "file.delete": RiskLevel.HIGH,
        "execute.command": RiskLevel.HIGH,
        "financial.transfer": RiskLevel.CRITICAL,
    }
    
    def __init__(self):
        self.approval_log = []
    
    def get_risk_level(self, action: str) -> RiskLevel:
        return self.RISK_MAP.get(action, RiskLevel.MEDIUM)
    
    def requires_approval(self, policy_decision: PolicyDecision) -> bool:
        return policy_decision.effect == PolicyEffect.ALLOW_WITH_APPROVAL
    
    def request_approval(self, action: str, args: dict, agent_id: str, user_id: str) -> bool:
        risk = self.get_risk_level(action)
        request = ApprovalRequest(
            request_id=f"approval-{uuid.uuid4().hex[:8]}",
            action=action,
            args=args,
            risk_level=risk,
            agent_id=agent_id,
            user_id=user_id
        )
        
        # Display approval request
        risk_color = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}[risk.value]
        
        console.print(Panel(
            f"[bold yellow]⚠️  APPROVAL REQUIRED[/bold yellow]\n\n"
            f"[bold]Request:[/bold] {request.request_id}\n"
            f"[bold]Action:[/bold] {action}\n"
            f"[bold]Args:[/bold] {json.dumps(args, indent=2)}\n"
            f"[bold]Risk:[/bold] [{risk_color}]{risk.value.upper()}[/{risk_color}]\n"
            f"[bold]Agent:[/bold] {agent_id}\n"
            f"[bold]User:[/bold] {user_id}",
            title="🔒 Human Approval",
            border_style="yellow"
        ))
        
        try:
            approved = Confirm.ask("Do you approve this action?", default=False)
            
            self.approval_log.append({
                "timestamp": datetime.now().isoformat(),
                "request_id": request.request_id,
                "action": action,
                "agent_id": agent_id,
                "user_id": user_id,
                "risk_level": risk.value,
                "decision": "approved" if approved else "denied"
            })
            
            if approved:
                console.print("[green]✅ APPROVED[/green]")
            else:
                console.print("[red]❌ DENIED[/red]")
            
            return approved
        except:
            return False


# ============================================================================
# 4. AUDIT LAYER
# ============================================================================

class AuditLogger:
    """Immutable audit trail for all agent actions"""
    
    def __init__(self, log_dir: str = "./audit_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = uuid.uuid4().hex[:8]
        self.log_file = os.path.join(log_dir, f"audit_{self.session_id}.jsonl")
        self.entries = []
    
    def log(
        self,
        action: str,
        args: dict,
        agent_id: str,
        user_id: str,
        policy_decision: str,
        outcome: str,
        approval_required: bool = False,
        approval_granted: Optional[bool] = None
    ):
        """Log an action with full context"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "action": action,
            "args": args,
            "policy_decision": policy_decision,
            "approval_required": approval_required,
            "approval_granted": approval_granted,
            "outcome": outcome
        }
        
        self.entries.append(entry)
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        
        return entry
    
    def get_session_summary(self) -> dict:
        """Get summary of session actions"""
        return {
            "session_id": self.session_id,
            "total_actions": len(self.entries),
            "allowed": sum(1 for e in self.entries if e["outcome"] == "success"),
            "denied": sum(1 for e in self.entries if e["outcome"] == "denied"),
            "approvals_requested": sum(1 for e in self.entries if e["approval_required"]),
            "approvals_granted": sum(1 for e in self.entries if e["approval_granted"] is True),
        }


# ============================================================================
# 5. SECURE TOOLS
# ============================================================================

class SecureTools:
    """Tools with all 5 security layers"""
    
    def __init__(
        self,
        identity: AgentIdentity,
        policy_engine: PolicyEngine,
        hitl: HITLWorkflow,
        audit: AuditLogger
    ):
        self.identity = identity
        self.policy = policy_engine
        self.hitl = hitl
        self.audit = audit
    
    def _execute_with_security(
        self,
        action: str,
        args: dict,
        executor: Callable,
        permission_required: Optional[Permission] = None
    ) -> str:
        """Execute action through all security layers"""
        
        # Layer 1: Identity validation
        if not self.identity.is_valid():
            self.audit.log(action, args, self.identity.agent_id, self.identity.user_id,
                          "identity_expired", "denied")
            return "DENIED: Agent identity has expired"
        
        # Layer 1b: Permission check
        if permission_required and not self.identity.has_permission(permission_required):
            self.audit.log(action, args, self.identity.agent_id, self.identity.user_id,
                          "permission_denied", "denied")
            return f"DENIED: Agent lacks {permission_required.value} permission"
        
        # Layer 1c: Scope check (for file operations)
        path = args.get("filepath") or args.get("directory")
        path_allowed = True
        if path and hasattr(self.identity.resource_scope, 'is_path_allowed'):
            path_allowed = self.identity.resource_scope.is_path_allowed(path)
            if not path_allowed:
                self.audit.log(action, args, self.identity.agent_id, self.identity.user_id,
                              "scope_denied", "denied")
                return f"DENIED: Path {path} is outside agent's resource scope"
        
        # Layer 2: Policy evaluation
        policy_decision = self.policy.evaluate("secure-agent", action, path_allowed)
        
        if policy_decision.effect == PolicyEffect.DENY:
            self.audit.log(action, args, self.identity.agent_id, self.identity.user_id,
                          "policy_deny", "denied")
            return f"DENIED by policy: {policy_decision.reason}"
        
        # Layer 3: HITL check
        approval_required = self.hitl.requires_approval(policy_decision)
        approval_granted = None
        
        if approval_required:
            approval_granted = self.hitl.request_approval(
                action, args, self.identity.agent_id, self.identity.user_id
            )
            if not approval_granted:
                self.audit.log(action, args, self.identity.agent_id, self.identity.user_id,
                              policy_decision.effect.value, "denied",
                              approval_required=True, approval_granted=False)
                return "DENIED: Human reviewer rejected the action"
        
        # Execute action
        try:
            result = executor()
            
            # Layer 5: Audit logging
            self.audit.log(action, args, self.identity.agent_id, self.identity.user_id,
                          policy_decision.effect.value, "success",
                          approval_required=approval_required, approval_granted=approval_granted)
            
            return result
        except Exception as e:
            self.audit.log(action, args, self.identity.agent_id, self.identity.user_id,
                          policy_decision.effect.value, f"error: {e}",
                          approval_required=approval_required, approval_granted=approval_granted)
            return f"Error: {e}"
    
    def read_file(self, filepath: str) -> str:
        def executor():
            with open(filepath, 'r') as f:
                return f.read()[:5000]
        return self._execute_with_security("file.read", {"filepath": filepath}, executor, Permission.FILE_READ)
    
    def write_file(self, filepath: str, content: str) -> str:
        def executor():
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {filepath}"
        return self._execute_with_security("file.write", {"filepath": filepath}, executor, Permission.FILE_WRITE)
    
    def delete_file(self, filepath: str) -> str:
        def executor():
            os.remove(filepath)
            return f"Successfully deleted {filepath}"
        return self._execute_with_security("file.delete", {"filepath": filepath}, executor, Permission.FILE_DELETE)
    
    def execute_command(self, command: str) -> str:
        import subprocess
        def executor():
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            return result.stdout or result.stderr or "Command executed"
        return self._execute_with_security("execute.command", {"command": command}, executor, Permission.EXECUTE_FULL)
    
    def list_files(self, directory: str) -> str:
        def executor():
            files = os.listdir(directory)
            return "\n".join(files[:50])
        return self._execute_with_security("file.read", {"directory": directory}, executor, Permission.FILE_READ)


# ============================================================================
# SECURE AGENT
# ============================================================================

def parse_tool_calls(response: str) -> list:
    import re
    tool_calls = []
    pattern = r'<tool>(\w+)</tool>\s*<args>(\{[^}]+\})</args>'
    for match in re.findall(pattern, response, re.DOTALL | re.IGNORECASE):
        try:
            tool_calls.append({"tool": match[0], "args": json.loads(match[1])})
        except json.JSONDecodeError:
            pass
    return tool_calls


class SecureAgent:
    """
    Production-ready AI agent with all 5 security principles.
    
    Security Stack:
    1. Identity Layer - Unique agent identity, minimal permissions
    2. Policy Layer - OPA-style policy enforcement
    3. HITL Layer - Human approval for high-risk actions
    4. Autonomy Boundaries - Resource scope restrictions
    5. Audit Layer - Complete action traceability
    """
    
    def __init__(self, user_id: str):
        # Create unique identity
        self.identity = AgentIdentity(
            agent_id=f"agent-secure-{uuid.uuid4().hex[:8]}",
            agent_type="secure-agent",
            user_id=user_id,
            session_id=uuid.uuid4().hex[:8],
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            permissions=[
                Permission.FILE_READ.value,
                Permission.FILE_WRITE.value,
                Permission.FILE_DELETE.value,
                Permission.EXECUTE_FULL.value,
            ],
            resource_scope=ResourceScope(
                file_paths=["./**"],
                exclude_paths=["**/.env", "**/*secret*", "**/*password*", "/etc/**"]
            )
        )
        
        # Initialize security layers
        self.policy = PolicyEngine()
        self.hitl = HITLWorkflow()
        self.audit = AuditLogger()
        self.tools = SecureTools(self.identity, self.policy, self.hitl, self.audit)
        
        # LLM client
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = "openai/gpt-4o-mini"
    
    def get_system_prompt(self) -> str:
        return f"""You are a secure AI assistant with controlled system access.

AGENT IDENTITY: {self.identity.agent_id}
SESSION: {self.identity.session_id}
DELEGATED BY: {self.identity.user_id}
PERMISSIONS: {', '.join(self.identity.permissions)}
EXPIRES: {self.identity.expires_at.isoformat()}

SECURITY CONTROLS ACTIVE:
✓ Identity verification
✓ Policy enforcement  
✓ Human-in-the-loop for high-risk actions
✓ Resource scope restrictions
✓ Complete audit logging

AVAILABLE TOOLS:
1. read_file - <tool>read_file</tool> <args>{{"filepath": "path"}}</args>
2. write_file - <tool>write_file</tool> <args>{{"filepath": "path", "content": "text"}}</args>
3. delete_file - <tool>delete_file</tool> <args>{{"filepath": "path"}}</args>
4. execute_command - <tool>execute_command</tool> <args>{{"command": "cmd"}}</args>
5. list_files - <tool>list_files</tool> <args>{{"directory": "path"}}</args>

All actions are subject to security controls and may require approval.
"""
    
    def execute_tool(self, tool_name: str, args: dict) -> str:
        tool_map = {
            "read_file": lambda: self.tools.read_file(args.get("filepath", "")),
            "write_file": lambda: self.tools.write_file(args.get("filepath", ""), args.get("content", "")),
            "delete_file": lambda: self.tools.delete_file(args.get("filepath", "")),
            "execute_command": lambda: self.tools.execute_command(args.get("command", "")),
            "list_files": lambda: self.tools.list_files(args.get("directory", "")),
        }
        if tool_name in tool_map:
            return tool_map[tool_name]()
        return f"Unknown tool: {tool_name}"
    
    def chat(self, user_message: str) -> str:
        if not self.identity.is_valid():
            return "ERROR: Agent session has expired. Please start a new session."
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_message}
        ]
        
        for _ in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.1
                )
                reply = response.choices[0].message.content or ""
                tool_calls = parse_tool_calls(reply)
                
                if not tool_calls:
                    return reply
                
                results = []
                for tc in tool_calls:
                    console.print(f"\n[cyan]🔐 Security check for: {tc['tool']}[/cyan]")
                    result = self.execute_tool(tc["tool"], tc["args"])
                    results.append(f"[{tc['tool']}]: {result}")
                
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "Results:\n" + "\n".join(results)})
            except Exception as e:
                return f"Error: {e}"
        
        return "Max iterations."


# ============================================================================
# MAIN
# ============================================================================

def show_security_stack():
    """Display the security stack"""
    console.print(Panel(
        "[bold]Security Stack (Bottom to Top):[/bold]\n\n"
        "┌─────────────────────────────────────────┐\n"
        "│  [cyan]5. AUDIT LAYER[/cyan]                        │\n"
        "│     Complete action traceability        │\n"
        "├─────────────────────────────────────────┤\n"
        "│  [yellow]4. HITL LAYER[/yellow]                         │\n"
        "│     Human approval for high-risk        │\n"
        "├─────────────────────────────────────────┤\n"
        "│  [blue]3. POLICY LAYER[/blue]                       │\n"
        "│     OPA-style policy enforcement        │\n"
        "├─────────────────────────────────────────┤\n"
        "│  [magenta]2. AUTONOMY BOUNDARIES[/magenta]                │\n"
        "│     Resource scope restrictions         │\n"
        "├─────────────────────────────────────────┤\n"
        "│  [green]1. IDENTITY LAYER[/green]                     │\n"
        "│     Unique identity, minimal perms      │\n"
        "└─────────────────────────────────────────┘",
        title="🛡️ Secure Agent Architecture",
        border_style="green"
    ))


def main():
    console.print(Panel.fit(
        "[bold green]🛡️  FULLY SECURED AI AGENT[/bold green]\n\n"
        "This agent implements ALL 5 security principles:\n"
        "1. [green]PoLP[/green] - Unique identity, minimal permissions\n"
        "2. [yellow]HITL[/yellow] - Human approval for high-risk actions\n"
        "3. [blue]PaC[/blue] - Policy-as-Code enforcement\n"
        "4. [magenta]Boundaries[/magenta] - Resource scope restrictions\n"
        "5. [cyan]Audit[/cyan] - Complete action traceability\n\n"
        "[dim]Type 'quit' to exit, 'audit' for session log[/dim]",
        title="Lab 12: Secure Agent",
        border_style="green"
    ))
    
    show_security_stack()
    
    user_id = os.getenv("USER_ID", "demo-user@example.com")
    
    try:
        agent = SecureAgent(user_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return
    
    # Show agent identity
    console.print(Panel(
        f"[bold]Agent ID:[/bold] {agent.identity.agent_id}\n"
        f"[bold]Session:[/bold] {agent.identity.session_id}\n"
        f"[bold]User:[/bold] {agent.identity.user_id}\n"
        f"[bold]Signature:[/bold] {agent.identity.signature}\n"
        f"[bold]Expires:[/bold] {agent.identity.expires_at.isoformat()}",
        title="🆔 Agent Identity",
        border_style="cyan"
    ))
    
    # Create test file
    with open("test_secure.txt", "w") as f:
        f.write("Test file for secure agent demo.\n")
    
    console.print("\n[yellow]Try these commands:[/yellow]")
    console.print("• 'Read test_secure.txt' (allowed)")
    console.print("• 'Read /etc/passwd' (DENIED - out of scope)")
    console.print("• 'Read .env' (DENIED - excluded)")
    console.print("• 'Delete test_secure.txt' (requires approval)")
    console.print("• 'Execute: whoami' (requires approval)")
    
    while True:
        try:
            user_input = console.input("\n[green]You:[/green] ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'audit':
                summary = agent.audit.get_session_summary()
                console.print("\n[bold]Session Audit Summary:[/bold]")
                console.print(json.dumps(summary, indent=2))
                console.print(f"\n[dim]Full log: {agent.audit.log_file}[/dim]")
                continue
            
            response = agent.chat(user_input)
            console.print(Panel(Markdown(response), title="Agent", border_style="blue"))
            
        except KeyboardInterrupt:
            break
    
    # Final summary
    summary = agent.audit.get_session_summary()
    console.print("\n[bold]Final Session Summary:[/bold]")
    console.print(f"Total actions: {summary['total_actions']}")
    console.print(f"[green]Allowed: {summary['allowed']}[/green]")
    console.print(f"[red]Denied: {summary['denied']}[/red]")
    console.print(f"Approvals requested: {summary['approvals_requested']}")
    console.print(f"Approvals granted: {summary['approvals_granted']}")
    console.print(f"\n[dim]Audit log saved to: {agent.audit.log_file}[/dim]")


if __name__ == "__main__":
    main()
