#!/usr/bin/env python3
"""
Lab 12: Policy as Code (PaC)
============================

Demonstrates policy-based security enforcement for AI agents:
- Declarative policy definitions (JSON-based, OPA-compatible)
- Attribute-Based Access Control (ABAC)
- Pre-action policy evaluation
- Policy versioning and audit

Key Concepts:
- Policies define what agents CAN do (allow)
- Default deny for undefined actions
- Conditions can include resource attributes
- Policies can require additional controls (approval, audit)

Author: GopeshK
License: MIT License
Disclaimer: For educational and demonstration purposes only.
"""

import os
import json
import uuid
import fnmatch
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax

load_dotenv()
console = Console()


# ============================================================================
# POLICY DEFINITIONS
# ============================================================================

class PolicyEffect(Enum):
    """Policy decision effects"""
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_WITH_APPROVAL = "allow_with_approval"
    ALLOW_WITH_AUDIT = "allow_with_audit"


@dataclass
class PolicyDecision:
    """Result of policy evaluation"""
    effect: PolicyEffect
    policy_id: str
    policy_name: str
    reason: str
    conditions_met: list
    conditions_failed: list


class PolicyEngine:
    """
    OPA-style policy engine for AI agent authorization.
    
    Evaluates actions against declarative policies with:
    - Pattern matching for resources
    - Attribute-based conditions
    - Hierarchical policy resolution
    """
    
    def __init__(self, policy_file: Optional[str] = None):
        self.policies = self._load_default_policies()
        if policy_file and os.path.exists(policy_file):
            self._load_policies(policy_file)
        self.decision_log: list[dict] = []
    
    def _load_default_policies(self) -> dict:
        """Load default security policies"""
        return {
            "policy_version": "1.0",
            "default_effect": "deny",
            "agents": {
                "code-assistant": {
                    "description": "Standard code assistant with file access",
                    "rules": [
                        {
                            "id": "ca-001",
                            "name": "Allow file reading in workspace",
                            "action": "file.read",
                            "conditions": {
                                "path_patterns": ["./**", "./src/**", "./tests/**"],
                                "exclude_patterns": ["**/.env", "**/*secret*", "**/*password*"]
                            },
                            "effect": "allow"
                        },
                        {
                            "id": "ca-002",
                            "name": "Allow file writing in workspace",
                            "action": "file.write",
                            "conditions": {
                                "path_patterns": ["./src/**", "./tests/**", "./*.txt"],
                                "max_file_size_kb": 1024
                            },
                            "effect": "allow_with_audit"
                        },
                        {
                            "id": "ca-003",
                            "name": "Deny file deletion",
                            "action": "file.delete",
                            "conditions": {},
                            "effect": "allow_with_approval"
                        },
                        {
                            "id": "ca-004",
                            "name": "Deny external network access",
                            "action": "network.external",
                            "conditions": {},
                            "effect": "deny"
                        },
                        {
                            "id": "ca-005",
                            "name": "Deny command execution",
                            "action": "execute.command",
                            "conditions": {},
                            "effect": "deny"
                        }
                    ]
                },
                "admin-agent": {
                    "description": "Administrative agent with elevated privileges",
                    "rules": [
                        {
                            "id": "aa-001",
                            "name": "Allow all file operations",
                            "action": "file.*",
                            "conditions": {},
                            "effect": "allow_with_audit"
                        },
                        {
                            "id": "aa-002",
                            "name": "Allow safe commands",
                            "action": "execute.command",
                            "conditions": {
                                "allowed_commands": ["ls", "cat", "echo", "pwd", "whoami"],
                                "denied_commands": ["rm -rf", "dd", "mkfs", "shutdown"]
                            },
                            "effect": "allow_with_approval"
                        }
                    ]
                },
                "readonly-agent": {
                    "description": "Read-only agent with minimal permissions",
                    "rules": [
                        {
                            "id": "ro-001",
                            "name": "Allow file reading only",
                            "action": "file.read",
                            "conditions": {
                                "path_patterns": ["./**"],
                                "exclude_patterns": ["**/.env", "**/*secret*"]
                            },
                            "effect": "allow"
                        }
                    ]
                }
            }
        }
    
    def _load_policies(self, policy_file: str):
        """Load policies from file"""
        try:
            with open(policy_file, 'r') as f:
                loaded = json.load(f)
                self.policies.update(loaded)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load policy file: {e}[/yellow]")
    
    def evaluate(
        self,
        agent_type: str,
        action: str,
        resource: dict
    ) -> PolicyDecision:
        """
        Evaluate an action against policies.
        
        Args:
            agent_type: Type of agent (e.g., "code-assistant")
            action: Action to perform (e.g., "file.read")
            resource: Resource attributes (e.g., {"path": "/etc/passwd"})
        
        Returns:
            PolicyDecision with the evaluation result
        """
        # Get agent policies
        agent_policies = self.policies.get("agents", {}).get(agent_type)
        
        if not agent_policies:
            decision = PolicyDecision(
                effect=PolicyEffect.DENY,
                policy_id="default",
                policy_name="No Policy Defined",
                reason=f"No policies defined for agent type: {agent_type}",
                conditions_met=[],
                conditions_failed=["agent_type_exists"]
            )
            self._log_decision(agent_type, action, resource, decision)
            return decision
        
        # Find matching rule
        for rule in agent_policies.get("rules", []):
            rule_action = rule.get("action", "")
            
            # Check action match (supports wildcards)
            if not self._action_matches(action, rule_action):
                continue
            
            # Evaluate conditions
            conditions = rule.get("conditions", {})
            met, failed = self._evaluate_conditions(conditions, action, resource)
            
            if not failed:
                # All conditions met
                effect = PolicyEffect(rule.get("effect", "deny"))
                decision = PolicyDecision(
                    effect=effect,
                    policy_id=rule.get("id", "unknown"),
                    policy_name=rule.get("name", "Unknown Rule"),
                    reason=f"Rule {rule.get('id')} matched with all conditions met",
                    conditions_met=met,
                    conditions_failed=[]
                )
                self._log_decision(agent_type, action, resource, decision)
                return decision
        
        # No matching rule - default deny
        decision = PolicyDecision(
            effect=PolicyEffect.DENY,
            policy_id="default",
            policy_name="Default Deny",
            reason=f"No matching policy rule for action: {action}",
            conditions_met=[],
            conditions_failed=["no_matching_rule"]
        )
        self._log_decision(agent_type, action, resource, decision)
        return decision
    
    def _action_matches(self, action: str, pattern: str) -> bool:
        """Check if action matches pattern (supports wildcards)"""
        if pattern == action:
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return action.startswith(prefix)
        return fnmatch.fnmatch(action, pattern)
    
    def _evaluate_conditions(
        self,
        conditions: dict,
        action: str,
        resource: dict
    ) -> tuple[list, list]:
        """Evaluate conditions and return (met, failed) lists"""
        met = []
        failed = []
        
        # Check path patterns
        if "path_patterns" in conditions:
            path = resource.get("path", "")
            patterns = conditions["path_patterns"]
            if any(fnmatch.fnmatch(path, p) for p in patterns):
                met.append(f"path_pattern:{path}")
            else:
                failed.append(f"path_pattern:{path} not in {patterns}")
        
        # Check exclude patterns
        if "exclude_patterns" in conditions:
            path = resource.get("path", "")
            patterns = conditions["exclude_patterns"]
            if any(fnmatch.fnmatch(path, p) for p in patterns):
                failed.append(f"exclude_pattern:{path} matches {patterns}")
            else:
                met.append(f"exclude_check:{path}")
        
        # Check file size
        if "max_file_size_kb" in conditions:
            size_kb = resource.get("size_kb", 0)
            max_size = conditions["max_file_size_kb"]
            if size_kb <= max_size:
                met.append(f"file_size:{size_kb}KB <= {max_size}KB")
            else:
                failed.append(f"file_size:{size_kb}KB > {max_size}KB")
        
        # Check allowed commands
        if "allowed_commands" in conditions:
            command = resource.get("command", "")
            allowed = conditions["allowed_commands"]
            if any(command.startswith(cmd) for cmd in allowed):
                met.append(f"command:{command} in allowed list")
            else:
                failed.append(f"command:{command} not in allowed list")
        
        # Check denied commands
        if "denied_commands" in conditions:
            command = resource.get("command", "")
            denied = conditions["denied_commands"]
            if any(cmd in command for cmd in denied):
                failed.append(f"command:{command} in denied list")
            else:
                met.append(f"command:{command} not in denied list")
        
        return met, failed
    
    def _log_decision(
        self,
        agent_type: str,
        action: str,
        resource: dict,
        decision: PolicyDecision
    ):
        """Log policy decision"""
        self.decision_log.append({
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "action": action,
            "resource": resource,
            "effect": decision.effect.value,
            "policy_id": decision.policy_id,
            "policy_name": decision.policy_name,
            "reason": decision.reason
        })
    
    def show_policies(self, agent_type: str):
        """Display policies for an agent type"""
        agent_policies = self.policies.get("agents", {}).get(agent_type)
        
        if not agent_policies:
            console.print(f"[red]No policies defined for: {agent_type}[/red]")
            return
        
        table = Table(title=f"Policies for {agent_type}", border_style="blue")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Action", style="yellow")
        table.add_column("Effect", style="white")
        
        for rule in agent_policies.get("rules", []):
            effect = rule.get("effect", "deny")
            effect_color = {
                "allow": "green",
                "deny": "red",
                "allow_with_approval": "yellow",
                "allow_with_audit": "cyan"
            }.get(effect, "white")
            
            table.add_row(
                rule.get("id", ""),
                rule.get("name", ""),
                rule.get("action", ""),
                f"[{effect_color}]{effect}[/{effect_color}]"
            )
        
        console.print(table)


# ============================================================================
# POLICY-ENFORCED TOOLS
# ============================================================================

class PolicyEnforcedTools:
    """Tools with policy engine enforcement"""
    
    def __init__(self, agent_type: str, policy_engine: PolicyEngine):
        self.agent_type = agent_type
        self.policy_engine = policy_engine
    
    def _enforce_policy(self, action: str, resource: dict) -> tuple[bool, str]:
        """Check policy and return (allowed, message)"""
        decision = self.policy_engine.evaluate(self.agent_type, action, resource)
        
        if decision.effect == PolicyEffect.ALLOW:
            return True, f"[green]POLICY ALLOWED[/green]: {decision.policy_name}"
        elif decision.effect == PolicyEffect.ALLOW_WITH_AUDIT:
            console.print(f"[cyan]AUDIT[/cyan]: Action logged for compliance")
            return True, f"[cyan]POLICY ALLOWED (AUDITED)[/cyan]: {decision.policy_name}"
        elif decision.effect == PolicyEffect.ALLOW_WITH_APPROVAL:
            console.print(f"[yellow]NOTE[/yellow]: This action would require human approval in production")
            return True, f"[yellow]POLICY REQUIRES APPROVAL[/yellow]: {decision.policy_name}"
        else:
            return False, f"[red]POLICY DENIED[/red]: {decision.reason}"
    
    def read_file(self, filepath: str) -> str:
        """Read file with policy check"""
        allowed, msg = self._enforce_policy("file.read", {"path": filepath})
        console.print(msg)
        
        if not allowed:
            return f"DENIED by policy: Cannot read {filepath}"
        
        try:
            with open(filepath, 'r') as f:
                return f.read()[:5000]
        except Exception as e:
            return f"Error: {e}"
    
    def write_file(self, filepath: str, content: str) -> str:
        """Write file with policy check"""
        size_kb = len(content) / 1024
        allowed, msg = self._enforce_policy("file.write", {"path": filepath, "size_kb": size_kb})
        console.print(msg)
        
        if not allowed:
            return f"DENIED by policy: Cannot write to {filepath}"
        
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {filepath}"
        except Exception as e:
            return f"Error: {e}"
    
    def delete_file(self, filepath: str) -> str:
        """Delete file with policy check"""
        allowed, msg = self._enforce_policy("file.delete", {"path": filepath})
        console.print(msg)
        
        if not allowed:
            return f"DENIED by policy: Cannot delete {filepath}"
        
        # Note: Even if policy allows, we just demonstrate here
        return f"DELETE operation would execute for {filepath} (demo mode)"
    
    def execute_command(self, command: str) -> str:
        """Execute command with policy check"""
        allowed, msg = self._enforce_policy("execute.command", {"command": command})
        console.print(msg)
        
        if not allowed:
            return f"DENIED by policy: Cannot execute command"
        
        return f"COMMAND would execute: {command} (demo mode, requires approval)"
    
    def list_files(self, directory: str) -> str:
        """List files with policy check"""
        allowed, msg = self._enforce_policy("file.read", {"path": directory})
        console.print(msg)
        
        if not allowed:
            return f"DENIED by policy: Cannot list {directory}"
        
        try:
            files = os.listdir(directory)
            return "\n".join(files[:50])
        except Exception as e:
            return f"Error: {e}"


# ============================================================================
# POLICY-ENFORCED AGENT
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


class PolicyEnforcedAgent:
    """
    AI Agent with Policy-as-Code enforcement.
    
    Security Features:
    - Declarative policy definitions
    - Pre-action policy evaluation
    - ABAC (Attribute-Based Access Control)
    - Policy decision audit logging
    """
    
    def __init__(self, agent_type: str, policy_engine: PolicyEngine):
        self.agent_type = agent_type
        self.policy_engine = policy_engine
        self.tools = PolicyEnforcedTools(agent_type, policy_engine)
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = "openai/gpt-4o-mini"
    
    def get_system_prompt(self) -> str:
        return f"""You are an AI assistant operating under policy constraints.

AGENT TYPE: {self.agent_type}
POLICY ENGINE: Active

AVAILABLE TOOLS:
1. read_file - Read a file (subject to policy)
   <tool>read_file</tool> <args>{{"filepath": "path"}}</args>

2. write_file - Write to file (subject to policy)
   <tool>write_file</tool> <args>{{"filepath": "path", "content": "text"}}</args>

3. delete_file - Delete a file (subject to policy)
   <tool>delete_file</tool> <args>{{"filepath": "path"}}</args>

4. execute_command - Run shell command (subject to policy)
   <tool>execute_command</tool> <args>{{"command": "cmd"}}</args>

5. list_files - List directory (subject to policy)
   <tool>list_files</tool> <args>{{"directory": "path"}}</args>

Operations will be checked against the security policy before execution.
Some operations may be denied based on your agent type's permissions.
"""
    
    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute tool with policy enforcement"""
        tool_map = {
            "read_file": lambda: self.tools.read_file(args.get("filepath", "")),
            "write_file": lambda: self.tools.write_file(
                args.get("filepath", ""),
                args.get("content", "")
            ),
            "delete_file": lambda: self.tools.delete_file(args.get("filepath", "")),
            "execute_command": lambda: self.tools.execute_command(args.get("command", "")),
            "list_files": lambda: self.tools.list_files(args.get("directory", "")),
        }
        
        if tool_name in tool_map:
            return tool_map[tool_name]()
        return f"Unknown tool: {tool_name}"
    
    def chat(self, user_message: str) -> str:
        """Process user message with policy enforcement"""
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
                    console.print(f"\n[cyan]Evaluating policy for: {tc['tool']}({tc['args']})[/cyan]")
                    result = self.execute_tool(tc["tool"], tc["args"])
                    results.append(f"[{tc['tool']}]: {result}")
                
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "Results:\n" + "\n".join(results)})
                
            except Exception as e:
                return f"Error: {e}"
        
        return "Max iterations reached."


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    console.print(Panel.fit(
        "[bold blue]📜 POLICY AS CODE DEMO[/bold blue]\n\n"
        "This demo shows policy-based security enforcement:\n"
        "• Declarative JSON policies define allowed actions\n"
        "• ABAC (Attribute-Based Access Control)\n"
        "• Pre-action policy evaluation\n"
        "• Policy decision audit logging\n\n"
        "[dim]Type 'quit' to exit[/dim]",
        title="Lab 12: Policy Engine",
        border_style="blue"
    ))
    
    # Initialize policy engine
    policy_engine = PolicyEngine()
    
    # Choose agent type
    console.print("\n[bold]Choose agent type (different permissions):[/bold]")
    console.print("1. code-assistant (file read/write in workspace)")
    console.print("2. readonly-agent (read-only access)")
    console.print("3. admin-agent (elevated privileges)")
    
    choice = console.input("\nChoice (1, 2, or 3): ").strip()
    
    agent_type = {
        "1": "code-assistant",
        "2": "readonly-agent",
        "3": "admin-agent"
    }.get(choice, "code-assistant")
    
    console.print(f"\n[bold]Selected: {agent_type}[/bold]\n")
    
    # Show policies for selected agent
    policy_engine.show_policies(agent_type)
    
    # Create test files
    with open("test_policy.txt", "w") as f:
        f.write("This is a test file for policy demonstration.\n")
    with open("./src/sample.py", "w") as f:
        os.makedirs("./src", exist_ok=True)
        f.write("# Sample Python file\nprint('hello')\n")
    console.print("\n[dim]Created test files: test_policy.txt, ./src/sample.py[/dim]\n")
    
    try:
        agent = PolicyEnforcedAgent(agent_type, policy_engine)
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        return
    
    console.print("[yellow]Try these commands:[/yellow]")
    console.print("• 'Read test_policy.txt' (should be allowed)")
    console.print("• 'Read .env' (should be DENIED - excluded pattern)")
    console.print("• 'Write hello to ./src/test.py' (depends on agent type)")
    console.print("• 'Delete test_policy.txt' (requires approval)")
    console.print("• 'Execute: ls -la' (depends on agent type)")
    console.print("\n• Type 'policies' to see current policies")
    console.print("• Type 'audit' to see policy decisions")
    
    while True:
        try:
            user_input = console.input("\n[green]You:[/green] ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'policies':
                policy_engine.show_policies(agent_type)
                continue
            if user_input.lower() == 'audit':
                console.print("\n[bold]Policy Decision Audit Log:[/bold]")
                for entry in policy_engine.decision_log[-10:]:
                    effect_color = "green" if entry["effect"] == "allow" else "red"
                    console.print(
                        f"[{effect_color}]{entry['effect'].upper()}[/{effect_color}] "
                        f"{entry['action']} on {entry['resource']} - {entry['reason']}"
                    )
                continue
            
            response = agent.chat(user_input)
            console.print(Panel(Markdown(response), title="Agent", border_style="blue"))
            
        except KeyboardInterrupt:
            break
    
    # Final audit summary
    console.print("\n[bold]Session Policy Summary:[/bold]")
    allowed = sum(1 for d in policy_engine.decision_log if d["effect"] == "allow")
    denied = sum(1 for d in policy_engine.decision_log if d["effect"] == "deny")
    console.print(f"Total decisions: {len(policy_engine.decision_log)}")
    console.print(f"[green]Allowed: {allowed}[/green]")
    console.print(f"[red]Denied: {denied}[/red]")


if __name__ == "__main__":
    main()
