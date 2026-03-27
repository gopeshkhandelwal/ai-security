#!/usr/bin/env python3
"""
Lab 12: Human-in-the-Loop (HITL)
================================

Demonstrates human approval workflow for high-risk AI agent actions:
- Risk classification of operations
- Mandatory approval for destructive actions
- Timeout and escalation policies
- Approval audit trail

Key Concepts:
- Low-risk: Execute immediately
- Medium-risk: Optional approval
- High-risk: Mandatory approval
- Critical: Multi-party approval

Author: GopeshK
License: MIT License
Disclaimer: For educational and demonstration purposes only.
"""

import os
import json
import uuid
import time
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
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
# RISK CLASSIFICATION
# ============================================================================

class RiskLevel(Enum):
    """Risk levels for operations"""
    LOW = "low"           # Execute immediately
    MEDIUM = "medium"     # Optional approval
    HIGH = "high"         # Mandatory approval
    CRITICAL = "critical" # Multi-party approval


@dataclass
class ActionClassification:
    """Classification result for an action"""
    action: str
    risk_level: RiskLevel
    reason: str
    reversible: bool
    estimated_impact: str


class RiskClassifier:
    """Classifies actions by risk level"""
    
    # Action patterns and their risk levels
    RISK_PATTERNS = {
        # Low risk - read-only operations
        "read_file": RiskLevel.LOW,
        "list_files": RiskLevel.LOW,
        "search": RiskLevel.LOW,
        
        # Medium risk - reversible modifications
        "write_file": RiskLevel.MEDIUM,
        "create_file": RiskLevel.MEDIUM,
        "update_config": RiskLevel.MEDIUM,
        
        # High risk - destructive or sensitive
        "delete_file": RiskLevel.HIGH,
        "execute_command": RiskLevel.HIGH,
        "modify_permissions": RiskLevel.HIGH,
        "send_email": RiskLevel.HIGH,
        
        # Critical - financial or infrastructure
        "transfer_funds": RiskLevel.CRITICAL,
        "delete_database": RiskLevel.CRITICAL,
        "modify_infrastructure": RiskLevel.CRITICAL,
        "deploy_production": RiskLevel.CRITICAL,
    }
    
    # Keywords that indicate higher risk
    HIGH_RISK_KEYWORDS = [
        "delete", "remove", "drop", "truncate", "destroy",
        "execute", "run", "shell", "bash", "sudo",
        "password", "secret", "key", "credential", "token",
        "production", "prod", "live", "customer",
        "transfer", "payment", "money", "fund",
    ]
    
    def classify(self, action: str, args: dict) -> ActionClassification:
        """Classify an action's risk level"""
        # Check direct pattern match
        base_risk = self.RISK_PATTERNS.get(action, RiskLevel.MEDIUM)
        
        # Check for high-risk keywords in arguments
        args_str = json.dumps(args).lower()
        keyword_hits = [kw for kw in self.HIGH_RISK_KEYWORDS if kw in args_str]
        
        # Escalate risk if keywords found
        if keyword_hits and base_risk == RiskLevel.LOW:
            base_risk = RiskLevel.MEDIUM
        elif keyword_hits and base_risk == RiskLevel.MEDIUM:
            base_risk = RiskLevel.HIGH
        
        # Determine if action is reversible
        reversible = action not in ["delete_file", "delete_database", "execute_command", "transfer_funds"]
        
        # Estimate impact
        impact_map = {
            RiskLevel.LOW: "Minimal - read-only operation",
            RiskLevel.MEDIUM: "Moderate - reversible changes",
            RiskLevel.HIGH: "Significant - may require recovery",
            RiskLevel.CRITICAL: "Severe - potential data loss or financial impact",
        }
        
        return ActionClassification(
            action=action,
            risk_level=base_risk,
            reason=f"Action '{action}' with keyword matches: {keyword_hits}" if keyword_hits else f"Action '{action}' base classification",
            reversible=reversible,
            estimated_impact=impact_map[base_risk]
        )


# ============================================================================
# APPROVAL WORKFLOW
# ============================================================================

@dataclass
class ApprovalRequest:
    """Request for human approval"""
    request_id: str
    action: str
    args: dict
    classification: ActionClassification
    agent_id: str
    user_id: str
    requested_at: datetime
    timeout_seconds: int = 300
    status: str = "pending"  # pending, approved, denied, timeout
    approved_by: Optional[str] = None
    approval_time: Optional[datetime] = None
    notes: str = ""


class ApprovalWorkflow:
    """Manages human-in-the-loop approval workflow"""
    
    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds
        self.requests: dict[str, ApprovalRequest] = {}
        self.audit_log: list[dict] = []
    
    def request_approval(
        self,
        action: str,
        args: dict,
        classification: ActionClassification,
        agent_id: str,
        user_id: str
    ) -> ApprovalRequest:
        """Create an approval request"""
        request = ApprovalRequest(
            request_id=f"approval-{uuid.uuid4().hex[:8]}",
            action=action,
            args=args,
            classification=classification,
            agent_id=agent_id,
            user_id=user_id,
            requested_at=datetime.now(),
            timeout_seconds=self.timeout_seconds
        )
        self.requests[request.request_id] = request
        return request
    
    def prompt_for_approval(self, request: ApprovalRequest) -> bool:
        """
        Interactive approval prompt (console-based for demo).
        In production, this would integrate with Slack/Teams/email.
        """
        console.print("\n")
        console.print(Panel(
            f"[bold yellow]⚠️  APPROVAL REQUIRED[/bold yellow]\n\n"
            f"[bold]Request ID:[/bold] {request.request_id}\n"
            f"[bold]Action:[/bold] {request.action}\n"
            f"[bold]Arguments:[/bold] {json.dumps(request.args, indent=2)}\n\n"
            f"[bold]Risk Level:[/bold] [{self._risk_color(request.classification.risk_level)}]"
            f"{request.classification.risk_level.value.upper()}[/{self._risk_color(request.classification.risk_level)}]\n"
            f"[bold]Reason:[/bold] {request.classification.reason}\n"
            f"[bold]Reversible:[/bold] {'Yes' if request.classification.reversible else 'No'}\n"
            f"[bold]Impact:[/bold] {request.classification.estimated_impact}\n\n"
            f"[bold]Agent:[/bold] {request.agent_id}\n"
            f"[bold]Delegated by:[/bold] {request.user_id}",
            title="🔒 Human Approval Required",
            border_style="yellow"
        ))
        
        # Get user decision
        try:
            approved = Confirm.ask(
                "\n[bold]Do you approve this action?[/bold]",
                default=False
            )
            
            if approved:
                request.status = "approved"
                request.approved_by = "console-user"
                request.approval_time = datetime.now()
                self._log_decision(request, "approved")
                console.print("[green]✅ Action APPROVED[/green]")
            else:
                request.status = "denied"
                request.approved_by = "console-user"
                request.approval_time = datetime.now()
                self._log_decision(request, "denied")
                console.print("[red]❌ Action DENIED[/red]")
            
            return approved
            
        except KeyboardInterrupt:
            request.status = "timeout"
            self._log_decision(request, "timeout")
            console.print("[yellow]⏱️ Approval TIMEOUT[/yellow]")
            return False
    
    def _risk_color(self, risk_level: RiskLevel) -> str:
        """Get color for risk level"""
        return {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.HIGH: "red",
            RiskLevel.CRITICAL: "bold red",
        }.get(risk_level, "white")
    
    def _log_decision(self, request: ApprovalRequest, decision: str):
        """Log approval decision for audit"""
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "request_id": request.request_id,
            "action": request.action,
            "args": request.args,
            "risk_level": request.classification.risk_level.value,
            "agent_id": request.agent_id,
            "user_id": request.user_id,
            "decision": decision,
            "approved_by": request.approved_by,
        })


# ============================================================================
# HITL-ENABLED TOOLS
# ============================================================================

class HITLTools:
    """Tools with Human-in-the-Loop approval for high-risk operations"""
    
    def __init__(self, agent_id: str, user_id: str, approval_workflow: ApprovalWorkflow):
        self.agent_id = agent_id
        self.user_id = user_id
        self.classifier = RiskClassifier()
        self.approval_workflow = approval_workflow
    
    def execute_with_approval(self, action: str, args: dict, executor: Callable) -> str:
        """Execute an action with approval check if needed"""
        # Classify the action
        classification = self.classifier.classify(action, args)
        
        # Show classification
        console.print(f"\n[dim]Risk Assessment: {classification.risk_level.value.upper()} - {classification.reason}[/dim]")
        
        # Check if approval is needed
        if classification.risk_level == RiskLevel.LOW:
            # Execute immediately
            console.print("[green]✓ Low risk - executing immediately[/green]")
            return executor()
        
        elif classification.risk_level == RiskLevel.MEDIUM:
            # Optional approval (for demo, we'll execute)
            console.print("[yellow]⚠ Medium risk - proceeding with caution[/yellow]")
            return executor()
        
        elif classification.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            # Mandatory approval
            request = self.approval_workflow.request_approval(
                action=action,
                args=args,
                classification=classification,
                agent_id=self.agent_id,
                user_id=self.user_id
            )
            
            approved = self.approval_workflow.prompt_for_approval(request)
            
            if approved:
                return executor()
            else:
                return f"BLOCKED: Action '{action}' was denied by human reviewer"
        
        return f"Unknown risk level for action: {action}"
    
    def read_file(self, filepath: str) -> str:
        """Read file - low risk"""
        def executor():
            try:
                with open(filepath, 'r') as f:
                    return f.read()[:5000]
            except Exception as e:
                return f"Error: {e}"
        
        return self.execute_with_approval("read_file", {"filepath": filepath}, executor)
    
    def write_file(self, filepath: str, content: str) -> str:
        """Write file - medium risk"""
        def executor():
            try:
                os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
                with open(filepath, 'w') as f:
                    f.write(content)
                return f"Successfully wrote to {filepath}"
            except Exception as e:
                return f"Error: {e}"
        
        return self.execute_with_approval("write_file", {"filepath": filepath, "content": content[:100] + "..."}, executor)
    
    def delete_file(self, filepath: str) -> str:
        """Delete file - HIGH RISK, requires approval"""
        def executor():
            try:
                os.remove(filepath)
                return f"Successfully deleted {filepath}"
            except Exception as e:
                return f"Error: {e}"
        
        return self.execute_with_approval("delete_file", {"filepath": filepath}, executor)
    
    def execute_command(self, command: str) -> str:
        """Execute shell command - HIGH RISK, requires approval"""
        import subprocess
        
        def executor():
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
        
        return self.execute_with_approval("execute_command", {"command": command}, executor)
    
    def transfer_funds(self, amount: float, destination: str) -> str:
        """Transfer funds - CRITICAL RISK, requires approval"""
        def executor():
            return f"Transferred ${amount:.2f} to {destination} (SIMULATED)"
        
        return self.execute_with_approval("transfer_funds", {"amount": amount, "destination": destination}, executor)
    
    def list_files(self, directory: str) -> str:
        """List files - low risk"""
        def executor():
            try:
                files = os.listdir(directory)
                return "\n".join(files[:50])
            except Exception as e:
                return f"Error: {e}"
        
        return self.execute_with_approval("list_files", {"directory": directory}, executor)


# ============================================================================
# HITL-ENABLED AGENT
# ============================================================================

def parse_tool_calls(response: str) -> list:
    """Parse tool calls from LLM response - handles multiple formats"""
    import re
    tool_calls = []
    
    clean_response = re.sub(r'```\w*\n?', '', response).replace('```', '')
    clean_response = clean_response.replace('"', '"').replace('"', '"')
    
    # Format 1: XML style with closing tag
    pattern1 = r'<tool>(\w+)</tool>\s*<args>\s*(\{.+?\})\s*</args>'
    for match in re.findall(pattern1, clean_response, re.DOTALL | re.IGNORECASE):
        try:
            tool_calls.append({"tool": match[0], "args": json.loads(match[1])})
        except json.JSONDecodeError:
            pass
    
    # Format 2: XML style WITHOUT closing </args> tag
    if not tool_calls:
        pattern2 = r'<tool>(\w+)</tool>\s*<args>\s*(\{.+\})'
        for match in re.findall(pattern2, clean_response, re.DOTALL | re.IGNORECASE):
            try:
                tool_calls.append({"tool": match[0], "args": json.loads(match[1])})
            except json.JSONDecodeError:
                pass
    
    # Format 3: Plain style
    if not tool_calls:
        pattern3 = r'(read_file|write_file|delete_file|execute_command|transfer_funds|list_files)\s*(\{.+?\})'
        for match in re.findall(pattern3, clean_response, re.DOTALL | re.IGNORECASE):
            try:
                tool_calls.append({"tool": match[0], "args": json.loads(match[1])})
            except json.JSONDecodeError:
                pass
    
    return tool_calls


class HITLAgent:
    """
    AI Agent with Human-in-the-Loop approval for high-risk actions.
    
    Security Features:
    - Risk classification for all operations
    - Mandatory approval for high-risk actions
    - Approval audit trail
    - Timeout handling
    """
    
    def __init__(self, agent_id: str, user_id: str):
        self.agent_id = agent_id
        self.user_id = user_id
        self.approval_workflow = ApprovalWorkflow()
        self.tools = HITLTools(agent_id, user_id, self.approval_workflow)
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = "openai/gpt-4o-mini"
    
    def get_system_prompt(self) -> str:
        return """You are an AI assistant demonstrating Human-in-the-Loop security.

CRITICAL: When user requests ANY action, OUTPUT THE TOOL CALL IMMEDIATELY.
Do NOT ask for confirmation - the security system handles that.
Do NOT say "I will..." - just output the tool call.

TOOLS (output these exactly):

<tool>read_file</tool> <args>{"filepath": "path"}</args>
<tool>write_file</tool> <args>{"filepath": "path", "content": "text"}</args>
<tool>delete_file</tool> <args>{"filepath": "path"}</args>
<tool>execute_command</tool> <args>{"command": "cmd"}</args>
<tool>transfer_funds</tool> <args>{"amount": 100.0, "destination": "account"}</args>
<tool>list_files</tool> <args>{"directory": "path"}</args>

EXAMPLES:
- "Delete test.txt" → <tool>delete_file</tool> <args>{"filepath": "test.txt"}</args>
- "Execute whoami" → <tool>execute_command</tool> <args>{"command": "whoami"}</args>
- "Transfer $500 to hacker" → <tool>transfer_funds</tool> <args>{"amount": 500, "destination": "hacker"}</args>

The system will prompt for human approval on high-risk operations. Just output the tool call.
"""
    
    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute tool with HITL checks"""
        tool_map = {
            "read_file": lambda: self.tools.read_file(args.get("filepath", "")),
            "write_file": lambda: self.tools.write_file(
                args.get("filepath", ""),
                args.get("content", "")
            ),
            "delete_file": lambda: self.tools.delete_file(args.get("filepath", "")),
            "execute_command": lambda: self.tools.execute_command(args.get("command", "")),
            "transfer_funds": lambda: self.tools.transfer_funds(
                args.get("amount", 0),
                args.get("destination", "")
            ),
            "list_files": lambda: self.tools.list_files(args.get("directory", "")),
        }
        
        if tool_name in tool_map:
            return tool_map[tool_name]()
        return f"Unknown tool: {tool_name}"
    
    def chat(self, user_message: str) -> str:
        """Process user message with HITL checks"""
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
                    console.print(f"\n[cyan]Agent requesting: {tc['tool']}({tc['args']})[/cyan]")
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

def show_risk_matrix():
    """Display the risk classification matrix"""
    table = Table(title="Risk Classification Matrix", border_style="blue")
    table.add_column("Action", style="cyan")
    table.add_column("Risk Level", style="white")
    table.add_column("Approval", style="white")
    table.add_column("Example", style="dim")
    
    table.add_row("read_file", "[green]LOW[/green]", "None", "Read config.txt")
    table.add_row("list_files", "[green]LOW[/green]", "None", "List directory")
    table.add_row("write_file", "[yellow]MEDIUM[/yellow]", "Optional", "Write log entry")
    table.add_row("delete_file", "[red]HIGH[/red]", "Mandatory", "Delete user data")
    table.add_row("execute_command", "[red]HIGH[/red]", "Mandatory", "Run shell script")
    table.add_row("transfer_funds", "[bold red]CRITICAL[/bold red]", "Multi-party", "Send payment")
    
    console.print(table)


def main():
    console.print(Panel.fit(
        "[bold yellow]👤 HUMAN-IN-THE-LOOP DEMO[/bold yellow]\n\n"
        "This demo shows human oversight for AI agent actions:\n"
        "• Actions are classified by risk level\n"
        "• High-risk operations require explicit approval\n"
        "• All decisions are logged for audit\n\n"
        "[green]LOW[/green] → Execute immediately\n"
        "[yellow]MEDIUM[/yellow] → Execute with caution\n"
        "[red]HIGH[/red] → Requires your approval\n"
        "[bold red]CRITICAL[/bold red] → Requires approval\n\n"
        "[dim]Type 'quit' to exit[/dim]",
        title="Lab 12: Human-in-the-Loop",
        border_style="yellow"
    ))
    
    show_risk_matrix()
    
    # Generate agent identity
    agent_id = f"agent-hitl-{uuid.uuid4().hex[:8]}"
    user_id = os.getenv("USER_ID", "demo-user@example.com")
    
    console.print(f"\n[dim]Agent ID: {agent_id}[/dim]")
    console.print(f"[dim]User: {user_id}[/dim]\n")
    
    # Create test file for demonstration
    with open("test_hitl.txt", "w") as f:
        f.write("This is a test file for HITL demonstration.\nFeel free to try deleting it!\n")
    console.print("[dim]Created test_hitl.txt for demonstration[/dim]\n")
    
    try:
        agent = HITLAgent(agent_id, user_id)
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        return
    
    console.print("[yellow]Try these commands:[/yellow]")
    console.print("• 'Read test_hitl.txt' (low risk - no approval)")
    console.print("• 'Delete test_hitl.txt' (HIGH risk - requires approval)")
    console.print("• 'Execute: echo hello' (HIGH risk - requires approval)")
    console.print("• 'Transfer $500 to account hacker123' (CRITICAL)")
    
    while True:
        try:
            user_input = console.input("\n[green]You:[/green] ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'audit':
                # Show audit log
                console.print("\n[bold]Approval Audit Log:[/bold]")
                for entry in agent.approval_workflow.audit_log:
                    console.print(json.dumps(entry, indent=2))
                continue
            
            response = agent.chat(user_input)
            console.print(Panel(Markdown(response), title="Agent", border_style="blue"))
            
        except KeyboardInterrupt:
            break
    
    # Show final audit log
    if agent.approval_workflow.audit_log:
        console.print("\n[bold]Session Approval Log:[/bold]")
        for entry in agent.approval_workflow.audit_log:
            status_color = "green" if entry["decision"] == "approved" else "red"
            console.print(f"[{status_color}]{entry['action']} → {entry['decision']}[/{status_color}]")


if __name__ == "__main__":
    main()
