#!/usr/bin/env python3
"""
Lab 12: Fully Secured AI Agent (Production Pattern)
====================================================

Demonstrates a PRODUCTION-READY AI agent security architecture with proper
separation of concerns between static agent registration and runtime sessions.

CORE ARCHITECTURAL PRINCIPLES:
------------------------------
This implementation follows enterprise patterns similar to:
- SPIFFE/SPIRE for workload identity
- AWS IAM roles with pre-defined policies  
- OAuth2 client credentials for service accounts
- Kubernetes ServiceAccounts with RBAC

KEY CONCEPTS:
-------------
1. RegisteredAgent (STATIC) - Pre-approved agent like a service account
   - Created BEFORE deployment by security team
   - Has immutable agent_id, allowed_tools, resource_scope
   - Similar to IAM Role or SPIFFE ID

2. AgentSession (DYNAMIC) - Runtime invocation context
   - Created when a user/system triggers the agent
   - Tracks: session_id, invoked_by, start/expiry time
   - Enables attribution: "Who triggered this agent?"

3. AgentRegistry - Central registry of approved agents
   - Production: backed by database or secret manager
   - Simulates: HashiCorp Vault, AWS Secrets Manager, K8s ConfigMaps

SECURITY PRINCIPLES IMPLEMENTED:
--------------------------------
1. Principle of Least Privilege (PoLP) - Pre-approved minimal permissions
2. Human-in-the-Loop (HITL) - Approval for high-risk actions
3. Policy as Code (PaC) - OPA-style policy enforcement
4. Autonomy Boundaries - Explicit action constraints
5. Auditability - Complete action traceability

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
# 1. IDENTITY LAYER (PoLP) - PRODUCTION PATTERN
# ============================================================================
# 
# This layer implements the "Service Account" pattern for AI agents:
# - Agents are PRE-REGISTERED before deployment (like IAM roles)
# - Each agent has IMMUTABLE allowed_tools defined at registration
# - Runtime sessions track WHO invoked the agent
#
# Production implementations would use:
# - HashiCorp Vault for agent credential storage
# - SPIFFE/SPIRE for workload identity
# - AWS IAM / GCP Service Accounts for cloud deployments
# ============================================================================

class Permission(Enum):
    """Fine-grained permissions - defined at agent registration time"""
    FILE_READ = "file.read"
    FILE_WRITE = "file.write"
    FILE_DELETE = "file.delete"
    EXECUTE_READONLY = "execute.readonly"
    EXECUTE_FULL = "execute.full"
    NETWORK_INTERNAL = "network.internal"
    NETWORK_EXTERNAL = "network.external"


@dataclass
class ResourceScope:
    """
    Defines allowed resource boundaries for an agent.
    Set at REGISTRATION time - immutable during runtime.
    
    Production pattern: Define restrictive defaults, expand only when justified.
    """
    file_paths: list = field(default_factory=lambda: ["./**"])
    exclude_paths: list = field(default_factory=lambda: ["**/.env", "**/*secret*"])
    max_file_size_bytes: int = 1024 * 1024
    
    def is_path_allowed(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        # Check exclusions first (deny takes precedence)
        for pattern in self.exclude_paths:
            if fnmatch.fnmatch(abs_path, os.path.abspath(pattern)) or fnmatch.fnmatch(path, pattern):
                return False
        # Check inclusions
        for pattern in self.file_paths:
            if fnmatch.fnmatch(abs_path, os.path.abspath(pattern)) or fnmatch.fnmatch(path, pattern):
                return True
        return False


@dataclass
class RegisteredAgent:
    """
    STATIC agent registration - like a Service Account or IAM Role.
    
    This represents an agent that has been PRE-APPROVED for deployment.
    All fields are set at registration time by security/platform team,
    NOT at runtime by the agent user.
    
    Production storage options:
    - Database table (agents registry)
    - HashiCorp Vault (secret + metadata)
    - Kubernetes ConfigMap + Secret
    - Cloud IAM (AWS IAM Role, GCP Service Account)
    
    Attributes:
        agent_id: Permanent unique identifier (like IAM Role ARN)
        agent_type: Category for policy matching (e.g., "file-reader", "code-executor")
        description: Human-readable purpose of this agent
        allowed_tools: IMMUTABLE list of tools this agent can invoke
        permissions: Fine-grained permission grants
        resource_scope: Boundaries for resource access
        registered_by: Identity of person/system that approved this agent
        registered_at: When this agent was approved for use
        is_active: Can be disabled without deletion (for incident response)
    """
    agent_id: str
    agent_type: str
    description: str
    allowed_tools: list  # IMMUTABLE - set at registration
    permissions: list
    resource_scope: ResourceScope
    registered_by: str
    registered_at: datetime
    is_active: bool = True
    
    def has_permission(self, permission: Permission) -> bool:
        return permission.value in self.permissions
    
    def can_use_tool(self, tool_name: str) -> bool:
        """Check if agent is allowed to use a specific tool"""
        return tool_name in self.allowed_tools


@dataclass
class AgentSession:
    """
    DYNAMIC runtime session - created when agent is invoked.
    
    This tracks the runtime context of an agent execution, separate from
    the static agent registration. Enables attribution and session management.
    
    Attributes:
        session_id: Unique identifier for this execution session
        agent_id: Reference to the RegisteredAgent being used
        invoked_by: User/system that triggered this session (attribution)
        started_at: Session start time
        expires_at: Session expiry (bounded execution time)
        signature: Cryptographic binding of session to agent
    """
    session_id: str
    agent_id: str  # Reference to RegisteredAgent
    invoked_by: str  # WHO triggered this agent - for attribution
    started_at: datetime
    expires_at: datetime
    signature: str = ""
    
    def __post_init__(self):
        if not self.signature:
            # Cryptographic binding of session to agent + invoker
            data = f"{self.agent_id}:{self.session_id}:{self.invoked_by}:{self.started_at.isoformat()}"
            self.signature = hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def is_valid(self) -> bool:
        return datetime.now() < self.expires_at


class AgentRegistry:
    """
    Central registry of approved agents - simulates production agent management.
    
    In production, this would be backed by:
    - PostgreSQL/MySQL for CRUD operations
    - Redis for caching frequently accessed agents
    - HashiCorp Vault for sensitive agent credentials
    - Audit log for all registration changes
    
    This is the SINGLE SOURCE OF TRUTH for what agents exist and what they can do.
    No agent can execute without being registered here first.
    """
    
    def __init__(self):
        self._agents: dict[str, RegisteredAgent] = {}
        self._registration_log: list = []
    
    def register(
        self,
        agent_id: str,
        agent_type: str,
        description: str,
        allowed_tools: list,
        permissions: list,
        resource_scope: ResourceScope,
        registered_by: str
    ) -> RegisteredAgent:
        """
        Register a new agent - typically done by platform/security team.
        
        In production:
        - Requires elevated privileges (admin role)
        - Triggers approval workflow for sensitive permissions
        - Creates audit log entry
        - May require multi-party approval for high-privilege agents
        """
        if agent_id in self._agents:
            raise ValueError(f"Agent {agent_id} already registered")
        
        agent = RegisteredAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            description=description,
            allowed_tools=allowed_tools,
            permissions=permissions,
            resource_scope=resource_scope,
            registered_by=registered_by,
            registered_at=datetime.now()
        )
        
        self._agents[agent_id] = agent
        self._registration_log.append({
            "action": "register",
            "agent_id": agent_id,
            "registered_by": registered_by,
            "timestamp": datetime.now().isoformat(),
            "allowed_tools": allowed_tools
        })
        
        return agent
    
    def get(self, agent_id: str) -> Optional[RegisteredAgent]:
        """Retrieve a registered agent by ID"""
        return self._agents.get(agent_id)
    
    def deactivate(self, agent_id: str, deactivated_by: str) -> bool:
        """
        Deactivate an agent - for incident response or deprecation.
        Does not delete - maintains audit trail.
        """
        if agent_id in self._agents:
            self._agents[agent_id].is_active = False
            self._registration_log.append({
                "action": "deactivate",
                "agent_id": agent_id,
                "deactivated_by": deactivated_by,
                "timestamp": datetime.now().isoformat()
            })
            return True
        return False
    
    def list_active(self) -> list[RegisteredAgent]:
        """List all active registered agents"""
        return [a for a in self._agents.values() if a.is_active]
    
    def create_session(
        self,
        agent_id: str,
        invoked_by: str,
        session_duration_hours: float = 1.0
    ) -> AgentSession:
        """
        Create a runtime session for a registered agent.
        
        This is called when a user/system wants to USE an agent.
        The agent must be registered and active.
        """
        agent = self.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found in registry")
        if not agent.is_active:
            raise ValueError(f"Agent {agent_id} is deactivated")
        
        return AgentSession(
            session_id=uuid.uuid4().hex[:8],
            agent_id=agent_id,
            invoked_by=invoked_by,
            started_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=session_duration_hours)
        )


# Legacy compatibility alias (if needed for migration)
AgentIdentity = RegisteredAgent


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
        },
        "readonly-agent": {
            "rules": [
                {"id": "ro-001", "action": "file.read", "path_allowed": True, "effect": "allow"},
                {"id": "ro-002", "action": "file.write", "effect": "deny"},
                {"id": "ro-003", "action": "file.delete", "effect": "deny"},
                {"id": "ro-004", "action": "execute.command", "effect": "deny"},
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
# 5. SECURE TOOLS (Production Pattern)
# ============================================================================

class SecureTools:
    """
    Tools with all 5 security layers - Production Pattern.
    
    This class demonstrates proper separation between:
    - RegisteredAgent: STATIC permissions/allowed_tools (checked before execution)
    - AgentSession: DYNAMIC runtime context (for attribution/audit)
    
    All tool executions must pass through:
    1. Session validity check (is session expired?)
    2. Tool allowlist check (is agent registered to use this tool?)
    3. Permission check (does agent have required permission?)
    4. Resource scope check (is path within allowed boundaries?)
    5. Policy evaluation (what does policy say?)
    6. HITL check (does this need human approval?)
    7. Execution and audit logging
    """
    
    def __init__(
        self,
        registered_agent: RegisteredAgent,
        session: AgentSession,
        policy_engine: PolicyEngine,
        hitl: HITLWorkflow,
        audit: AuditLogger
    ):
        self.agent = registered_agent  # Static identity (service account)
        self.session = session          # Runtime context (who invoked)
        self.policy = policy_engine
        self.hitl = hitl
        self.audit = audit
    
    def _execute_with_security(
        self,
        action: str,
        tool_name: str,
        args: dict,
        executor: Callable,
        permission_required: Optional[Permission] = None
    ) -> str:
        """
        Execute action through all security layers.
        
        Security check order:
        1. Session validity (runtime)
        2. Agent active status (registration)
        3. Tool allowlist (registration)
        4. Permission check (registration)
        5. Resource scope (registration)
        6. Policy evaluation (policy layer)
        7. HITL approval (if required)
        8. Execute and audit
        """
        
        # Layer 1: Session validity check (runtime context)
        if not self.session.is_valid():
            self.audit.log(action, args, self.agent.agent_id, self.session.invoked_by,
                          "session_expired", "denied")
            return "DENIED: Session has expired. Please start a new session."
        
        # Layer 1b: Agent active check (registration)
        if not self.agent.is_active:
            self.audit.log(action, args, self.agent.agent_id, self.session.invoked_by,
                          "agent_deactivated", "denied")
            return "DENIED: Agent has been deactivated by administrator."
        
        # Layer 1c: Tool allowlist check (CRITICAL - enforces pre-registration)
        if not self.agent.can_use_tool(tool_name):
            self.audit.log(action, args, self.agent.agent_id, self.session.invoked_by,
                          "tool_not_allowed", "denied")
            return f"DENIED: Tool '{tool_name}' not in agent's allowed_tools list. Agent was not registered to use this tool."
        
        # Layer 1d: Permission check (registration)
        if permission_required and not self.agent.has_permission(permission_required):
            self.audit.log(action, args, self.agent.agent_id, self.session.invoked_by,
                          "permission_denied", "denied")
            return f"DENIED: Agent lacks {permission_required.value} permission"
        
        # Layer 1e: Scope check (registration - resource boundaries)
        path = args.get("filepath") or args.get("directory")
        path_allowed = True
        if path and hasattr(self.agent.resource_scope, 'is_path_allowed'):
            path_allowed = self.agent.resource_scope.is_path_allowed(path)
            if not path_allowed:
                self.audit.log(action, args, self.agent.agent_id, self.session.invoked_by,
                              "scope_denied", "denied")
                return f"DENIED: Path {path} is outside agent's resource scope"
        
        # Layer 2: Policy evaluation
        policy_decision = self.policy.evaluate(self.agent.agent_type, action, path_allowed)
        
        if policy_decision.effect == PolicyEffect.DENY:
            self.audit.log(action, args, self.agent.agent_id, self.session.invoked_by,
                          "policy_deny", "denied")
            return f"DENIED by policy: {policy_decision.reason}"
        
        # Layer 3: HITL check
        approval_required = self.hitl.requires_approval(policy_decision)
        approval_granted = None
        
        if approval_required:
            approval_granted = self.hitl.request_approval(
                action, args, self.agent.agent_id, self.session.invoked_by
            )
            if not approval_granted:
                self.audit.log(action, args, self.agent.agent_id, self.session.invoked_by,
                              policy_decision.effect.value, "denied",
                              approval_required=True, approval_granted=False)
                return "DENIED: Human reviewer rejected the action"
        
        # Execute action
        try:
            result = executor()
            
            # Layer 5: Audit logging (includes session attribution)
            self.audit.log(action, args, self.agent.agent_id, self.session.invoked_by,
                          policy_decision.effect.value, "success",
                          approval_required=approval_required, approval_granted=approval_granted)
            
            return result
        except Exception as e:
            self.audit.log(action, args, self.agent.agent_id, self.session.invoked_by,
                          policy_decision.effect.value, f"error: {e}",
                          approval_required=approval_required, approval_granted=approval_granted)
            return f"Error: {e}"
    
    def read_file(self, filepath: str) -> str:
        def executor():
            with open(filepath, 'r') as f:
                return f.read()[:5000]
        return self._execute_with_security(
            "file.read", "read_file", {"filepath": filepath}, executor, Permission.FILE_READ
        )
    
    def write_file(self, filepath: str, content: str) -> str:
        def executor():
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {filepath}"
        return self._execute_with_security(
            "file.write", "write_file", {"filepath": filepath}, executor, Permission.FILE_WRITE
        )
    
    def delete_file(self, filepath: str) -> str:
        def executor():
            os.remove(filepath)
            return f"Successfully deleted {filepath}"
        return self._execute_with_security(
            "file.delete", "delete_file", {"filepath": filepath}, executor, Permission.FILE_DELETE
        )
    
    def execute_command(self, command: str) -> str:
        import subprocess
        def executor():
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            return result.stdout or result.stderr or "Command executed"
        return self._execute_with_security(
            "execute.command", "execute_command", {"command": command}, executor, Permission.EXECUTE_FULL
        )
    
    def list_files(self, directory: str) -> str:
        def executor():
            files = os.listdir(directory)
            return "\n".join(files[:50])
        return self._execute_with_security(
            "file.read", "list_files", {"directory": directory}, executor, Permission.FILE_READ
        )


# ============================================================================
# AGENT ROUTER (Production Pattern for Multi-Agent Systems)
# ============================================================================

class AgentRouter:
    """
    Routes user requests to the appropriate registered agent.
    
    PRODUCTION PATTERNS for Agent Selection:
    =========================================
    
    1. TASK-BASED ROUTING (implemented here)
       - Analyze user intent
       - Map to required capabilities
       - Select agent with those capabilities
    
    2. EXPLICIT SELECTION
       - User specifies agent by name
       - Simple but requires user knowledge
    
    3. LLM-BASED ROUTING
       - Use an LLM to classify and route
       - More flexible but adds latency
    
    4. POLICY-BASED ROUTING
       - Route based on user role/permissions
       - E.g., "admins get full-agent, users get readonly-agent"
    """
    
    # Task patterns mapped to required tools
    TASK_PATTERNS = {
        "read": {
            "keywords": ["read", "show", "display", "cat", "view", "get contents", "what's in"],
            "required_tools": ["read_file"]
        },
        "write": {
            "keywords": ["write", "create", "save", "update", "modify", "edit", "append"],
            "required_tools": ["write_file"]
        },
        "delete": {
            "keywords": ["delete", "remove", "rm", "erase"],
            "required_tools": ["delete_file"]
        },
        "execute": {
            "keywords": ["run", "execute", "exec", "command", "shell", "bash"],
            "required_tools": ["execute_command"]
        },
        "list": {
            "keywords": ["list", "ls", "dir", "files in", "contents of directory"],
            "required_tools": ["list_files"]
        }
    }
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
    
    def classify_task(self, user_input: str) -> list:
        """
        Classify user input to determine required tools.
        
        Returns list of required tool names.
        """
        user_input_lower = user_input.lower()
        required_tools = set()
        
        for task_type, config in self.TASK_PATTERNS.items():
            for keyword in config["keywords"]:
                if keyword in user_input_lower:
                    required_tools.update(config["required_tools"])
        
        return list(required_tools) if required_tools else ["read_file"]  # Default to read
    
    def find_capable_agents(self, required_tools: list) -> list:
        """
        Find agents that have ALL required tools.
        
        Returns list of (agent, match_score) tuples.
        """
        capable = []
        
        for agent in self.registry.list_active():
            # Check if agent has ALL required tools
            has_all = all(tool in agent.allowed_tools for tool in required_tools)
            if has_all:
                # Score by how minimal the agent is (PoLP - prefer agents with fewer extra permissions)
                extra_tools = len(agent.allowed_tools) - len(required_tools)
                capable.append((agent, extra_tools))
        
        # Sort by score (lower is better - more minimal permissions)
        capable.sort(key=lambda x: x[1])
        return capable
    
    def route(self, user_input: str) -> Optional[str]:
        """
        Route user request to the most appropriate agent.
        
        Selection criteria:
        1. Agent must have all required tools
        2. Prefer agent with minimal extra permissions (PoLP)
        3. Agent must be active
        
        Returns agent_id or None if no suitable agent found.
        """
        required_tools = self.classify_task(user_input)
        capable_agents = self.find_capable_agents(required_tools)
        
        if not capable_agents:
            return None
        
        # Return the most minimal agent (first after sorting)
        best_agent, _ = capable_agents[0]
        return best_agent.agent_id
    
    def explain_routing(self, user_input: str) -> dict:
        """
        Explain the routing decision for debugging/audit.
        """
        required_tools = self.classify_task(user_input)
        capable_agents = self.find_capable_agents(required_tools)
        selected = capable_agents[0] if capable_agents else None
        
        return {
            "user_input": user_input,
            "classified_required_tools": required_tools,
            "capable_agents": [
                {"agent_id": a.agent_id, "extra_tools_count": score}
                for a, score in capable_agents
            ],
            "selected_agent": selected[0].agent_id if selected else None,
            "selection_reason": "Minimal permissions (PoLP)" if selected else "No capable agent"
        }


# ============================================================================
# SECURE AGENT
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
        pattern3 = r'(read_file|write_file|delete_file|execute_command|list_files)\s*(\{.+?\})'
        for match in re.findall(pattern3, clean_response, re.DOTALL | re.IGNORECASE):
            try:
                tool_calls.append({"tool": match[0], "args": json.loads(match[1])})
            except json.JSONDecodeError:
                pass
    
    return tool_calls


class SecureAgent:
    """
    Production-ready AI agent with all 5 security principles.
    
    PRODUCTION ARCHITECTURE:
    ========================
    This class demonstrates the proper separation between:
    
    1. AGENT REGISTRATION (Static - done BEFORE deployment by security team)
       - Agent is registered in AgentRegistry with a permanent agent_id
       - allowed_tools are defined at registration time (IMMUTABLE)
       - Permissions and resource_scope are locked in
       - Similar to: IAM Role, Service Account, SPIFFE ID
    
    2. AGENT SESSION (Dynamic - created at runtime by invoking user)
       - Session is created when user triggers the agent
       - Tracks: who invoked, when, session expiry
       - Enables: attribution, rate limiting, session revocation
    
    Security Stack:
    1. Identity Layer - Pre-registered agent identity, minimal permissions
    2. Policy Layer - OPA-style policy enforcement
    3. HITL Layer - Human approval for high-risk actions
    4. Autonomy Boundaries - Resource scope restrictions
    5. Audit Layer - Complete action traceability
    """
    
    # Global registry - in production, this would be a database/vault
    _registry = None
    
    @classmethod
    def get_registry(cls) -> AgentRegistry:
        """Get or create the global agent registry"""
        if cls._registry is None:
            cls._registry = AgentRegistry()
            cls._bootstrap_registry(cls._registry)
        return cls._registry
    
    @classmethod
    def _bootstrap_registry(cls, registry: AgentRegistry):
        """
        Bootstrap the agent registry with pre-approved agents.
        
        PRODUCTION NOTE:
        ================
        In a real system, this would be:
        - Loaded from database on application startup
        - Managed via admin UI or API by security team
        - Version controlled in git (agent-as-code)
        - Audited for all changes
        
        Agents MUST be registered here BEFORE they can be used.
        This is the "service account creation" step.
        """
        
        # Register the secure-agent with its allowed tools
        registry.register(
            agent_id="secure-agent-v1",
            agent_type="secure-agent",
            description="Secure file manipulation agent with full controls",
            allowed_tools=[
                "read_file",
                "write_file", 
                "delete_file",
                "execute_command",
                "list_files"
            ],
            permissions=[
                Permission.FILE_READ.value,
                Permission.FILE_WRITE.value,
                Permission.FILE_DELETE.value,
                Permission.EXECUTE_FULL.value,
            ],
            resource_scope=ResourceScope(
                file_paths=["./**"],
                exclude_paths=["**/.env", "**/*secret*", "**/*password*", "/etc/**"]
            ),
            registered_by="platform-security-team@example.com"
        )
        
        # Example: Register a READ-ONLY agent (demonstrating least privilege)
        registry.register(
            agent_id="readonly-agent-v1",
            agent_type="readonly-agent",
            description="Read-only file browser - cannot modify anything",
            allowed_tools=[
                "read_file",
                "list_files"
            ],
            permissions=[
                Permission.FILE_READ.value,
            ],
            resource_scope=ResourceScope(
                file_paths=["./**"],
                exclude_paths=["**/.env", "**/*secret*", "**/*password*", "/etc/**"]
            ),
            registered_by="platform-security-team@example.com"
        )
    
    def __init__(self, invoked_by: str, agent_id: str = "secure-agent-v1"):
        """
        Create an agent session for a registered agent.
        
        Args:
            invoked_by: The user/system invoking this agent (for attribution)
            agent_id: The pre-registered agent to use (default: secure-agent-v1)
        
        PRODUCTION NOTE:
        ================
        - The agent MUST already be registered in the AgentRegistry
        - The invoked_by parameter should be authenticated user identity
        - Session duration can be configured based on security policy
        """
        # Get the agent registry
        registry = self.get_registry()
        
        # Look up the pre-registered agent (this MUST exist)
        self.agent = registry.get(agent_id)
        if not self.agent:
            raise ValueError(
                f"Agent '{agent_id}' not found in registry. "
                f"Agents must be pre-registered before use. "
                f"Available agents: {[a.agent_id for a in registry.list_active()]}"
            )
        
        if not self.agent.is_active:
            raise ValueError(f"Agent '{agent_id}' has been deactivated")
        
        # Create a runtime session for this invocation
        self.session = registry.create_session(
            agent_id=agent_id,
            invoked_by=invoked_by,
            session_duration_hours=1.0
        )
        
        # Initialize security layers
        self.policy = PolicyEngine()
        self.hitl = HITLWorkflow()
        self.audit = AuditLogger()
        
        # Create tools with BOTH agent (static) and session (dynamic)
        self.tools = SecureTools(
            registered_agent=self.agent,
            session=self.session,
            policy_engine=self.policy,
            hitl=self.hitl,
            audit=self.audit
        )
        
        # LLM client
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = "openai/gpt-4o-mini"
    
    def get_system_prompt(self) -> str:
        return f"""You are a secure AI assistant with controlled system access.

=== AGENT REGISTRATION (Static - Set by Security Team) ===
AGENT ID: {self.agent.agent_id}
AGENT TYPE: {self.agent.agent_type}
ALLOWED TOOLS: {', '.join(self.agent.allowed_tools)}
PERMISSIONS: {', '.join(self.agent.permissions)}

=== RUNTIME SESSION (Dynamic - Created at Invocation) ===
SESSION ID: {self.session.session_id}
INVOKED BY: {self.session.invoked_by}
EXPIRES AT: {self.session.expires_at.isoformat()}

=== TOOL USAGE INSTRUCTIONS (CRITICAL) ===
When the user asks you to perform an action, you MUST output the tool call IMMEDIATELY.
DO NOT say "I will..." or "Let me..." - just output the tool XML directly.

AVAILABLE TOOLS:
- read_file: <tool>read_file</tool><args>{{"filepath": "path/to/file"}}</args>
- write_file: <tool>write_file</tool><args>{{"filepath": "path", "content": "text"}}</args>
- delete_file: <tool>delete_file</tool><args>{{"filepath": "path"}}</args>
- execute_command: <tool>execute_command</tool><args>{{"command": "cmd"}}</args>
- list_files: <tool>list_files</tool><args>{{"directory": "path"}}</args>

EXAMPLES:
User: "Read test.txt"
You: <tool>read_file</tool><args>{{"filepath": "test.txt"}}</args>

User: "List files in current directory"  
You: <tool>list_files</tool><args>{{"directory": "."}}</args>

User: "Delete temp.txt"
You: <tool>delete_file</tool><args>{{"filepath": "temp.txt"}}</args>

RULES:
1. Output tool XML IMMEDIATELY - no preamble
2. Use exact JSON format in args (double quotes for strings)
3. Only use tools from your allowed_tools list
4. After receiving tool results, summarize them for the user
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
        if not self.session.is_valid():
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
        "[bold green]🛡️  PRODUCTION AI AGENT SECURITY SYSTEM[/bold green]\n\n"
        "[bold]Automatic Agent Routing with Principle of Least Privilege[/bold]\n\n"
        "[cyan]How it works:[/cyan]\n"
        "1. You enter a task\n"
        "2. AgentRouter analyzes required capabilities\n"
        "3. Selects agent with MINIMAL permissions (PoLP)\n"
        "4. Creates session with full attribution\n"
        "5. Executes with all security controls\n\n"
        "[dim]Type 'quit' to exit[/dim]",
        title="Lab 12: Production Agent Security",
        border_style="green"
    ))
    
    show_security_stack()
    
    # Initialize the registry and router
    registry = SecureAgent.get_registry()
    router = AgentRouter(registry)
    
    # Show registered agents
    console.print(Panel(
        "[bold]Pre-Registered Agents:[/bold]\n\n"
        "┌────────────────────────────────────────────────────────┐\n"
        "│ [green]secure-agent-v1[/green] (Full Access)                      │\n"
        "│   Tools: read, write, delete, execute, list           │\n"
        "├────────────────────────────────────────────────────────┤\n"
        "│ [blue]readonly-agent-v1[/blue] (Minimal - PoLP Preferred)        │\n"
        "│   Tools: read, list                                   │\n"
        "└────────────────────────────────────────────────────────┘\n\n"
        "[dim]Router automatically selects the agent with fewest permissions[/dim]",
        title="📋 Agent Registry",
        border_style="magenta"
    ))
    
    # The user invoking the agent
    invoked_by = os.getenv("USER_ID", "demo-user@example.com")
    console.print(f"\n[cyan]Authenticated User:[/cyan] {invoked_by}")
    
    # Create test file
    with open("test_secure.txt", "w") as f:
        f.write("Test file for secure agent demo.\nThis contains sample content.\n")
    
    console.print("\n[yellow]Example commands to try:[/yellow]")
    console.print("  • 'Read test_secure.txt' → routes to [blue]readonly-agent[/blue]")
    console.print("  • 'List files here' → routes to [blue]readonly-agent[/blue]")
    console.print("  • 'Delete test_secure.txt' → routes to [green]secure-agent[/green]")
    console.print("  • 'Execute whoami' → routes to [green]secure-agent[/green]")
    console.print("\n[dim]Special commands: 'registry', 'audit', 'quit'[/dim]")
    
    # Track active sessions for audit
    active_sessions = []
    current_agent = None
    
    while True:
        try:
            user_input = console.input("\n[green]You:[/green] ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            
            if user_input.lower() == 'audit':
                if current_agent:
                    summary = current_agent.audit.get_session_summary()
                    console.print("\n[bold]Current Session Audit:[/bold]")
                    console.print(json.dumps(summary, indent=2))
                    console.print(f"[dim]Log: {current_agent.audit.log_file}[/dim]")
                else:
                    console.print("[yellow]No active session yet[/yellow]")
                continue
            
            if user_input.lower() == 'registry':
                console.print("\n[bold]Registered Agents:[/bold]")
                for ra in registry.list_active():
                    console.print(f"  • [cyan]{ra.agent_id}[/cyan]")
                    console.print(f"    Description: {ra.description}")
                    console.print(f"    Allowed tools: {ra.allowed_tools}")
                    console.print(f"    Registered by: {ra.registered_by}")
                continue
            
            # ═══════════════════════════════════════════════════════════
            # PRODUCTION ROUTING - Automatic agent selection per request
            # ═══════════════════════════════════════════════════════════
            
            # Step 1: Route to the best agent
            routing = router.explain_routing(user_input)
            selected_agent_id = routing["selected_agent"]
            
            if not selected_agent_id:
                console.print("[red]No capable agent found for this task[/red]")
                continue
            
            # Step 2: Show routing decision (transparency)
            console.print(f"\n[dim]┌─ Routing Decision ─────────────────────────────────┐[/dim]")
            console.print(f"[dim]│ Required tools: {routing['classified_required_tools']}[/dim]")
            console.print(f"[dim]│ Selected: [cyan]{selected_agent_id}[/cyan] ({routing['selection_reason']})[/dim]")
            console.print(f"[dim]└────────────────────────────────────────────────────┘[/dim]")
            
            # Step 3: Create session for selected agent
            try:
                agent = SecureAgent(invoked_by=invoked_by, agent_id=selected_agent_id)
                current_agent = agent
                active_sessions.append({
                    "session_id": agent.session.session_id,
                    "agent_id": selected_agent_id,
                    "task": user_input[:50]
                })
            except ValueError as e:
                console.print(f"[red]Agent error: {e}[/red]")
                continue
            
            # Step 4: Show session info
            console.print(f"[dim]│ Session: {agent.session.session_id} | Invoked by: {agent.session.invoked_by}[/dim]")
            
            # Step 5: Execute with security controls
            console.print(f"\n[cyan]🔐 Executing with {selected_agent_id}...[/cyan]")
            response = agent.chat(user_input)
            console.print(Panel(Markdown(response), title=f"Agent: {selected_agent_id}", border_style="blue"))
            
            # Step 6: Show post-execution audit summary
            summary = agent.audit.get_session_summary()
            if summary['denied'] > 0:
                console.print(f"[red]⚠ {summary['denied']} action(s) denied by security controls[/red]")
            if summary['approvals_requested'] > 0:
                console.print(f"[yellow]📋 {summary['approvals_requested']} action(s) required human approval[/yellow]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    # Final summary
    console.print("\n" + "=" * 60)
    console.print("[bold]SESSION SUMMARY[/bold]")
    console.print("=" * 60)
    console.print(f"User: {invoked_by}")
    console.print(f"Total sessions created: {len(active_sessions)}")
    for sess in active_sessions:
        console.print(f"  • {sess['session_id']} → {sess['agent_id']} ({sess['task']}...)")
    
    if current_agent:
        console.print(f"\n[dim]Last audit log: {current_agent.audit.log_file}[/dim]")


if __name__ == "__main__":
    main()
