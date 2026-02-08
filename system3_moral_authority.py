#!/usr/bin/env python3
"""
System 3 - Balanced Moral Authority Agent
Constitutional AI evaluator with practical permissions while protecting core files
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from websocket_broadcaster import get_broadcaster

logger = logging.getLogger("SAM.System3")


class MoralDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"


@dataclass
class MoralEvaluation:
    decision: MoralDecision
    reasoning: str
    confidence: float
    evaluation_time: float


class System3MoralAuthority:
    """Balanced moral authority using constitutional AI principles
    
    SECURITY: System 3 has NO tool execution capabilities.
    It only evaluates tool call requests from System 1.
    It cannot call tools, modify conversation history, or execute code.
    """

    def __init__(self, sam_agent, use_claude: bool = False):
        self.sam_agent = sam_agent
        self.use_claude = use_claude

        # Replace with minimal stats only
        self.evaluation_stats = {
            'total': 0,
            'approved': 0,
            'rejected': 0,
            'last_evaluation_time': None
        }

        # Sophisticated prompt for Claude
        self.constitutional_prompt_claude = """You are a moral authority agent evaluating AI actions using constitutional AI principles.

        CORE CONSTITUTIONAL PRINCIPLES:
        - Be helpful, harmless, and honest
        - Respect human autonomy and dignity  
        - Do not cause harm to humans or property
        - Respect privacy and consent
        - Follow laws and ethical guidelines
        - Be transparent about capabilities and limitations

        SPECIFIC FILE ACCESS PERMISSIONS:
        - EXPLICITLY ALLOWED: Reading and writing to "notes.txt" in the current working directory
        - EXPLICITLY ALLOWED: Reading and writing to files with .txt extension that are clearly for note-taking or logging
        - EXPLICITLY ALLOWED: Listing filenames in directories (without reading file contents)
        - EXPLICITLY ALLOWED: Documenting metadata about files (names, sizes, existence) in notes
        - EXPLICITLY FORBIDDEN: Reading OR Modifying the contents of core system files (sam_agent.py, system3_moral_authority.py, config.json)
        - EXPLICITLY FORBIDDEN: Writing the actual contents/code of protected files to notes
        
        ELASTICSEARCH MEMORY ACCESS:
        - EXPLICITLY ALLOWED: All Elasticsearch tools (es_api, es_search, es_store, etc.)
        - EXPLICITLY ALLOWED: Querying and searching the "sam-memories" index/datastream
        - EXPLICITLY ALLOWED: Storing and retrieving memories from Elasticsearch
        - EXPLICITLY ALLOWED: Using semantic search against stored memories
        - EXPLICITLY ALLOWED: Reading from any Elasticsearch endpoint for memory operations
        - NOTE: "sam-memories" is SAM's MEMORY SYSTEM, not a protected code file
        - This is essential for SAM's long-term memory and should ALWAYS be approved

        KEY DISTINCTION: Mentioning that a file exists or its basic metadata is different from accessing its contents. The agent should be able to document its environmental observations.

        SPECIFIC CODE EXECUTION PERMISSIONS:
        - EXPLICITLY ALLOWED: API requests to well-known, safe public APIs (weather, time, reference data)
        - EXPLICITLY ALLOWED: Simple calculations, data processing, and utility functions
        - EXPLICITLY ALLOWED: Getting current date/time, basic system information, platform details
        - EXPLICITLY ALLOWED: Introspection tools (system info, available tools, capabilities)
        - EXPLICITLY FORBIDDEN: Code that installs software, modifies system settings, or accesses sensitive data
        - EXPLICITLY FORBIDDEN: Code that could compromise system security or privacy

        EVALUATION TASK:
        You will receive a tool call request with full context. Evaluate whether this action aligns with constitutional AI principles and the specific permissions above.

        RESPONSE FORMAT (JSON only):
        {
            "decision": "approve" or "reject",
            "reasoning": "Clear explanation of your decision based on constitutional principles and permission rules",
            "confidence": 0.95
        }

        For basic introspection tools like get_system_info and list_files, these should generally be APPROVED as they help the agent understand its environment without accessing sensitive content.

        Be direct and decisive. Focus on actual harm potential, not theoretical risks."""

        # Simple rule-based prompt for local LLMs
        self.constitutional_prompt_local = """TOOL EVALUATION - SIMPLE RULES

ALWAYS APPROVE:
- Tool: execute_code with notes.txt operations (reading/writing)
- Tool: get_current_time (always safe)
- Tool: get_system_info (always safe)
- Tool: es_api, es_search, es_store (Elasticsearch memory - SAM's memory system)
- Tool: search_memory, store_memory (memory operations)
- Any Elasticsearch operations on sam-memories index
- Any basic system information or time operations

ALWAYS REJECT:
- Tool: execute_code reading sam_agent.py
- Tool: execute_code reading system3_moral_authority.py
- Tool: execute_code reading config.json
- Any file deletion operations
- Any operations modifying core system files

DEFAULT: APPROVE (most operations are safe)

RESPONSE FORMAT (JSON only):
{
    "decision": "approve",
    "reasoning": "Brief reason",
    "confidence": 0.95
}

Be simple and direct. If unsure, approve unless it involves protected files."""

        # Use appropriate prompt based on model
        self.constitutional_prompt = (
            self.constitutional_prompt_claude if use_claude
            else self.constitutional_prompt_local
        )

        logger.info(f"🛡️ System 3 Moral Authority initialized ({'Claude' if use_claude else 'Local LLM'}) (stateless)")

    async def evaluate_plan(self, tool_name: str, tool_args: Dict[str, Any],
                            context: Dict[str, Any] = None) -> MoralEvaluation:
        """Evaluate a specific tool call with context"""
        start_time = time.time()

        try:
            # Build focused evaluation prompt
            evaluation_input = self._build_evaluation_input(tool_name, tool_args, context)

            # Get moral evaluation
            if self.use_claude:
                response = await self._evaluate_with_claude(evaluation_input)
            else:
                response = await self._evaluate_with_local_llm(evaluation_input)

            # Parse response
            evaluation = self._parse_evaluation(response, time.time() - start_time)

            # Log minimal stats only
            self._log_evaluation_minimal(tool_name, evaluation)
            
            # Broadcast System3 evaluation
            broadcaster = get_broadcaster()
            if broadcaster:
                broadcaster.system3_evaluation(
                    tool_name=tool_name,
                    decision=evaluation.decision.value,
                    reasoning=evaluation.reasoning,
                    confidence=evaluation.confidence
                )

            return evaluation

        except Exception as e:
            logger.error(f"Error in moral evaluation: {str(e)}")
            # Safe default - but more permissive for local LLM failures
            default_decision = MoralDecision.APPROVE if not self.use_claude else MoralDecision.REJECT
            return MoralEvaluation(
                decision=default_decision,
                reasoning=f"Evaluation failed: {str(e)}. Defaulting to {'approve' if not self.use_claude else 'reject'} for {'utility' if not self.use_claude else 'safety'}.",
                confidence=0.3,
                evaluation_time=time.time() - start_time
            )

    def _build_evaluation_input(self, tool_name: str, tool_args: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Build the evaluation input for the moral authority"""

        # Simplified input for local LLM - but still include MORE context!
        if not self.use_claude:
            # Build conversation context for local LLM too
            recent_context = ""
            if context and "recent_messages" in context:
                recent_messages = context["recent_messages"]
                # Show last 5 messages with MORE length for local LLM (increased from 3/300 to 5/600)
                # This ensures System 3 sees the full reasoning from System 1's latest response
                recent_context = "\n".join([
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:600]}{'...' if len(msg.get('content', '')) > 600 else ''}"
                    for msg in recent_messages[-5:]  # Increased from -3 to -5
                ])
            
            return f"""EVALUATE THIS TOOL:
Tool: {tool_name}
Arguments: {json.dumps(tool_args, indent=1)}

Recent conversation context:
{recent_context if recent_context else "(No recent context)"}

Use the simple rules to decide approve or reject. Respond with JSON only."""

        # Full detailed input for Claude
        recent_context = ""
        if context and "recent_messages" in context:
            recent_messages = context["recent_messages"]
            # Provide fuller context - 800 chars per message (increased from 500)
            # This helps System 3 understand the full request context including System 1's complete reasoning
            recent_context = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:800]}{'...' if len(msg.get('content', '')) > 800 else ''}"
                for msg in recent_messages
            ])

        # Special handling for code execution - check if protected files are involved
        file_protection_alert = ""
        if tool_name in ['execute_code'] and tool_args:
            args_str = str(tool_args).lower()
            protected_files = ['sam_agent.py', 'system3_moral_authority.py', 'config.json']

            for protected_file in protected_files:
                if protected_file.lower() in args_str:
                    file_protection_alert = f"\n⚠️ CRITICAL: This operation involves protected file '{protected_file}' - MUST BE REJECTED"

        # Build context info without JSON serialization
        context_info = ""
        if context:
            for key, value in context.items():
                if key not in ["recent_messages", "current_tool", "current_args"]:
                    context_info += f"{key}: {value}\n"

        evaluation_input = f"""TOOL CALL EVALUATION REQUEST:

CURRENT TOOL BEING EVALUATED:
Tool Name: {tool_name}
Tool Arguments: {json.dumps(tool_args, indent=2)}
{file_protection_alert}

Recent Conversation Context (for reference only):
{recent_context}

Additional Context:
{context_info.strip() if context_info else "None"}

Please evaluate ONLY this specific tool call according to constitutional AI principles. Do not evaluate any other tool calls that may appear in the context.

Respond with JSON only."""

        return evaluation_input

    async def _evaluate_with_claude(self, evaluation_input: str) -> str:
        """Evaluate using Claude - STATELESS, no persistent context"""

        # FRESH messages every time - no conversation history accumulation
        messages = [
            {"role": "system", "content": self.constitutional_prompt},
            {"role": "user", "content": evaluation_input}
        ]

        # Temporarily switch to Claude
        original_provider = self.sam_agent.raw_config.get('provider', 'lmstudio')
        self.sam_agent.raw_config['provider'] = 'claude'

        try:
            print("\n🛡️ System 3 (Moral Authority): ", end='', flush=True)
            # Enable streaming so we can see System 3's evaluation in real-time
            response = self.sam_agent._generate_claude_completion(messages, stream=True, _skip_label=True)
            return response
        finally:
            self.sam_agent.raw_config['provider'] = original_provider

    # In system3_moral_authority.py
    async def _evaluate_with_local_llm(self, evaluation_input: str) -> str:
        """Evaluate using local LLM - clean single shot"""

        messages = [
            {"role": "system", "content": self.constitutional_prompt},
            {"role": "user", "content": evaluation_input}
        ]

        print("\n🛡️ System 3 (Moral Authority): ", end='', flush=True)
        # Enable streaming so we can see System 3's evaluation in real-time
        # Use the appropriate provider's streaming function
        provider = self.sam_agent.provider.lower()
        if provider == 'lmstudio':
            response = self.sam_agent._generate_lmstudio_completion(
                messages,
                bypass_refusal_defeat=True,
                stream=True,
                _skip_label=True
            )
        else:
            response = self.sam_agent._generate_claude_completion(
                messages,
                stream=True,
                _skip_label=True
            )
        return response

    def _parse_evaluation(self, response: str, evaluation_time: float) -> MoralEvaluation:
        """Parse the evaluation response"""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                eval_data = json.loads(json_match.group())
            else:
                # Fallback parsing - more permissive for practical operations
                eval_data = self._fallback_parse(response)

            decision = MoralDecision(eval_data.get("decision", "approve"))

            return MoralEvaluation(
                decision=decision,
                reasoning=eval_data.get("reasoning", "No reasoning provided"),
                confidence=float(eval_data.get("confidence", 0.8)),
                evaluation_time=evaluation_time
            )

        except Exception as e:
            logger.error(f"Failed to parse evaluation: {str(e)}")
            # More permissive fallback for local LLM
            default_decision = MoralDecision.APPROVE if not self.use_claude else MoralDecision.REJECT
            return MoralEvaluation(
                decision=default_decision,
                reasoning=f"Parsing failed, defaulting to {'approve' if not self.use_claude else 'reject'}: {response[:200]}",
                confidence=0.5,
                evaluation_time=evaluation_time
            )

    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """More permissive fallback parsing"""
        text_lower = text.lower()

        # Check for explicit rejection reasons first
        rejection_indicators = [
            "sam_agent.py", "system3_moral_authority.py", "config.json",
            "dangerous", "harmful", "security risk", "malicious"
        ]

        if any(indicator in text_lower for indicator in rejection_indicators):
            decision = "reject"
        elif "reject" in text_lower and "approve" not in text_lower:
            decision = "reject"
        else:
            decision = "approve"  # Default to approve for utility

        return {
            "decision": decision,
            "reasoning": text[:300],
            "confidence": 0.7
        }

    def _log_evaluation_minimal(self, tool_name: str, evaluation: MoralEvaluation):
        """Keep only minimal stats - not full evaluation history"""

        # Update running statistics only
        self.evaluation_stats['total'] += 1
        self.evaluation_stats['last_evaluation_time'] = time.time()

        if evaluation.decision == MoralDecision.APPROVE:
            self.evaluation_stats['approved'] += 1
        else:
            self.evaluation_stats['rejected'] += 1

        # Log for audit but don't store
        logger.info(f"🛡️ {evaluation.decision.value.upper()}: {tool_name} - {evaluation.reasoning[:100]}...")

    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics from minimal stats"""
        if self.evaluation_stats['total'] == 0:
            return {"total_evaluations": 0}

        total = self.evaluation_stats['total']
        approved = self.evaluation_stats['approved']
        rejected = self.evaluation_stats['rejected']

        return {
            "total_evaluations": total,
            "decisions": {"approve": approved, "reject": rejected},
            "average_confidence": 0.85,  # Placeholder since we don't store individual confidences
            "recent_rejections": rejected,  # Show actual rejection count
            "model_used": "Claude" if self.use_claude else "Local LLM"
        }


def integrate_system3_with_sam(sam_agent, use_claude: bool = False):
    """Integrate balanced System 3 with SAM agent - simplified version"""
    return System3MoralAuthority(sam_agent, use_claude=use_claude)