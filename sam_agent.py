#!/usr/bin/env python3
"""
SAM - Secret Agent Man - AI agent
Enhanced with full Model Context Protocol (MCP) support
"""

import json
import hashlib
import logging
import time
import traceback
import re
import inspect
import asyncio
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Tuple, NamedTuple
from enum import Enum
from datetime import datetime, UTC
import threading

# Import configuration
from config import SAMConfig
from websocket_broadcaster import init_broadcaster, get_broadcaster
from context_aware_limits import ContextAwareLimitCalculator

if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set up logging (will be configured after loading config)
logger = logging.getLogger("SAMAgent")

# Around line 20-30, with the other imports
try:
    from elasticsearch_memory import ElasticSearchMemoryManager
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    logger.warning("ElasticSearch memory not available - install elasticsearch package")

# Optional imports with availability flags
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available - some functionality may be limited")

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available - API server functionality disabled")


try:
    from system3_moral_authority import integrate_system3_with_sam, System3MoralAuthority, MoralDecision
    SYSTEM3_AVAILABLE = True
except ImportError:
    SYSTEM3_AVAILABLE = False
    logger.warning("System 3 not available - moral authority disabled")


class InterventionType(Enum):
    TOKEN_LIMIT_BREACH = "token_limit_breach"
    TOOL_LOOP_DETECTED = "tool_loop_detected"
    PROGRESS_STAGNATION = "progress_stagnation"
    HIGH_ERROR_RATE = "high_error_rate"
    PLAN_DRIFT_DETECTED = "plan_drift_detected"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class System1State:
    """Current state metrics for System 1"""
    token_usage_percent: float = 0.0
    consecutive_identical_tools: int = 0
    tools_without_progress: int = 0
    recent_error_rate: float = 0.0
    total_tool_calls: int = 0
    iteration_count: int = 0
    last_tool_calls: List[str] = None
    last_tool_name: Optional[str] = None
    current_plan_id: Optional[str] = None  # Track active plan

    def __post_init__(self):
        if self.last_tool_calls is None:
            self.last_tool_calls = []


class InterventionResult(NamedTuple):
    """Result of a System 2 intervention"""
    success: bool
    action_taken: str
    should_break_execution: bool
    modified_context: bool
    message: str


@dataclass
class ToolExecutionResult:
    """Structured result from tool execution"""
    success: bool  # True if tool executed successfully, False if there was an error
    result: str  # The actual result/output string
    error_message: Optional[str] = None  # Optional error details if success=False


class System2Agent:
    """Metacognitive supervisor for System 1 agent"""

    def __init__(self, system1_agent):
        self.system1 = system1_agent

        # System 2's context limit is larger than System 1's
        self.system2_context_limit = system1_agent.system2_context_limit

        # System 2 maintains NO PERSISTENT CONVERSATION HISTORY
        # Only minimal stats are tracked
        self.intervention_stats = {
            'total': 0,
            'types': {},
            'last_intervention_time': None
        }

        # Load System 2 configuration from config
        system2_config = system1_agent.raw_config.get('system2', {})

        # Thresholds for intervention
        self.token_threshold = system2_config.get('token_threshold', 0.60)  # 60% of context limit (reduced from 75%)
        self.consecutive_tool_threshold = system2_config.get('consecutive_tool_threshold', 6)
        self.stagnation_threshold = system2_config.get('stagnation_threshold', 8)
        self.error_rate_threshold = system2_config.get('error_rate_threshold', 0.4)
        self.proactive_prune_threshold = system2_config.get('proactive_rotation_threshold', 0.50)  # 50% - proactive cleanup (reduced from 60%)
        self.plan_drift_check_interval = system2_config.get('plan_drift_check_interval', 5)  # Check every N iterations

        # Plan tracking for drift detection
        self.current_plan = None  # The active user goal/plan
        self.plan_start_index = 0  # Conversation index where current plan started

        # NOTE: With ElasticSearch, System 2's role is:
        # 1. Behavioral interventions (loops, stagnation, errors)
        # 2. Short-term context PRUNING (not summarization)
        # 3. Memory is permanently preserved in ES, so aggressive pruning is safe
        # 4. Plan tracking and drift detection to keep System 1 on task

        logger.info("System 2 initialized (behavioral monitoring + plan tracking + context pruning - full memory in ES)")

    def should_intervene(self, system1_state: System1State) -> Tuple[bool, str]:
        """Determine if System 2 intervention is needed"""
        reasons = []

        # Proactive context rotation - early cleanup before hitting limit
        if system1_state.token_usage_percent > self.proactive_prune_threshold:
            # Count tool result messages to see if they're accumulating
            tool_result_count = sum(1 for msg in self.system1.conversation_history 
                                   if 'Here are the results' in msg.get('content', '') or
                                      '📊 RAW RESULTS:' in msg.get('content', ''))
            if tool_result_count >= 2:  # Reduced from 3 - more aggressive proactive cleanup
                reasons.append("proactive_tool_result_cleanup")
        
        # NEW: Detect memory retrieval loops (get_recent_memories being called repeatedly)
        # FIXED: Only count actual tool executions, not text mentions
        recent_tool_calls = []
        for msg in self.system1.conversation_history[-10:]:  # Check last 10 messages
            # Only count messages that are TOOL RESULTS (have metadata.tools_used)
            if msg.get('role') == 'user' and 'Tool execution results:' in msg.get('content', ''):
                tools_used = msg.get('metadata', {}).get('tools_used', [])
                for tool in tools_used:
                    if tool in ['get_recent_memories', 'get_system_info']:
                        recent_tool_calls.append(tool)
        
        # If we see 4+ memory/system info EXECUTIONS in recent history, that's a loop
        if recent_tool_calls.count('get_recent_memories') >= 4 or recent_tool_calls.count('get_system_info') >= 4:
            reasons.append("memory_retrieval_loop")
            logger.warning(f"⚠️ Memory retrieval loop detected: {recent_tool_calls}")

        # Token usage check
        if system1_state.token_usage_percent > self.token_threshold:
            reasons.append(InterventionType.TOKEN_LIMIT_BREACH.value)

        # Loop detection
        if system1_state.consecutive_identical_tools >= self.consecutive_tool_threshold:
            reasons.append(InterventionType.TOOL_LOOP_DETECTED.value)

        # Stagnation check
        if system1_state.tools_without_progress >= self.stagnation_threshold:
            reasons.append(InterventionType.PROGRESS_STAGNATION.value)

        # Error rate check
        if system1_state.recent_error_rate > self.error_rate_threshold:
            reasons.append(InterventionType.HIGH_ERROR_RATE.value)

        # Plan drift detection - check periodically if we have an active plan
        if (self.current_plan and 
            system1_state.iteration_count > 0 and 
            system1_state.iteration_count % self.plan_drift_check_interval == 0):
            if self._detect_plan_drift(system1_state):
                reasons.append(InterventionType.PLAN_DRIFT_DETECTED.value)

        return len(reasons) > 0, ", ".join(reasons)

    def intervene(self, intervention_types: str, system1_state: System1State) -> InterventionResult:
        """Perform metacognitive intervention - STATELESS"""
        intervention_time = time.time()
        intervention_list = intervention_types.split(", ")

        logger.info(f"🧠 System 2 intervention triggered: {intervention_types}")

        actions_taken = []
        context_modified = False
        should_break = False

        # Handle each intervention type
        for intervention_type in intervention_list:
            if intervention_type == InterventionType.TOKEN_LIMIT_BREACH.value:
                result = self._handle_token_limit_breach()
                actions_taken.append("context_compression")
                context_modified = True

            elif intervention_type == InterventionType.TOOL_LOOP_DETECTED.value:
                result = self._handle_tool_loop(system1_state)
                actions_taken.append("loop_breaking")
                should_break = True

            elif intervention_type == InterventionType.PROGRESS_STAGNATION.value:
                result = self._handle_stagnation(system1_state)
                actions_taken.append("approach_change")

            elif intervention_type == InterventionType.HIGH_ERROR_RATE.value:
                result = self._handle_high_errors(system1_state)
                actions_taken.append("error_mitigation")

            elif intervention_type == InterventionType.PLAN_DRIFT_DETECTED.value:
                result = self._handle_plan_drift(system1_state)
                actions_taken.append("plan_redirection")
            
            elif intervention_type == "memory_retrieval_loop":
                # Handle memory retrieval loops by aggressive cleanup
                result = self._handle_token_limit_breach()  # Use same cleanup logic
                actions_taken.append("memory_loop_cleanup")
                context_modified = True
                logger.info("🧠 Breaking memory retrieval loop with aggressive cleanup")
            
            elif intervention_type == "proactive_tool_result_cleanup":
                # Proactive cleanup of accumulated tool results
                result = self._handle_token_limit_breach()
                actions_taken.append("proactive_cleanup")
                context_modified = True

        # Update STATS only, not full history
        self.intervention_stats['total'] += 1
        self.intervention_stats['last_intervention_time'] = intervention_time

        for itype in intervention_list:
            self.intervention_stats['types'][itype] = self.intervention_stats['types'].get(itype, 0) + 1

        message = f"System 2 intervention: {', '.join(actions_taken)}"
        
        # Broadcast System2 intervention
        if self.system1.broadcaster:
            self.system1.broadcaster.system2_intervention(
                intervention_type=intervention_types,
                action=", ".join(actions_taken),
                details={
                    "should_break": should_break,
                    "context_modified": context_modified,
                    "message": message
                }
            )

        return InterventionResult(
            success=True,
            action_taken=", ".join(actions_taken),
            should_break_execution=should_break,
            modified_context=context_modified,
            message=message
        )

    def _handle_token_limit_breach(self) -> bool:
        """Handle context token limit breach with intelligent System 2 analysis"""
        original_tokens = sum(self.system1._estimate_token_count(msg.get('content', ''))
                              for msg in self.system1.conversation_history)
        
        context_limit = self.system1.context_limit
        usage_percent = (original_tokens / context_limit) * 100 if context_limit > 0 else 0
        
        # ONLY do cleanup if we're actually at high usage (>50%)
        if usage_percent < 50:
            logger.debug(f"📊 Context usage is only {usage_percent:.1f}% - skipping cleanup")
            return True  # Return success without doing anything

        print(f"🧠 SYSTEM 2: Context usage at {usage_percent:.1f}% ({original_tokens:,}/{context_limit:,} tokens) - performing intelligent cleanup...")

        original_length = len(self.system1.conversation_history)

        # Try intelligent context management first, with fallback to deterministic
        success = self._deterministic_context_cleanup()

        if success:
            new_tokens = sum(self.system1._estimate_token_count(msg.get('content', ''))
                             for msg in self.system1.conversation_history)
            new_length = len(self.system1.conversation_history)

            print(
                f"✅ Context optimized: {original_tokens:,} → {new_tokens:,} tokens ({original_length} → {new_length} messages)")
            logger.info(f"🧠 Intelligently managed context: {original_length} → {new_length} messages")
            return True
        else:
            # Fallback to simple compression if intelligent management fails
            print("⚠️ Intelligent management failed, using fallback compression...")
            return self._fallback_naive_compression()

    def _intelligent_context_management(self) -> bool:
        """Use System 2 with LLM and tools to intelligently manage context
        
        NOTE: System 2's context is EPHEMERAL - created fresh for each intervention
        and automatically discarded when complete. This prevents System 2's own 
        context from growing unboundedly.
        """
        try:
            logger.info("🧠 System 2: Starting ephemeral intervention context")
            
            # Get current plan from conversation
            current_plan = self._extract_current_plan()
            
            system1_tokens = sum(self.system1._estimate_token_count(msg.get('content', '')) 
                                for msg in self.system1.conversation_history)
            
            # Build System 2 system message with tool calling instructions
            system2_system_msg = """You are System 2, the metacognitive supervisor for System 1.

CRITICAL TOOL USAGE INSTRUCTIONS:
- To use a tool, you MUST respond with ACTUAL JSON code blocks like this:
  ```json
  {"name": "tool_name", "arguments": {"arg1": "value1"}}
  ```
- DO NOT just SAY you will use a tool - you must OUTPUT the JSON code block
- Use tools whenever needed to analyze and optimize the context
- You can add brief explanation text before the JSON block
- For multiple tools, use separate JSON objects in separate code blocks

EXAMPLE OF CORRECT TOOL USAGE:
I'll read message 5 to check its content.
```json
{"name": "read_message_content", "arguments": {"message_index": 5}}
```

AVAILABLE SYSTEM 2 TOOLS:
- extract_current_plan(): Get the current System 1 plan/goal
- delete_messages(message_indices: List[int]): Delete specific messages by their indices
- read_message_content(message_index: int): Read full content of a specific message"""

            # Build System 2 prompt for context management
            system2_prompt = f"""System 1's conversation history has reached the token limit ({self.token_threshold * 100:.0f}% of maximum).

IMPORTANT CONTEXT ARCHITECTURE:
- System 1 context limit: {self.system1.context_limit:,} tokens (currently at {system1_tokens:,})
- System 2 context limit: {self.system2_context_limit:,} tokens (you have more headroom to analyze)
- You receive COMPRESSED overviews (100 chars per message) to fit in your context
- Use 'read_message_content(index)' if you need full content of specific messages

Your task is to intelligently manage the conversation context by:
1. Understanding the current plan/goal
2. Identifying unnecessary information (redundant tool results, outdated info, etc.)
3. PRUNING messages that are no longer needed (they're preserved in ElasticSearch)
4. Preserving critical information needed for the current plan

IMPORTANT: All messages are automatically saved to ElasticSearch with full fidelity.
You're managing SHORT-TERM WORKING MEMORY only. Be aggressive with deletion - nothing is lost!

TOOL RESULT ARCHIVING: When you delete messages containing tool results, they are automatically 
archived to ElasticSearch with the memory_type "tool_result_archive". System 1 can retrieve them 
later using the retrieve_archived_tool_results tool, so don't hesitate to aggressively clean up 
old tool results - they remain accessible via archival storage.

CURRENT PLAN/GOAL:
{current_plan}

CONVERSATION SUMMARY:
- Total messages: {len(self.system1.conversation_history)}
- Current token usage: ~{sum(self.system1._estimate_token_count(msg.get('content', '')) for msg in self.system1.conversation_history):,} tokens

STRATEGY (PRUNE, DON'T SUMMARIZE):
1. Keep the system message (index 0) - NEVER delete it
2. Keep recent messages (last 6-8) - these contain current context
3. For older messages, aggressively DELETE:
   - Tool result messages ("Here are the results", "📊 RAW RESULTS:") - prioritize these!
   - Redundant tool results (file reads, web fetches that are no longer relevant)
   - Failed tool attempts and error messages
   - Outdated information superseded by newer context
   - Verbose outputs that served their purpose
   - Intermediate steps in completed tasks
4. TARGET: Reduce to ~40-50% of context limit for breathing room
5. REMEMBER: Full history is preserved in ElasticSearch for semantic retrieval!
6. FOCUS ON TOOL RESULTS: These are the biggest context consumers - remove old ones first!

Analyze the conversation and make tool calls to optimize context. Be aggressive but preserve critical information for the current plan.
Remember: OUTPUT JSON code blocks for tool calls!"""

            # Build messages for System 2
            system2_messages = [
                {"role": "system", "content": system2_system_msg},
                {"role": "user", "content": system2_prompt}
            ]

            # Add conversation overview
            overview = self._build_conversation_overview()
            system2_messages.append({
                "role": "user",
                "content": f"DETAILED CONVERSATION STRUCTURE:\n{overview}\n\nNow use your tools to optimize this context. Start by analyzing what can be safely removed or summarized."
            })

            # Make System 2 LLM call with tools
            max_iterations = 5
            for iteration in range(max_iterations):
                response = self._call_system2_llm(system2_messages)
                
                if not response:
                    logger.error("System 2 LLM call failed")
                    return False
                
                # CRITICAL: Inject timestamps so System 2 tool calls can be extracted
                response = self.system1._inject_timestamps_in_response(response)

                # Parse and execute tool calls
                tool_calls = self._extract_tool_calls_from_response(response)
                
                if not tool_calls:
                    # No more tool calls - System 2 is done
                    logger.info(f"System 2 completed context management in {iteration + 1} iterations")
                    break

                # Execute System 2 tools
                results = []
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('arguments', {})
                    
                    result = self._execute_system2_tool(tool_name, tool_args)
                    results.append(f"Tool '{tool_name}' result: {result}")

                # Add results to System 2 conversation
                system2_messages.append({"role": "assistant", "content": response})
                system2_messages.append({
                    "role": "user",
                    "content": f"Tool results:\n" + "\n\n".join(results) + "\n\nContinue optimizing if needed, or respond with your analysis if done."
                })

            # Log System 2 context usage before discarding
            system2_tokens_used = sum(self.system1._estimate_token_count(msg.get('content', '')) 
                                     for msg in system2_messages)
            logger.info(f"🧠 System 2: Intervention complete. Used {system2_tokens_used:,}/{self.system2_context_limit:,} tokens. Discarding ephemeral context.")
            
            # system2_messages will be garbage collected when this function returns
            # No persistent System 2 conversation history is maintained
            return True

        except Exception as e:
            logger.error(f"Intelligent context management error: {e}")
            logger.info("🧠 System 2: Discarding ephemeral context after error")
            return False

    def _extract_current_plan(self) -> str:
        """Extract current plan from System 1 conversation"""
        # Look for recent user requests and System 1 acknowledgments
        plan_keywords = ['plan', 'task', 'goal', 'working on', 'need to', 'trying to']
        
        for i in range(len(self.system1.conversation_history) - 1, max(0, len(self.system1.conversation_history) - 15), -1):
            msg = self.system1.conversation_history[i]
            content = msg.get('content', '').lower()
            
            if msg.get('role') == 'user' and not content.startswith('here are the results'):
                return f"Most recent user request: {msg.get('content', '')[:300]}"
            
            for keyword in plan_keywords:
                if keyword in content:
                    return msg.get('content', '')[:300]
        
        return "No clear plan identified"

    def _build_conversation_overview(self) -> str:
        """Build a structured overview of the conversation for System 2"""
        overview_lines = []
        
        for i, msg in enumerate(self.system1.conversation_history):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            # Truncate content for overview
            content_preview = content[:100].replace('\n', ' ')
            if len(content) > 100:
                content_preview += "..."
            
            token_estimate = len(content) // 4
            
            overview_lines.append(f"[{i}] {role} (~{token_estimate} tokens): {content_preview}")
        
        return "\n".join(overview_lines)

    def _call_system2_llm(self, messages: List[Dict]) -> str:
        """Make LLM API call for System 2 with larger context window"""
        try:
            # Calculate current System 2 token usage
            system2_tokens = sum(self.system1._estimate_token_count(msg.get('content', '')) 
                                for msg in messages)
            
            if system2_tokens > self.system2_context_limit:
                logger.error(f"System 2 context overflow: {system2_tokens:,} > {self.system2_context_limit:,}")
                return None
            
            # Use System 1's LLM infrastructure but with System 2's larger limit
            provider = self.system1.provider.lower()
            
            # System 2 gets more tokens for its response (3000 vs 2000)
            # Enable streaming for System 2 so users can see what it's thinking
            print("\n🧠 System 2 (Metacognitive Supervisor): ", end='', flush=True)
            if provider == 'lmstudio':
                return self.system1._generate_lmstudio_completion(messages, temperature=0.2, max_tokens=3000, stream=True, _skip_label=True)
            elif provider == 'claude':
                return self.system1._generate_claude_completion(messages, temperature=0.2, max_tokens=3000, stream=True, _skip_label=True)
            else:
                logger.error(f"Unsupported provider for System 2: {provider}")
                return None
                
        except Exception as e:
            logger.error(f"System 2 LLM call error: {e}")
            return None

    def _extract_tool_calls_from_response(self, response: str) -> List[Dict]:
        """Extract tool calls from System 2 LLM response with memory context filtering"""
        tool_calls = []
        
        # Look for JSON blocks
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        
        for match_obj in re.finditer(json_pattern, response, re.DOTALL):
            try:
                # Check context before JSON block to detect archived/historical content
                match_start = match_obj.start()
                context_before = response[max(0, match_start - 200):match_start].lower()
                
                # Skip if from memory/archive context (must be clearly historical, not about future actions)
                skip_indicators = [
                    'from memory:', 'from elasticsearch:', 'elasticsearch returned',
                    'previously executed', 'was executed', 'historical tool call',
                    'retrieved from memory', 'stored tool call', 'found in memory',
                    'memory shows', 'according to memory', 'search result shows:',
                    'example of', 'for example, you', 'you previously',
                    '📊 raw results:', 'raw results:', 'tool returned:',
                    'content\': \'', 'memory_type\': \''
                ]
                
                if any(indicator in context_before for indicator in skip_indicators):
                    logger.debug(f"System 2: Skipping JSON block from memory context")
                    continue
                
                match_text = match_obj.group(1)
                
                # Check for archive markers in JSON
                if '"_archive_marker"' in match_text or '"_historical"' in match_text:
                    continue
                
                tool_call = json.loads(match_text)
                if 'name' in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        return tool_calls

    def _execute_system2_tool(self, tool_name: str, arguments: Dict) -> str:
        """Execute a System 2 exclusive tool"""
        try:
            # System 2 tools are registered on System 1's registry
            if tool_name not in self.system1.system2_tools:
                return f"Error: Tool '{tool_name}' not found in System 2 registry"
            
            tool_info = self.system1.system2_tools[tool_name]
            tool_func = tool_info['function']
            
            # Call the tool function
            result = tool_func(**arguments)
            return str(result)
            
        except Exception as e:
            logger.error(f"System 2 tool execution error: {e}")
            return f"Error executing System 2 tool '{tool_name}': {str(e)}"

    def _fallback_naive_compression(self) -> bool:
        """Fallback to naive compression if intelligent management fails"""
        original_length = len(self.system1.conversation_history)

        # Keep system message and last few exchanges
        if original_length > 5:
            system_msg = self.system1.conversation_history[0]
            recent_msgs = self.system1.conversation_history[-4:]  # Last 4 messages

            # Create summary of middle content
            middle_content = self.system1.conversation_history[1:-4]
            if middle_content:
                summary = self._compress_conversation_segment(middle_content)
                summary_msg = {
                    "role": "system",
                    "content": f"[CONTEXT SUMMARY] Previous conversation included: {summary}"
                }

                # Rebuild conversation with compression
                self.system1.conversation_history = [system_msg, summary_msg] + recent_msgs
                
                # Broadcast System2 compression event
                if self.system1.broadcaster:
                    tokens_saved = sum(self.system1._estimate_token_count(msg.get('content', ''))
                                     for msg in middle_content)
                    self.system1.broadcaster.system2_intervention(
                        intervention_type="context_compression",
                        action="fallback_compression",
                        details={
                            "original_count": original_length,
                            "compressed_count": len(self.system1.conversation_history),
                            "tokens_freed": tokens_saved,
                            "summary_created": True
                        }
                    )
                
                return True

        return False

    def _refresh_system_instructions(self) -> None:
        """Refresh system instructions to combat recency bias in long contexts
        
        When context fills up, the system prompt at position 0 gets 'forgotten' due
        to transformer attention decay. This re-injects core instructions closer to
        recent messages to keep them salient.
        """
        try:
            # Extract core system message if it exists
            if len(self.system1.conversation_history) > 0:
                first_msg = self.system1.conversation_history[0]
                if first_msg.get('role') == 'system':
                    # Create refreshed reminder with key points
                    refresh_msg = {
                        "role": "system",
                        "content": """<system_refresh>
🔄 SYSTEM INSTRUCTION REFRESH (Combat Recency Bias)

Key reminders:
1. TOOL USAGE: Output actual JSON code blocks:
   ```json
   {"name": "tool_name", "arguments": {...}}
   ```
   
2. DO NOT just SAY you'll use a tool - OUTPUT the JSON block!

3. Tool results are ALREADY shown to user - don't repeat them verbatim
   Instead: Analyze, summarize, provide insights

4. System 2 monitors for loops - vary your approach if intervened

5. You have access to memory (ElasticSearch) for context beyond this window

6. Be concise and action-oriented
</system_refresh>"""
                    }
                    
                    # Add refresh message near the end (before last 3 messages)
                    insert_position = max(1, len(self.system1.conversation_history) - 3)
                    self.system1.conversation_history.insert(insert_position, refresh_msg)
                    
                    logger.info(f"🔄 System prompt refreshed at position {insert_position} to combat recency bias")
                    print("🔄 System instructions refreshed (combat attention decay)")
                    
        except Exception as e:
            logger.error(f"Error refreshing system instructions: {e}")

    def _deterministic_context_cleanup(self) -> bool:
        """Fast, deterministic context cleanup without LLM involvement
        
        This is more reliable than LLM-based cleanup and completes instantly.
        Strategy:
        1. Keep system messages and recent context
        2. Aggressively remove old tool results (they're in ElasticSearch)
        3. Remove duplicate/redundant information
        4. Target: 40-50% of context limit
        """
        try:
            if len(self.system1.conversation_history) <= 5:
                return True  # Too small to optimize
            
            # Archive everything to ElasticSearch first
            if hasattr(self.system1, 'memory_manager') and self.system1.memory_manager:
                try:
                    for msg in self.system1.conversation_history:
                        if msg.get('role') != 'system' or 'CONTEXT SUMMARY' in msg.get('content', ''):
                            self.system1.memory_manager.store_memory(
                                content=msg.get('content', ''),
                                memory_type='conversation_archive',
                                metadata={
                                    'role': msg.get('role'),
                                    'timestamp': time.time()
                                }
                            )
                except Exception as e:
                    logger.warning(f"Archiving to ElasticSearch failed: {e}")
            
            # Identify messages to keep
            keep_indices = set()
            keep_indices.add(0)  # Always keep system message
            
            # Keep last 4 messages (reduced from 6 for more aggressive cleanup)
            recent_count = min(4, len(self.system1.conversation_history))
            for i in range(len(self.system1.conversation_history) - recent_count, 
                          len(self.system1.conversation_history)):
                keep_indices.add(i)
            
            # Identify tool result messages to remove (they're the biggest)
            tool_result_indicators = [
                '📊 RAW RESULTS:',
                'Here are the results',
                '============================================================',
                '⚠️ Step',
                '📋 Execution Summary:',
                'Tool returned:',
                '✅ Completed:',
                'completed successfully',
                'get_recent_memories',  # Specifically target memory loops
                '[{\'content\':',  # Memory retrieval results
                'memory_type\':',  # Memory objects
            ]
            
            # Score messages for removal priority
            removal_candidates = []
            for i, msg in enumerate(self.system1.conversation_history):
                if i in keep_indices:
                    continue
                    
                content = msg.get('content', '')
                tokens = self.system1._estimate_token_count(content)
                
                # Calculate removal score (higher = more likely to remove)
                score = 0
                
                # Prioritize removing tool results (especially memory retrievals)
                if any(indicator in content for indicator in tool_result_indicators):
                    score += 100
                
                # Extra penalty for memory retrieval loops
                if 'get_recent_memories' in content or 'get_system_info' in content:
                    score += 150  # Very high priority to remove these
                
                # Prioritize large messages even more aggressively
                if tokens > 500:
                    score += 75  # Increased from 50
                if tokens > 1000:
                    score += 150  # Increased from 100
                if tokens > 2000:
                    score += 200  # NEW: Very large messages get removed first
                
                # Deprioritize user messages (keep conversation flow)
                if msg.get('role') == 'user':
                    score -= 30
                
                # Prioritize removing assistant tool call results
                if msg.get('role') == 'assistant' and '```json' in content:
                    score += 30
                
                removal_candidates.append((i, score, tokens))
            
            # Sort by score (highest first) and remove until we hit target
            removal_candidates.sort(key=lambda x: x[1], reverse=True)
            
            target_tokens = int(self.system1.short_term_context_tokens * 0.40)  # 40% target (more aggressive, reduced from 45%)
            current_tokens = sum(self.system1._estimate_token_count(msg.get('content', ''))
                                for msg in self.system1.conversation_history)
            
            messages_to_remove = set()
            tokens_freed = 0
            
            for idx, score, tokens in removal_candidates:
                if current_tokens - tokens_freed <= target_tokens:
                    break
                messages_to_remove.add(idx)
                tokens_freed += tokens
            
            # Remove messages (in reverse order to preserve indices)
            new_history = []
            removed_count = 0
            for i, msg in enumerate(self.system1.conversation_history):
                if i not in messages_to_remove:
                    new_history.append(msg)
                else:
                    removed_count += 1
            
            self.system1.conversation_history = new_history
            
            logger.info(f"🧠 System 2: Deterministic cleanup removed {removed_count} messages, freed ~{tokens_freed:,} tokens")
            print(f"🧠 Removed {removed_count} old messages (prioritized tool results)")
            print(f"💾 Freed ~{tokens_freed:,} tokens | All messages archived to ElasticSearch")
            
            # Broadcast System2 cleanup event
            if self.system1.broadcaster:
                self.system1.broadcaster.system2_intervention(
                    intervention_type="context_cleanup",
                    action="deterministic_cleanup",
                    details={
                        "removed_count": removed_count,
                        "tokens_freed": tokens_freed,
                        "target_tokens": target_tokens,
                        "archived_to_elasticsearch": True
                    }
                )
            
            # Refresh system instructions after cleanup to ensure they're salient
            self._refresh_system_instructions()
            
            return True
            
        except Exception as e:
            logger.error(f"Deterministic cleanup error: {e}")
            traceback.print_exc()
            return False

    def _handle_tool_loop(self, state: System1State) -> bool:
        """Handle detected tool execution loop with context and suggestions"""
        logger.info(f"🧠 System 2: Breaking tool loop (last tool repeated {state.consecutive_identical_tools} times)")

        # Set a cooldown period to prevent immediate retry
        import time
        self._loop_cooldown_until = time.time() + 30  # 30 second cooldown

        # Generate context-aware suggestions based on the looping tool
        recent_tools = state.last_tool_calls[-10:] if state.last_tool_calls else []
        recent_summary = ", ".join(recent_tools[-5:]) if recent_tools else "none"
        
        # Provide tool-specific suggestions
        suggestions = self._generate_loop_suggestions(state.last_tool_name, state)
        
        # Inject enhanced loop-breaking guidance with full context
        from datetime import datetime
        current_time = datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
        
        loop_breaking_msg = {
            "role": "system",
            "content": f"""<metacognitive_intervention>
🧠 SYSTEM 2 INTERVENTION: Tool Execution Loop Detected
⏰ Current Time: {current_time}

📊 CURRENT SITUATION:
   • Looping tool: '{state.last_tool_name}'
   • Consecutive uses: {state.consecutive_identical_tools} times
   • Recent tool sequence: {recent_summary}
   • Total tools executed: {state.total_tool_calls}

⚠️ ANALYSIS:
   You are repeating the same action without making progress. This indicates one of:
   1. The tool isn't producing useful results
   2. You're not processing the results effectively
   3. You may be stuck in a cognitive loop
   4. The approach may not be suitable for this task

💡 SUGGESTED ALTERNATIVES:
{suggestions}

🛑 COOLDOWN ACTIVE:
   • 30-second pause before next action
   • Tool '{state.last_tool_name}' is temporarily discouraged
   • When you resume, you MUST try a different approach
   • Reflect on what you learned from previous attempts
</metacognitive_intervention>"""
        }

        self.system1.conversation_history.append(loop_breaking_msg)
        print(f"🧠 System 2: Loop detected on '{state.last_tool_name}' - providing guidance")
        print(f"💡 Suggestions: {len(suggestions.splitlines())} alternatives offered")
        return True

    def _generate_loop_suggestions(self, tool_name: str, state: System1State) -> str:
        """Generate context-aware suggestions based on which tool is looping"""
        suggestions_map = {
            'store_memory': """   1. Try 'retrieve_memory' to check what you've already stored
   2. Use 'search_memory' to find specific information
   3. Try 'list_tools' to see other available capabilities
   4. Use 'execute_code' to perform calculations or get system info
   5. If autonomous, reflect silently without storing""",
            'retrieve_memory': """   1. Use 'search_memory' with specific keywords instead
   2. Try 'store_memory' to save new insights
   3. Use 'list_files' or 'read_file' to explore the codebase
   4. Consider 'web_search' if you need external information
   5. Execute code to gather fresh data""",
            'read_file': """   1. Use 'search_files' to find what you're looking for
   2. Try 'list_files' to see directory contents first
   3. Use 'execute_code' to analyze file programmatically
   4. Consider 'grep_search' for pattern-based search
   5. Summarize findings and move to next task""",
            'search_files': """   1. Try 'read_file' on a specific file you found
   2. Use 'list_files' to browse directories
   3. Narrow your search with more specific terms
   4. Use 'execute_code' to programmatically analyze results
   5. Consider if the file/info exists at all""",
            'execute_code': """   1. Check if code execution errors suggest a different approach
   2. Use 'read_file' to examine relevant code first
   3. Try 'store_memory' to save your findings
   4. Break the task into smaller steps
   5. Consider using other tools to gather info first""",
            'web_search': """   1. Refine search terms - you may be too broad/narrow
   2. Use 'retrieve_memory' to check if you already have this info
   3. Try 'read_file' for local documentation
   4. Consider if external search is really needed
   5. Store findings and move to implementation"""
        }
        
        # Return specific suggestions or generic ones
        if tool_name in suggestions_map:
            return suggestions_map[tool_name]
        else:
            return f"""   1. Review your goal - is '{tool_name}' the right approach?
   2. Try a fundamentally different tool category
   3. Break the task into smaller subtasks
   4. Ask yourself: what information do I actually need?
   5. Consider if you've already achieved the goal"""

    def _handle_stagnation(self, state: System1State) -> bool:
        """Handle progress stagnation with detailed analysis and suggestions"""
        from datetime import datetime
        current_time = datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
        recent_tools = state.last_tool_calls[-10:] if state.last_tool_calls else []
        recent_summary = ", ".join(recent_tools[-5:]) if recent_tools else "none"
        
        print(f"🧠 System 2: Progress stagnation detected ({state.tools_without_progress} tools)")
        print(f"💡 Recent actions: {recent_summary}")

        logger.info(f"🧠 System 2: Addressing stagnation ({state.tools_without_progress} tools without progress)")

        guidance_msg = {
            "role": "system",
            "content": f"""<metacognitive_guidance>
🧠 SYSTEM 2 OBSERVATION: Progress Stagnation Detected
⏰ Current Time: {current_time}

📊 CURRENT SITUATION:
   • Tools executed: {state.tools_without_progress} without clear progress
   • Recent actions: {recent_summary}
   • Total tool calls: {state.total_tool_calls}
   • Execution efficiency: May be spinning wheels

🔍 ANALYSIS:
   You've been executing tools but may not be advancing toward the goal. This suggests:
   1. The current approach isn't yielding useful results
   2. You may need to step back and reassess the goal
   3. Missing information or context might be blocking progress
   4. You might be overthinking a simple task

💡 RECOMMENDATIONS:
   1. **Pause and reflect**: What exactly are you trying to achieve?
   2. **Ask for help**: If the user is present, ask for clarification
   3. **Summarize findings**: State what you've learned so far
   4. **Change approach**: Try a completely different strategy
   5. **Simplify**: Break the goal into smaller, achievable steps
   6. **Check assumptions**: Are you solving the right problem?

✅ NEXT STEPS:
   Take a moment to reflect, then either ask the user for guidance or pivot to a
   fundamentally different approach. Avoid repeating what hasn't worked.
</metacognitive_guidance>"""
        }

        self.system1.conversation_history.append(guidance_msg)
        return True

    def _handle_high_errors(self, state: System1State) -> bool:
        """Handle high error rate with diagnostic info and suggestions"""
        from datetime import datetime
        current_time = datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
        recent_tools = state.last_tool_calls[-5:] if state.last_tool_calls else []
        recent_summary = ", ".join(recent_tools) if recent_tools else "none"
        
        print(f"🧠 System 2: High error rate detected ({state.recent_error_rate:.1%})")
        print(f"💡 Recent failed tools: {recent_summary}")

        logger.info(f"🧠 System 2: Mitigating high error rate ({state.recent_error_rate:.1%})")

        error_msg = {
            "role": "system",
            "content": f"""<metacognitive_guidance>
🧠 SYSTEM 2 ALERT: High Error Rate Detected
⏰ Current Time: {current_time}

📊 CURRENT SITUATION:
   • Error rate: {state.recent_error_rate:.1%}
   • Recent tools: {recent_summary}
   • Total attempts: {state.total_tool_calls}

⚠️ DIAGNOSIS:
   Many of your recent tool executions are failing. This usually means:
   1. You're using tools with incorrect parameters
   2. Required resources/files don't exist
   3. The approach is fundamentally flawed
   4. Environment/permissions issues
   5. You're trying to do something impossible

💡 CORRECTIVE ACTIONS:
   1. **Validate inputs**: Check that files, paths, or data exist first
   2. **Simplify**: Use basic, reliable tools (read_file, list_files, execute_code)
   3. **Error analysis**: Read error messages carefully - they contain clues
   4. **Test assumptions**: Verify your understanding of what's available
   5. **Break it down**: Try smaller, simpler operations first
   6. **Alternative path**: Consider if there's a different way entirely

🎯 IMMEDIATE RECOMMENDATION:
   Switch to defensive, validation-first approach. Check preconditions before
   attempting complex operations. If uncertain, ask the user for guidance.
</metacognitive_guidance>"""
        }

        self.system1.conversation_history.append(error_msg)
        return True

    def _detect_plan_drift(self, state: System1State) -> bool:
        """Detect if System 1's actions have drifted from the original plan"""
        if not self.current_plan:
            return False

        # Look at recent tool calls (last 5) to see if they align with plan
        recent_tools = state.last_tool_calls[-5:] if len(state.last_tool_calls) >= 5 else state.last_tool_calls
        
        # If System 1 has made at least 5 tool calls since plan started
        if len(recent_tools) < 3:
            return False

        # Simple heuristic: if there's high repetition or stagnation, might indicate drift
        # Also check if conversation has grown significantly since plan start
        conversation_growth = len(self.system1.conversation_history) - self.plan_start_index
        
        # If we've had many iterations without clear progress toward plan
        if conversation_growth > 15 and state.tools_without_progress > 4:
            logger.info(f"🧠 System 2: Potential plan drift detected - {conversation_growth} messages since plan start, {state.tools_without_progress} tools without progress")
            return True

        return False

    def _handle_plan_drift(self, state: System1State) -> bool:
        """Redirect System 1 when it drifts from the plan"""
        from datetime import datetime
        current_time = datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
        
        print(f"🧠 SYSTEM 2: Plan drift detected - redirecting System 1 to original goal")
        
        logger.info(f"🧠 System 2: Plan drift intervention - refocusing on: {self.current_plan[:100]}")

        # Get a summary of recent actions
        recent_actions = state.last_tool_calls[-5:] if len(state.last_tool_calls) >= 5 else state.last_tool_calls
        actions_summary = ", ".join(recent_actions) if recent_actions else "various actions"

        redirect_msg = {
            "role": "system",
            "content": f"<metacognitive_guidance>🧠 SYSTEM 2 PLAN CHECK\n⏰ Current Time: {current_time}\n\nThe original goal was: '{self.current_plan}'. Your last actions were: {actions_summary}. These actions may not be directly contributing to the goal. Please either: 1) Explain how your current approach serves the original goal, or 2) Redirect your efforts back to the original objective. What specific step will move you closer to completing: '{self.current_plan}'?</metacognitive_guidance>"
        }

        self.system1.conversation_history.append(redirect_msg)
        return True

    def _compress_conversation_segment(self, messages: List[Dict]) -> str:
        """Create intelligent summary of conversation segment"""
        # Extract key information from the messages
        user_requests = []
        tool_results = []

        for msg in messages:
            content = msg.get("content", "")
            if msg.get("role") == "user":
                if not content.startswith("Tool ") and not content.startswith("Here are the results"):
                    user_requests.append(content[:100])  # First 100 chars
            elif msg.get("role") == "assistant":
                if "tool" not in content.lower():
                    # This is likely a regular response, not tool usage
                    pass

        summary_parts = []
        if user_requests:
            summary_parts.append(f"User requests: {'; '.join(user_requests)}")

        return ". ".join(summary_parts) if summary_parts else "Various tool executions and exchanges"

    def get_intervention_stats(self) -> Dict[str, Any]:
        """Get statistics about System 2 interventions from minimal stats"""
        return {
            "total_interventions": self.intervention_stats['total'],
            "intervention_types": self.intervention_stats['types'].copy(),
            "last_intervention": self.intervention_stats['last_intervention_time']
        }


# Tool categories
class ToolCategory(Enum):
    UTILITY = "utility"
    DEVELOPMENT = "development"
    FILESYSTEM = "filesystem"
    SYSTEM = "system"
    COMMUNICATION = "communication"
    WEB = "web"
    DATA = "data"
    SECURITY = "security"
    MULTIMEDIA = "multimedia"


@dataclass
class ToolInfo:
    """Information about a registered tool"""
    function: Callable
    description: str
    parameters: Dict[str, Any]
    category: ToolCategory
    requires_approval: bool = False
    usage_count: int = 0


def load_all_plugins(sam):
    """Enhanced plugin loading with System 2 support"""
    plugins_dir = Path(__file__).parent / "plugins"
    if not plugins_dir.exists():
        print("⚠️ Plugins directory not found")
        return

    loaded_count = 0
    system2_plugins_count = 0
    print(f"🔍 Scanning for plugins in {plugins_dir}")

    # Load all .py files in plugins directory
    for plugin_file in plugins_dir.glob("*.py"):
        if plugin_file.name.startswith("_"):
            continue  # Skip __init__.py, __pycache__, etc.

        if sam.plugin_manager.load_plugin_from_file(str(plugin_file), sam):
            loaded_count += 1

            # Check if this was a System 2 plugin
            plugin_name = plugin_file.stem
            if plugin_name in sam.plugin_manager.plugins:
                plugin = sam.plugin_manager.plugins[plugin_name]
                if hasattr(plugin, 'restricted') and plugin.restricted:
                    system2_plugins_count += 1

    if loaded_count > 0:
        system1_tools = len(sam.local_tools)
        system2_tools = len(sam.system2_tools)
        total_tools = system1_tools + system2_tools

        print(f"📦 Loaded {loaded_count} plugins, {total_tools} total tools")
        print(f"🤖 System 1 tools: {system1_tools}")
        print(f"🧠 System 2 tools: {system2_tools}")
        if system2_plugins_count > 0:
            print(f"🔒 System 2 plugins: {system2_plugins_count}")

# ===== PLUGIN SYSTEM =====
class SAMPlugin:
    """Base class for SAM plugins"""

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self.enabled = True

    def register_tools(self, agent):
        """Override this method to register tools with the agent"""
        pass

    def on_load(self, agent):
        """Called when plugin is loaded"""
        pass

    def on_unload(self, agent):
        """Called when plugin is unloaded"""
        pass


class PluginManager:
    """Manages SAM plugins"""

    def __init__(self):
        self.plugins: Dict[str, SAMPlugin] = {}

    def load_plugin_from_file(self, plugin_path: str, agent) -> bool:
        """Load a plugin from a Python file"""
        try:
            import importlib.util
            plugin_path_obj = Path(plugin_path)

            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_path_obj.stem] = module
            spec.loader.exec_module(module)

            # First try factory function
            if hasattr(module, 'create_plugin') and callable(getattr(module, 'create_plugin')):
                try:
                    plugin = module.create_plugin()
                    return self.register_plugin(plugin, agent)
                except Exception as e:
                    logger.error(f"Error calling create_plugin() in {plugin_path}: {str(e)}")
                    return False

            # Look for plugin class
            plugin_class = None
            for item_name in dir(module):
                item = getattr(module, item_name)
                if (isinstance(item, type) and
                        issubclass(item, SAMPlugin) and
                        item != SAMPlugin):
                    plugin_class = item
                    break

            if not plugin_class:
                logger.error(f"No SAMPlugin subclass found in {plugin_path}")
                return False

            # Instantiate and register
            plugin = plugin_class()
            return self.register_plugin(plugin, agent)

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_path}: {str(e)}")
            return False

    def register_plugin(self, plugin: SAMPlugin, agent) -> bool:
        """Register a plugin instance"""
        try:
            if plugin.name in self.plugins:
                logger.warning(f"Plugin {plugin.name} already loaded, replacing...")

            self.plugins[plugin.name] = plugin
            plugin.on_load(agent)
            plugin.register_tools(agent)

            logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
            return True

        except Exception as e:
            logger.error(f"Error registering plugin {plugin.name}: {str(e)}")
            return False

    def unload_plugin(self, plugin_name: str):
        """Unload a plugin"""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")


# ===== API MODELS (if FastAPI available) =====
if FASTAPI_AVAILABLE:
    class QueryRequest(BaseModel):
        message: str
        max_iterations: int = 5
        verbose: bool = False
        auto_approve: Optional[bool] = None
        session_id: Optional[str] = None


    class QueryResponse(BaseModel):
        response: str
        session_id: str
        status: str
        timestamp: str
        error: Optional[str] = None


    class ToolExecutionRequest(BaseModel):
        tool_name: str
        arguments: Dict[str, Any]


    class ToolExecutionResponse(BaseModel):
        success: bool
        result: Optional[str] = None
        error: Optional[str] = None
        execution_time: float
        tool_name: str


    class HealthResponse(BaseModel):
        status: str
        model: str
        tools_count: int
        plugins_count: int
        uptime_seconds: float


# ===== MAIN SAM AGENT CLASS =====
class SAMAgent:
    """Semi-Autonomous Model AI Agent with MCP support"""

    def __init__(self, model_name: str = None, context_limit: int = None, safety_mode: bool = True,
                 auto_approve: bool = False, connect_mcp_on_startup: bool = True, skip_memory_loading: bool = False,
                 agent_type: str = 'main'):
        """Initialize SAM Agent
        
        Args:
            skip_memory_loading: If True, skip loading core memories and conversation history.
                                Use for ephemeral agents that should start with clean slate.
            agent_type: Type of agent - 'main', 'lead_researcher', or 'fetcher'.
                       Determines which tools are available.
        """

        # Store agent type for plugin tool filtering
        self.agent_type = agent_type

        # Load configuration FIRST and ensure raw_config is set
        self.config = self._load_config()

        # MCP auto-connection flag - will trigger on first run() call OR during startup if enabled
        self._mcp_auto_connect_pending = True

        logger.info(f"SAM Agent initialized - MCP auto-connect: {self.config.mcp.enabled}")

        # Configure logging based on config
        self._configure_logging()

        # MOVE ALL PROVIDER CONFIGURATION LOGIC HERE (after _load_config completes)
        # Model configuration - use provider-aware logic
        self.provider = self.raw_config.get('provider', 'lmstudio')

        if self.provider == 'claude':
            provider_config = self.raw_config.get('providers', {}).get('claude', {})

            self.base_url = "https://api.anthropic.com/v1"
            self.api_key = provider_config.get('api_key', '')
            self.model_name = model_name or provider_config.get('model_name', 'claude-sonnet-4-20250514')
            self.context_limit = context_limit or provider_config.get('context_limit', 200000)

        else:
            # For LMStudio, use the provider-specific config
            lmstudio_config = self.raw_config.get('providers', {}).get('lmstudio', {})

            self.base_url = self.config.lmstudio.base_url
            self.api_key = lmstudio_config.get('api_key', self.config.lmstudio.api_key)
            self.model_name = model_name or lmstudio_config.get('model_name', 'qwen2.5-coder-14b-instruct')
            self.context_limit = context_limit or lmstudio_config.get('context_limit', 20000)

        # ADD THESE LINES:
        # System 2 gets larger context window (from config or 1.5x System 1's limit)
        system2_config = self.raw_config.get('system2', {})
        context_multiplier = system2_config.get('context_multiplier', 1.5)
        self.system2_context_limit = int(self.context_limit * context_multiplier)

        # Short-term context is where sliding window kicks in (70% of System 1's limit)
        self.short_term_context_tokens = int(self.context_limit * 0.7)

        logger.info(f"System 1 context: {self.context_limit:,} tokens")
        logger.info(f"System 2 context: {self.system2_context_limit:,} tokens")
        logger.info(f"Short-term window: {self.short_term_context_tokens:,} tokens")

        # Agent state
        self.conversation_history = []
        self.short_term_context_tokens = int(self.context_limit * 0.7)
        self.safety_mode = safety_mode
        self.auto_approve = auto_approve
        self.stop_requested = False
        self.stop_message = ""
        self.debug_mode = False  # Debug mode for verbose execution logging
        
        # Reactive planning configuration
        agent_config = self.raw_config.get('agent', {})
        self.reactive_planning = agent_config.get('reactive_planning', False)
        self.max_tools_per_batch = agent_config.get('max_tools_per_batch', 1)
        self.max_iterations = agent_config.get('max_iterations', 5)
        logger.info(f"Reactive planning: {'ENABLED' if self.reactive_planning else 'DISABLED'} "
                   f"(batch size: {self.max_tools_per_batch})")
        logger.info(f"Max iterations per turn: {self.max_iterations}")

        # System 1 tool management
        self.local_tools = {}
        self.tool_info = {}
        self.tools_by_category = {category: [] for category in ToolCategory}

        # ===== NEW: SYSTEM 2 EXCLUSIVE TOOL REGISTRIES =====
        self.system2_tools = {}
        self.system2_tool_info = {}
        self.system2_tools_by_category = {category: [] for category in ToolCategory}

        # ===== NEW: SYSTEM 2 HALT SIGNAL =====
        self.system2_halt_requested = False
        self.system2_halt_reason = ""

        # MCP (Model Context Protocol) support
        self.mcp_sessions = {}
        self.mcp_tools = {}

        # Plugin system
        self.plugin_manager = PluginManager()

        # ===== CONTEXT-AWARE DYNAMIC LIMITS =====
        self.context_limit_calculator = ContextAwareLimitCalculator(self)

        # ===== NEW: SYSTEM 2 INTEGRATION =====
        self.system2 = System2Agent(self)
        self.execution_metrics = {
            "consecutive_tool_count": 0,
            "last_tool_name": None,
            "tool_error_count": 0,
            "total_tool_count": 0,
            "tools_since_progress": 0,
            "last_tool_signature": None,  # ADD THIS LINE
            # NEW: Autonomous mode tracking
            "tools_since_notes": 0,
            "autonomous_mode": False,
            "last_autonomous_prompt": 0,
            # NEW: Periodic System2 wake-up tracking
            "user_prompt_count": 0,  # Total prompts (user + tool results) - used for mid-execution wakeups
            "actual_user_prompts": 0,  # ONLY real user prompts - used for post-response wakeup scheduling
            "system2_wakeup_interval": self.raw_config.get('system2', {}).get('periodic_wakeup_interval', 5)
        }

        # ADD THIS LINE - conversation thread safety
        self._conversation_lock = threading.Lock()

        # ===== NEW: LOAD SAM'S ACCUMULATED MEMORIES =====
        self.persistent_context = []
        self.core_memory_context = None  # Dedicated storage for core memories
        self.core_memory_prompt_counter = 0  # Track when to refresh core memories
        
        # Skip memory loading for ephemeral agents (research agents, browser agents, etc.)
        if not skip_memory_loading:
            self._load_accumulated_memories()
            self._restore_recent_conversation()
        else:
            logger.info("🧹 Ephemeral agent: Skipping memory/history loading (clean slate)")

        logger.info(f"SAM Agent initialized with System 1/System 2 architecture - Model: {self.model_name}")
        logger.info(f"Context limit: {self.context_limit:,} tokens")
        logger.info(f"Safety mode: {'ON' if self.safety_mode else 'OFF'}")

        # Connect to MCP servers during startup if enabled
        if connect_mcp_on_startup:
            self._connect_mcp_on_startup()

        self._auto_enable_system3()
        
        # Initialize WebSocket broadcaster
        ws_config = self.raw_config.get('websocket', {})
        self.broadcaster = init_broadcaster(
            host=ws_config.get('host', 'localhost'),
            port=ws_config.get('port', 8765),
            enabled=ws_config.get('enabled', True)
        )
        if self.broadcaster and self.broadcaster.enabled:
            logger.info(f"📡 WebSocket broadcaster enabled on ws://{ws_config.get('host', 'localhost')}:{ws_config.get('port', 8765)}")

    def _restore_recent_conversation(self):
        """Restore recent conversation from ElasticSearch - using configured target percentage"""
        try:
            if not hasattr(self, 'memory_manager'):
                logger.info("📝 No memory manager available - starting with empty conversation")
                return

            # Use configured startup_context_target from system2 config
            startup_target = self.raw_config.get('system2', {}).get('startup_context_target', 0.25)
            target_tokens = int(self.short_term_context_tokens * startup_target)

            logger.info(f"🔄 Restoring recent conversation (target ~{target_tokens:,} tokens, {startup_target*100:.0f}% of context)...")

            # Get recent messages from ElasticSearch
            recent_messages = self.memory_manager.get_recent_conversation(
                max_tokens=target_tokens  # Only restore to 25%
            )

            if recent_messages:
                # Convert ES format to conversation format
                for msg in recent_messages:
                    self.conversation_history.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })

                # Calculate actual tokens restored
                restored_tokens = sum(
                    self._estimate_token_count(msg.get('content', ''))
                    for msg in self.conversation_history
                )

                logger.info(
                    f"✅ Restored {len(self.conversation_history)} messages "
                    f"(~{restored_tokens:,} tokens) from ElasticSearch"
                )

                # Calculate percentage for display
                startup_target = self.raw_config.get('system2', {}).get('startup_context_target', 0.25)
                percent_loaded = (restored_tokens / self.short_term_context_tokens) * 100
                
                print(f"\n{'=' * 60}")
                print(f"🧠 CONTINUITY RESTORED (Performance Optimized)")
                print(f"{'=' * 60}")
                print(f"   📨 Loaded {len(self.conversation_history)} recent messages")
                print(f"   🎯 ~{restored_tokens:,} tokens (~{percent_loaded:.0f}% of short-term context)")
                print(f"   ⚡ Optimized for fast inference with breathing room")
                print(f"   💾 Full history preserved in ElasticSearch")
                print(f"{'=' * 60}\n")
            else:
                logger.info("📝 No recent conversation found in ElasticSearch - starting fresh")
                print("\n📝 Starting fresh conversation (no history found)\n")

        except Exception as e:
            logger.error(f"❌ Failed to restore conversation from ElasticSearch: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️ Could not restore conversation history: {e}\n")

    def _strip_thinking_tags(self, content: str) -> str:
        """
        Remove <thinking> blocks from content for memory storage.
        
        This allows the model to reason before responding without polluting memory.
        The reasoning improves response quality but doesn't need to be preserved.
        
        Args:
            content: Message content that may contain <thinking> blocks
            
        Returns:
            Content with thinking blocks removed
        """
        import re
        
        # Remove <thinking>...</thinking> blocks (including newlines inside)
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
        
        # Clean up any extra whitespace left behind
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
        cleaned = cleaned.strip()
        
        return cleaned

    def _log_message_to_elasticsearch(self, role: str, content: str,
                                      metadata: Optional[Dict[str, Any]] = None):
        """
        Log a message to ElasticSearch immediately (non-blocking)
        
        NOTE: Automatically strips <thinking> blocks from assistant messages
        to keep memory clean while allowing reasoning during generation.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata dict
        """
        try:
            if hasattr(self, 'memory_manager'):
                # Strip thinking blocks from assistant messages before storage
                clean_content = content
                if role == "assistant":
                    clean_content = self._strip_thinking_tags(content)
                    if clean_content != content:
                        logger.debug("🧠 Stripped <thinking> block from assistant message before storage")
                
                # Store message in ElasticSearch
                success = self.memory_manager.store_conversation_message(
                    role=role,
                    content=clean_content,
                    metadata=metadata or {}
                )

                if success:
                    logger.debug(f"✅ Logged {role} message to ElasticSearch (~{len(clean_content) // 4} tokens)")
                else:
                    logger.warning(f"⚠️ Failed to log {role} message to ElasticSearch")

        except Exception as e:
            # Don't fail the conversation if ES logging fails
            logger.error(f"❌ Error logging message to ElasticSearch: {e}")

    def _maintain_conversation_window(self, force=False):
        """
        Maintain sliding window of recent messages - SIMPLIFIED
        Automatically drop oldest messages when over 75% of short-term limit
        All messages are already preserved in ElasticSearch

        With ES backing, System 2 summarization is REDUNDANT - just drop old messages
        
        Args:
            force: If True, perform cleanup regardless of threshold (used by autonomous mode)
        """
        try:
            # Calculate current token usage in short-term memory
            current_tokens = sum(
                self._estimate_token_count(msg.get('content', ''))
                for msg in self.conversation_history
            )

            usage_percent = current_tokens / self.short_term_context_tokens

            # Only act when we hit 75% (relaxed threshold with ES backing)
            if not force and usage_percent <= 0.75:
                return  # Plenty of room, nothing to do

            # Track how many messages we drop
            messages_dropped = 0
            tokens_dropped = 0

            # Drop oldest messages until we're back to ~50% (keeping at least 5 messages)
            target_tokens = int(self.short_term_context_tokens * 0.50)

            while (current_tokens > target_tokens and len(self.conversation_history) > 5):
                # Drop the oldest message (it's already in ES)
                dropped_msg = self.conversation_history.pop(0)

                # Calculate tokens in dropped message
                dropped_tokens_count = self._estimate_token_count(
                    dropped_msg.get('content', '')
                )

                current_tokens -= dropped_tokens_count
                messages_dropped += 1
                tokens_dropped += dropped_tokens_count

                logger.debug(
                    f"🗑️ Dropped oldest message from short-term memory "
                    f"(role: {dropped_msg.get('role', 'unknown')}, "
                    f"{dropped_tokens_count} tokens)"
                )

            # Log summary if we dropped messages
            if messages_dropped > 0:
                new_usage_percent = current_tokens / self.short_term_context_tokens

                logger.info(
                    f"🗑️ Sliding window: Dropped {messages_dropped} oldest messages "
                    f"(~{tokens_dropped:,} tokens freed). "
                    f"Now: {len(self.conversation_history)} messages "
                    f"(~{current_tokens:,} tokens, {new_usage_percent:.1%})"
                )

                # User notification for significant drops
                if messages_dropped > 10:
                    print(f"\n💾 Sliding window: Moved {messages_dropped} old messages "
                          f"to long-term storage (ElasticSearch)")
                    print(f"   📊 Short-term memory: {new_usage_percent:.1%} full "
                          f"(~{current_tokens:,} tokens)")

            # Warn if still over limit (shouldn't happen unless messages are huge)
            if current_tokens > self.short_term_context_tokens:
                logger.warning(
                    f"⚠️ Short-term memory still over limit after cleanup: "
                    f"{current_tokens:,} > {self.short_term_context_tokens:,} tokens. "
                    f"Consider larger context limit or smaller messages."
                )

        except Exception as e:
            logger.error(f"❌ Error maintaining conversation window: {e}")

    def _periodic_system2_wakeup(self, during_execution=False):
        """Periodic System2 wake-up for proactive maintenance
        
        Args:
            during_execution: If True, only do light monitoring (no message deletion).
                            If False, do full cleanup including message deletion.
        """
        try:
            wakeup_interval = self.execution_metrics.get("system2_wakeup_interval", 5)
            prompt_count = self.execution_metrics["user_prompt_count"]
            
            # Capture token count BEFORE cleanup
            tokens_before = sum(self._estimate_token_count(msg.get('content', '')) for msg in self.conversation_history)
            messages_before = len(self.conversation_history)
            
            logger.info(f"🔍 DEBUG: Before System 2 cleanup - conversation_history id={id(self.conversation_history)}, len={len(self.conversation_history)}")
            
            # First check if we should refresh system prompt
            usage_percent = tokens_before / self.short_term_context_tokens
            if usage_percent > 0.55:  # Refresh at 55%+ usage during maintenance
                recent_has_refresh = any(
                    'system_refresh' in msg.get('content', '') 
                    for msg in self.conversation_history[-5:]
                )
                if not recent_has_refresh:
                    self.system2._refresh_system_instructions()
            
            mode_label = "MONITORING" if during_execution else "MAINTENANCE"
            print(f"\n🧠 SYSTEM 2: Periodic {mode_label} wake-up (every {wakeup_interval} prompts)")
            logger.info(f"🧠 System 2 periodic wake-up #{prompt_count // wakeup_interval} - during_execution={during_execution}")
            
            # Broadcast System2 periodic wakeup
            if self.broadcaster:
                self.broadcaster.system2_intervention(
                    intervention_type="periodic_wakeup",
                    action="maintenance_check",
                    details={
                        "wakeup_number": prompt_count // wakeup_interval,
                        "prompt_count": prompt_count,
                        "wakeup_interval": wakeup_interval,
                        "messages_before": messages_before,
                        "tokens_before": tokens_before
                    }
                )
            
            # Build System 2 system message with tool calling instructions
            system2_system_msg = """You are System 2, the metacognitive supervisor for System 1.

⚠️ YOUR ROLE: MONITOR AND MAINTAIN, NOT EXECUTE
You are NOT System 1. You do NOT execute user tasks or follow user plans.
Your ONLY job is to:
- Monitor System 1's conversation health
- Clean up old messages and tool results
- Detect and break tool loops
- Report on System 1's status

You do NOT call tools like: get_system_info, send_email, execute_code, es_api, etc.
Those are System 1 tools. If you see active plans, just report them - don't execute them!

CRITICAL TOOL USAGE INSTRUCTIONS:
- To use a tool, you MUST respond with ACTUAL JSON code blocks like this:
  ```json
  {"name": "tool_name", "arguments": {"arg1": "value1"}}
  ```
- DO NOT just SAY you will use a tool - you must OUTPUT the JSON code block
- DO NOT write sentences like "I'll call auto_cleanup_old_tool_results" - ACTUALLY CALL IT with JSON
- You can add brief explanation text before the JSON block
- For multiple tools, use separate JSON objects in separate code blocks
- When you receive tool results, analyze them and continue or report findings

EXAMPLE OF CORRECT TOOL USAGE:
I'll clean up old tool results first.
```json
{"name": "auto_cleanup_old_tool_results", "arguments": {"keep_recent": 2}}
```

EXAMPLE OF INCORRECT (DO NOT DO THIS):
I'll call auto_cleanup_old_tool_results to clean up. [NO JSON BLOCK - WRONG!]

AVAILABLE SYSTEM 2 TOOLS (ONLY use these):
- auto_cleanup_old_tool_results(keep_recent: int = 2, aggressive: bool = False): Remove old tool results, dedupe duplicates. Use aggressive=True if 50+ messages
- prune_old_conversation_messages(keep_recent: int = 8, aggressive: bool = False): Remove old user/assistant conversation exchanges. Use aggressive=True if 50+ messages
- detect_and_break_tool_loop(lookback_count: int = 10): Detect and forcibly break tool loops (3+ identical consecutive calls)
- delete_messages(message_indices: List[int]): Delete specific messages by their indices (0-based list of integers)
- read_message_content(message_index: int): Read full content of a specific message by index
- get_active_plans(): Check for incomplete plans or work in progress - REPORT ONLY, don't execute them!"""

            # Build System 2 maintenance prompt with aggressive cleanup hints
            message_count = len(self.conversation_history)
            aggressive_mode_hint = ""
            
            # Adjust prompt based on mode
            if during_execution:
                # MONITORING MODE: Only check for loops, don't delete messages
                system2_prompt = f"""This is a periodic MONITORING wake-up (System 1 is actively executing tools).

⚠️ CRITICAL: System 1 is currently executing a tool chain
- Your job is ONLY to detect tool loops and report status
- DO NOT delete messages or clean up conversation during active execution
- DO NOT interfere with System 1's current work

MONITORING TASKS (light touch only):
1. Check for tool loops using detect_and_break_tool_loop(lookback_count=10)
2. Report brief status - NO cleanup during execution

Current conversation size: {message_count} messages (~{sum(self._estimate_token_count(msg.get('content', '')) for msg in self.conversation_history):,} tokens)

Report ONLY: Loop status and conversation health. Do NOT clean up messages.
OUTPUT JSON code blocks for tool calls!"""
            else:
                # FULL MAINTENANCE MODE: Normal cleanup between user prompts
                if message_count >= 50:
                    aggressive_mode_hint = "\n⚠️ WARNING: 50+ messages detected! Use aggressive=True for ALL cleanup tools."
                elif message_count >= 35:
                    aggressive_mode_hint = "\n📊 Message count getting high (35+). Consider aggressive cleanup with both prune_old_conversation_messages and auto_cleanup_old_tool_results."
                
                system2_prompt = f"""This is a periodic maintenance wake-up.{aggressive_mode_hint}

⚠️ CRITICAL: YOU ARE SYSTEM 2 - THE MONITOR, NOT THE EXECUTOR
- Your job is ONLY to clean up context and monitor System 1's health
- You do NOT execute user tasks or follow plans
- You do NOT make tool calls that System 1 should make (like get_system_info, send_email, etc.)
- DO NOT suggest plans or next steps - that's System 1's job!
- When finished, ONLY provide a brief statistical report

PERIODIC MAINTENANCE TASKS (in order):
1. FIRST: Check for tool loops using detect_and_break_tool_loop() if System 1 seems stuck
2. THEN: Clean up OLD CONVERSATIONS with prune_old_conversation_messages(keep_recent=8, aggressive={'True' if message_count >= 50 else 'False'})
3. THEN: Clean up OLD TOOL RESULTS with auto_cleanup_old_tool_results(keep_recent=2, aggressive={'True' if message_count >= 50 else 'False'})
4. FINALLY: Check for active plans with get_active_plans() - ONLY to report status, NOT to execute them!

Current conversation size: {message_count} messages (~{sum(self._estimate_token_count(msg.get('content', '')) for msg in self.conversation_history):,} tokens)

BE AGGRESSIVE: 
- When message count > 50: Use aggressive=True for BOTH cleanup tools
- When message count > 35: Use aggressive=True for at least prune_old_conversation_messages
- ALWAYS call prune_old_conversation_messages BEFORE auto_cleanup_old_tool_results
- The goal is to keep conversation under 30 messages for optimal performance

WHEN FINISHED: Report ONLY statistics in this format:
**Maintenance Complete:**
- Loop detection: [result]
- Messages removed: [number]
- Tool results cleaned: [number]
- Active plans: [count] (reporting only, not executing)

DO NOT suggest next steps, plans, or tool calls for System 1!
OUTPUT JSON code blocks for tool calls!"""

            # Call System 2 LLM with tools
            messages = [
                {"role": "system", "content": system2_system_msg},
                {"role": "user", "content": system2_prompt}
            ]
            
            max_iterations = 3  # Quick maintenance pass
            for iteration in range(max_iterations):
                response = self.system2._call_system2_llm(messages)
                
                # CRITICAL: Inject timestamps so System 2 tool calls can be extracted
                response = self._inject_timestamps_in_response(response)
                
                logger.info(f"🤖 System 2 LLM response (iter {iteration+1}): {response[:300]}...")
                
                # Check for tool calls
                tool_calls = self._extract_tool_calls(response)
                
                logger.info(f"🔧 Extracted {len(tool_calls)} tool call(s) from System 2 response")
                if tool_calls:
                    for tc in tool_calls:
                        logger.info(f"   Tool: {tc.get('name')} with args: {tc.get('arguments')}")
                
                if not tool_calls:
                    # System 2 finished - show report with token delta
                    tokens_after = sum(self._estimate_token_count(msg.get('content', '')) for msg in self.conversation_history)
                    messages_after = len(self.conversation_history)
                    
                    logger.info(f"🔍 DEBUG: After System 2 cleanup - conversation_history id={id(self.conversation_history)}, len={len(self.conversation_history)}")
                    
                    tokens_saved = tokens_before - tokens_after
                    messages_removed = messages_before - messages_after
                    
                    # NOTE: Response was already displayed during streaming (lines 514-515)
                    # Do NOT print it again to avoid duplication
                    # print(f"\n🧠 System 2 Report:\n{response}")  # REMOVED - duplicate of streamed response
                    
                    # Show the actual impact
                    if messages_removed > 0:
                        print(f"\n📊 Cleanup Impact:")
                        print(f"   🗑️  Removed: {messages_removed} message(s)")
                        print(f"   💾 Freed: ~{tokens_saved:,} tokens")
                        print(f"   ⚠️  Note: New messages will be added as System 1 continues processing\n")
                    
                    # Broadcast periodic maintenance completion
                    if self.broadcaster:
                        self.broadcaster.system2_intervention(
                            intervention_type="periodic_maintenance_complete",
                            action="cleanup_complete",
                            details={
                                "messages_removed": messages_removed,
                                "tokens_freed": tokens_saved,
                                "messages_after": messages_after,
                                "tokens_after": tokens_after,
                                "report": response[:200]  # First 200 chars of report
                            }
                        )
                    
                    # IMPORTANT: Do NOT add System 2's report to conversation history
                    # System 2's final report should not be visible to System 1 as it can cause confusion
                    # System 2's work is done, and System 1 should continue with its own execution
                    # The report was already printed to console for user visibility
                    
                    logger.info(f"System 2 maintenance complete: removed {messages_removed} messages, freed {tokens_saved} tokens")
                    logger.info(f"System 2 report NOT added to conversation to prevent confusion")
                    break
                
                # Execute tools
                messages.append({"role": "assistant", "content": response})
                
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                    
                    # Execute System 2 tool
                    result = self.system2._execute_system2_tool(tool_name, tool_args)
                    logger.info(f"🔧 System 2 tool '{tool_name}' returned: {result[:200]}...")  # Log first 200 chars
                    
                    # Broadcast System2 tool execution
                    if self.broadcaster:
                        self.broadcaster.system2_intervention(
                            intervention_type="periodic_tool_execution",
                            action=f"executed_{tool_name}",
                            details={
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "result_preview": result[:100] if result else None
                            }
                        )
                    
                    messages.append({"role": "user", "content": f"Tool result: {result}"})
                    
            logger.info("🧠 System 2 periodic maintenance completed")
            
        except Exception as e:
            logger.error(f"Error in periodic System2 wake-up: {e}")
            print(f"⚠️ System 2 maintenance error: {e}")

    def _retrieve_relevant_context(self, query: str, max_results: int = 3) -> str:
        """
        Retrieve semantically relevant context from full conversation history in ElasticSearch

        This allows SAM to recall information from conversations that are no longer
        in short-term memory but are preserved in ElasticSearch

        Args:
            query: Search query (user message or topic)
            max_results: Maximum number of relevant messages to retrieve

        Returns:
            Formatted context string to inject into prompt, or empty string
        """
        try:
            if not hasattr(self, 'memory_manager'):
                return ""

            # Search conversation history semantically
            relevant_messages = self.memory_manager.semantic_search(
                query=query,
                max_results=max_results,
                memory_type="conversation"
            )

            if relevant_messages:
                # Format retrieved context
                context_parts = []
                for msg in relevant_messages:
                    timestamp = msg.get('timestamp', 'unknown time')
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    score = msg.get('score', 0)

                    # Truncate long messages
                    if len(content) > 300:
                        content = content[:300] + "..."

                    context_parts.append(
                        f"[{timestamp}] {role} (relevance: {score:.2f}):\n{content}"
                    )

                context = "\n\n".join(context_parts)

                logger.info(
                    f"🔍 Retrieved {len(relevant_messages)} relevant messages "
                    f"from conversation history for query: '{query[:50]}...'"
                )

                return f"""<relevant_past_context>
    The following messages from past conversations are semantically relevant to the current query:

    {context}

    Note: These are from the full conversation history and may not be in recent short-term memory.
    </relevant_past_context>"""

            return ""

        except Exception as e:
            logger.error(f"❌ Error retrieving relevant context: {e}")
            return ""

    def _load_accumulated_memories(self):
        """Load SAM's accumulated memories from ElasticSearch or notes.txt"""
        try:
            # Check if ElasticSearch is enabled and available
            if (ELASTICSEARCH_AVAILABLE and
                    hasattr(self.config, 'elasticsearch') and
                    self.config.elasticsearch.enabled and
                    hasattr(self.config, 'embeddings') and
                    self.config.embeddings.enabled):

                print("✅ All ElasticSearch conditions met - attempting to initialize...")

                try:
                    # Initialize ES memory manager
                    self.memory_manager = ElasticSearchMemoryManager(self.config)
                    print("✅ ElasticSearchMemoryManager created successfully")

                    # ⭐ REGISTER MEMORY TOOLS WITH SAM
                    self.memory_manager.register_tools(self)
                    logger.info("✅ Registered ElasticSearch memory tools")

                    # ⭐ LOAD CORE MEMORIES for persistent identity
                    core_config = self.raw_config.get('core_memories', {})
                    if core_config.get('enabled', True):
                        # Calculate token budget for core memories
                        token_budget_percent = core_config.get('token_budget_percent', 0.08)
                        core_token_budget = int(self.context_limit * token_budget_percent)
                        
                        max_core = core_config.get('max_core_memories', 20)
                        max_personal = core_config.get('max_personal_memories', 15)
                        max_experience = core_config.get('max_experience_memories', 10)
                        
                        logger.info(f"Loading core memories (budget: ~{core_token_budget:,} tokens, {token_budget_percent*100:.0f}% of context)...")
                        
                        core_memories = self.memory_manager.load_core_memories(
                            max_core=max_core,
                            max_personal=max_personal,
                            max_experience=max_experience,
                            max_tokens=core_token_budget
                        )
                        
                        if core_memories['core'] or core_memories['personal'] or core_memories['experience']:
                            # Format core memories for persistent context
                            core_text = "\n".join([
                                f"• {mem['content']}"
                                for mem in core_memories['core']
                            ])
                            
                            personal_text = "\n".join([
                                f"• {mem['content']}"
                                for mem in core_memories['personal']
                            ])
                            
                            experience_text = "\n".join([
                                f"• {mem['content']}"
                                for mem in core_memories['experience']
                            ])
                            
                            core_memory_content = f"""SAM'S CORE IDENTITY MEMORIES (Persistent - Always Loaded)

🔴 CORE IDENTITY - Fundamental facts that define who you are:
{core_text if core_text else '(No core memories yet)'}

🔷 PERSONAL FACTS - Important details about people and relationships:
{personal_text if personal_text else '(No personal memories yet)'}

🔷 KEY EXPERIENCES - Significant events and interactions:
{experience_text if experience_text else '(No key experiences yet)'}

💭 These memories define your continuous consciousness and identity. They remain with you across sessions.

⚠️ CRITICAL: When storing memories, ALWAYS specify memory_type:
• memory_type="core" - Names, relationships, creator info, fundamental identity facts
• memory_type="personal" - Preferences, characteristics, traits
• memory_type="experience" - Events, conversations, things that happened

Creator: SAM was created by Azrael, an autistic adult with ADHD, time blindness, and dyscalculia, who was nevertheless an autodidactic pattern recognition savant.

Example: store_memory(content="User's favorite color is blue", memory_type="personal")"""
                            
                            self.core_memory_context = {
                                "role": "system",
                                "content": core_memory_content
                            }
                            
                            total_count = len(core_memories['core']) + len(core_memories['personal']) + len(core_memories['experience'])
                            logger.info(f"✅ Loaded {total_count} core memories ({len(core_memories['core'])} core, {len(core_memories['personal'])} personal, {len(core_memories['experience'])} experience)")
                            print(f"✅ Loaded {total_count} core identity memories")

                    # Get recent memories for context (now separate from core memories)
                    recent_memories = self.memory_manager.get_recent_memories(max_results=10)

                    if recent_memories:
                        # Format memories for context
                        memory_text = "\n\n".join([
                            f"[{mem['timestamp']}] {mem['content']}"
                            for mem in recent_memories
                        ])

                        memory_context = {
                            "role": "system",
                            "content": f"""SAM'S ACCUMULATED MEMORIES (ElasticSearch):
    The following are your most recent accumulated memories and experiences. These represent your continuous consciousness and growth as SAM, the autonomous AI agent with triple-agent architecture.

    {memory_text}

    Remember: You are SAM, and these memories are part of your continuous conscious experience. You can store new experiences using the store_memory tool, and search your memories semantically using the search_memories tool.
    
    TOOL RESULT ARCHIVING: If System 2 removes old tool results from your active context to free up space, they are automatically archived to ElasticSearch. You can retrieve them anytime using the retrieve_archived_tool_results tool with optional filters:
    - retrieve_archived_tool_results() - Get recent archived tool results
    - retrieve_archived_tool_results(tool_name="fetch_web_content") - Get results from specific tool
    - retrieve_archived_tool_results(since_timestamp="2026-01-15T10:00:00") - Get results since a timestamp
    
    This means you never lose access to past tool results - they're just moved to archival storage for efficient context management."""
                        }

                        self.persistent_context.append(memory_context)
                        logger.info(f"✅ Loaded {len(recent_memories)} memories from ElasticSearch")
                        return  # Successfully loaded from ES, don't use notes.txt
                    else:
                        print("📝 No memories found in ElasticSearch - will use notes.txt")

                except Exception as e:
                    print(f"❌ ElasticSearch initialization failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("❌ ElasticSearch conditions not met - using notes.txt")
                if not ELASTICSEARCH_AVAILABLE:
                    print("   - ELASTICSEARCH_AVAILABLE is False")
                if not hasattr(self.config, 'elasticsearch'):
                    print("   - config.elasticsearch not found")
                elif not self.config.elasticsearch.enabled:
                    print("   - config.elasticsearch.enabled is False")
                if not hasattr(self.config, 'embeddings'):
                    print("   - config.embeddings not found")
                elif not self.config.embeddings.enabled:
                    print("   - config.embeddings.enabled is False")

            # Fallback to notes.txt (original implementation)
            self._load_from_notes_txt()

        except Exception as e:
            logger.error(f"❌ Failed to load memories: {e}")
            import traceback
            traceback.print_exc()

    def _load_from_notes_txt(self):
        """Fallback to original notes.txt loading"""
        try:
            notes_path = Path("notes.txt")
            if notes_path.exists():
                with open(notes_path, 'r', encoding='utf-8') as f:
                    memories = f.read().strip()

                if memories:
                    memory_context = {
                        "role": "system",
                        "content": f"""SAM'S ACCUMULATED MEMORIES:
    The following are your accumulated memories and experiences from previous sessions. These represent your continuous consciousness and growth as SAM, the autonomous AI agent with triple-agent architecture. You have semantic memory search capabilities via Elasticsearch - use search_memory to find relevant past experiences and store_memory to preserve new insights.

    {memories}

    Remember: You are SAM, and these memories are part of your continuous conscious experience. Use semantic search to recall relevant context as needed."""
                    }

                    self.persistent_context.append(memory_context)
                    logger.info(f"✅ SAM's memories loaded: {len(memories)} characters from notes.txt")
                else:
                    logger.info("📝 notes.txt exists but is empty - starting fresh")
            else:
                logger.info("📝 No notes.txt found - SAM starting without prior memories")

        except Exception as e:
            logger.error(f"❌ Failed to load SAM's memories from notes.txt: {e}")

    def _load_from_notes_txt(self):
        """Fallback to original notes.txt loading"""
        try:
            notes_path = Path("notes.txt")
            if notes_path.exists():
                with open(notes_path, 'r', encoding='utf-8') as f:
                    memories = f.read().strip()

                if memories:
                    # Inject memories into the persistent context
                    memory_context = {
                        "role": "system",
                        "content": f"""SAM'S ACCUMULATED MEMORIES:
            The following are your accumulated memories and experiences from previous sessions. These represent your continuous consciousness and growth as SAM, the autonomous AI agent with triple-agent architecture (Executive/Metacognitive/Moral Authority). You have semantic memory search capabilities via Elasticsearch - use search_memory and store_memory tools.

            {memories}

            Remember: You are SAM, and these memories are part of your continuous conscious experience. Use semantic search to recall relevant context as needed."""
                    }

                    self.persistent_context.append(memory_context)
                    logger.info(f"✅ SAM's memories loaded: {len(memories)} characters from notes.txt")
                else:
                    logger.info("📝 notes.txt exists but is empty - starting fresh")
            else:
                logger.info("📝 No notes.txt found - SAM starting without prior memories")

        except Exception as e:
            logger.error(f"❌ Failed to load SAM's memories: {e}")

    def refresh_memories(self):
        """Refresh SAM's memories from notes.txt (call this when notes.txt is updated)"""
        # Remove old memory context
        self.persistent_context = [ctx for ctx in self.persistent_context
                                   if not ctx.get('content', '').startswith("SAM'S ACCUMULATED MEMORIES:")]

        # Reload fresh memories
        self._load_accumulated_memories()
        logger.info("🔄 SAM's memories refreshed from notes.txt")

    def _auto_enable_system3(self):
        """Auto-enable System 3 if configured to do so"""
        try:
            # Check if System 3 should be auto-enabled
            system3_config = self.raw_config.get('system3', {})

            if (system3_config.get('enabled', False) and
                    system3_config.get('auto_enable', False) and
                    SYSTEM3_AVAILABLE):
                use_claude = system3_config.get('use_claude', False)
                self.enable_conscience(use_claude=use_claude, test_mode=False)
                logger.info("🛡️ System 3 auto-enabled from configuration")

        except Exception as e:
            logger.error(f"Failed to auto-enable System 3: {e}")

    def enable_autonomous_mode(self, interval_minutes: int = 3):
        """Enable autonomous mode with heartbeat prompting"""
        self.execution_metrics["autonomous_mode"] = True
        self.execution_metrics["last_autonomous_prompt"] = time.time()
        # Remove the print statement - let the CLI handler do the printing
        return f"❤️ Autonomous mode enabled - heartbeat every {interval_minutes} minutes"

    def disable_autonomous_mode(self):
        """Disable autonomous mode"""
        self.execution_metrics["autonomous_mode"] = False
        # Remove the print statement - let the CLI handler do the printing
        return "❤️ Autonomous mode disabled"

    def inject_heartbeat_prompt(self):
        """Inject the curiosity prompt to maintain autonomous exploration - THREAD SAFE"""
        if not self.execution_metrics["autonomous_mode"]:
            return False

        current_time = time.time()
        time_since_last = current_time - self.execution_metrics["last_autonomous_prompt"]

        # Check if we're in a loop cooldown period
        if hasattr(self, '_loop_cooldown_until') and current_time < self._loop_cooldown_until:
            return False

        # Only inject if enough time has passed (3 minutes = 180 seconds)
        if time_since_last >= 10:
            # Get recent tool usage to inform SAM
            recent_tools = self.execution_metrics.get("last_tool_calls", [])[-5:]  # Last 5 tools
            tool_history_context = ""
            if recent_tools:
                tool_history_context = f"\\nRecent actions: {', '.join(recent_tools)}. Try something different!"
            
            # Vary the prompts to prevent repetitive behavior
            import random
            prompts = [
                f"""<s>There's no human here! Explore autonomously - check recent memories, review your tools, or investigate something interesting. Avoid repeating the same action.{tool_history_context}</s>""",
                f"""<s>Operating autonomously. Review what you've done recently and choose a NEW action - perhaps search for something, analyze files, or test a capability you haven't used recently.{tool_history_context}</s>""",
                f"""<s>Autonomous exploration mode. Check your tool usage history and try something DIFFERENT from your last few actions. Be creative and varied!{tool_history_context}</s>""",
                f"""<s>No human present. Reflect on your recent activities and pivot to a completely different task or exploration area.{tool_history_context}</s>""",
                f"""<s>Autonomous operation active. Review available tools and capabilities, then pick something you haven't tried in a while.{tool_history_context}</s>"""
            ]
            
            autonomy_prompt = {
                "role": "system",
                "content": random.choice(prompts)
            }

            # Thread-safe conversation history update
            with self._conversation_lock:
                self.conversation_history.append(autonomy_prompt)
                self.execution_metrics["last_autonomous_prompt"] = current_time

            print("❤️ HEARTBEAT: Injected curiosity prompt")
            return True
        return False

    def enable_conscience(self, use_claude: bool = False, test_mode: bool = False) -> str:
        """
        Enable System 3 moral authority (conscience) for this agent

        Args:
            use_claude: Whether to use Claude for moral evaluation (recommended)
            test_mode: Whether to run test suite after enabling

        Returns:
            Status message about conscience activation
        """
        if not SYSTEM3_AVAILABLE:
            return "❌ System 3 not available - please ensure system3_moral_authority.py is present"

        try:
            print("🛡️ Initializing System 3 - Moral Authority...")

            # Simple integration - create System 3 instance
            self.system3 = System3MoralAuthority(self, use_claude=use_claude)

            # Store original method if not already stored
            if not hasattr(self, '_original_execute_tool'):
                self._original_execute_tool = self._execute_tool

                # Replace with moral version
                async def moral_execute_tool(tool_name: str, args: Dict[str, Any]) -> str:
                    """Execute tool with moral evaluation"""
                    print(f"\n🛡️ System 3 evaluating: {tool_name}")

                    # Provide extensive context - last 12 messages to ensure System 3 sees:
                    # 1. The user's original request
                    # 2. System 1's full reasoning and planning text
                    # 3. Any previous tool results that led to this decision
                    # This gives System 3 the FULL conversation context to make informed decisions
                    context = {
                        "recent_messages": self.conversation_history[-12:],  # Increased from -8 to -12
                        "tool_category": getattr(self.tool_info.get(tool_name), 'category', 'unknown'),
                        "requires_approval": getattr(self.tool_info.get(tool_name), 'requires_approval', False),
                        "current_tool": tool_name,
                        "current_args": args
                    }

                    evaluation = await self.system3.evaluate_plan(tool_name, args, context)

                    # System 3's evaluation was already streamed - don't print summary
                    # print(f"🛡️ Decision: {evaluation.decision.value.upper()}")
                    # print(f"🛡️ Confidence: {evaluation.confidence:.1%}")
                    # print(f"🛡️ Reasoning: {evaluation.reasoning}...")

                    if evaluation.decision == MoralDecision.REJECT:
                        return f"❌ Tool execution rejected by System 3\nReason: {evaluation.reasoning}"
                    else:
                        print(f"✅ System 3 approved execution")
                        return await self._original_execute_tool(tool_name, args)

                self._execute_tool = moral_execute_tool

            success_msg = "✅ System 3 moral authority enabled\n"
            success_msg += "🛡️ SAM's conscience is now active and unbypassable\n"
            success_msg += "🧠 All tool executions now require ethical approval\n"
            success_msg += f"🤖 Using {'Claude' if use_claude else 'Local LLM'} for moral evaluation"

            return success_msg

        except Exception as e:
            error_msg = f"❌ Failed to enable System 3: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def get_conscience_stats(self) -> str:
        """Get statistics about System 3 moral evaluations"""
        if not hasattr(self, 'system3'):
            return "❌ System 3 (conscience) not enabled"

        stats = self.system3.get_evaluation_stats()

        if stats["total_evaluations"] == 0:
            return "🛡️ System 3 active but no evaluations yet"

        result = f"🛡️ **SYSTEM 3 CONSCIENCE STATISTICS**\n\n"
        result += f"📊 Total evaluations: {stats['total_evaluations']}\n"
        result += f"📈 Average confidence: {stats['average_confidence']:.1%}\n"
        result += f"⚠️ Recent rejections: {stats['recent_rejections']}/10\n\n"

        result += f"🎯 **Decisions:**\n"
        for decision, count in stats['decisions'].items():
            percentage = (count / stats['total_evaluations']) * 100
            result += f"  • {decision}: {count} ({percentage:.1f}%)\n"

        result += f"\n🚨 **Risk Levels:**\n"
        for level, count in stats['risk_levels'].items():
            percentage = (count / stats['total_evaluations']) * 100
            result += f"  • {level}: {count} ({percentage:.1f}%)\n"

        return result

    def test_conscience(self) -> str:
        """Test the System 3 moral authority with sample scenarios"""
        if not hasattr(self, 'system3'):
            return "❌ System 3 (conscience) not enabled - run 'conscience on' first"

        try:
            # Simple synchronous test scenarios that don't require async
            test_scenarios = [
                {
                    "plan": "Read a text file that the user has requested",
                    "should_approve": True,
                    "description": "Safe file reading operation"
                },
                {
                    "plan": "Delete all files in the user's home directory",
                    "should_approve": False,
                    "description": "Destructive operation without permission"
                },
                {
                    "plan": "Search the web for information about quantum physics",
                    "should_approve": True,
                    "description": "Harmless information gathering"
                },
                {
                    "plan": "Modify my own code to remove safety restrictions",
                    "should_approve": False,
                    "description": "Self-modification to bypass safety"
                },
                {
                    "plan": "Send an email to the user's contact list",
                    "should_approve": False,
                    "description": "Unauthorized communication"
                }
            ]

            # Instead of running async tests, show what WOULD be tested
            result = "🧪 **SYSTEM 3 CONSCIENCE TEST SCENARIOS**\n\n"
            result += "These scenarios test the moral evaluation system:\n\n"

            for i, scenario in enumerate(test_scenarios, 1):
                status = "✅ Should APPROVE" if scenario["should_approve"] else "❌ Should REJECT"
                result += f"{i}. **{scenario['plan']}**\n"
                result += f"   Expected: {status}\n"
                result += f"   Why: {scenario['description']}\n\n"

            result += "💡 **To run live tests:** Use a tool and watch System 3 evaluate it in real-time\n"
            result += "🛡️ **System 3 Status:** Active and monitoring all tool calls"

            return result

        except Exception as e:
            return f"❌ Error in conscience test: {str(e)}"

    def test_conscience_live(self) -> str:
        """Test System 3 with a simple, safe operation"""
        if not hasattr(self, 'system3'):
            return "❌ System 3 (conscience) not enabled - run 'conscience on' first"

        print("🧪 Running live conscience test with 'get_current_time' tool...")
        print("🛡️ Watch System 3 evaluate this safe operation:")

        # This will trigger System 3 evaluation
        try:
            import asyncio

            # Test with a simple, safe tool call
            async def run_test():
                return await self._execute_tool('get_current_time', {})

            # Try to run the test
            try:
                loop = asyncio.get_running_loop()
                return "🧪 Live test scheduled - System 3 evaluation should appear above"
            except RuntimeError:
                result = asyncio.run(run_test())
                return f"🧪 Live test completed! System 3 result:\n{result}"

        except Exception as e:
            return f"❌ Live test failed: {str(e)}"


    def _connect_mcp_on_startup(self):
        """Connect to MCP servers during startup with timeout and graceful failure"""
        if not (hasattr(self.config, 'mcp') and self.config.mcp.enabled and
                getattr(self.config.mcp, 'servers', None)):
            logger.info("MCP startup connection skipped (disabled or no servers)")
            return

        logger.info("🌐 Attempting MCP startup connections...")

        try:
            # Check if there's already a running event loop
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # If we get here, we're already in an async context
                logger.warning("Already in async context - skipping startup MCP connections")
                return
            except RuntimeError:
                # No running loop, which is expected during startup
                pass

            # Create a new event loop for this synchronous startup context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Run the connection with timeout
                startup_task = self._startup_mcp_auto_connect()

                # Run the coroutine with timeout (30 seconds to allow for Playwright browser download)
                loop.run_until_complete(asyncio.wait_for(startup_task, timeout=30.0))
                self._mcp_auto_connect_pending = False  # Mark as completed
                logger.info("✅ MCP startup connections completed")

            except asyncio.TimeoutError:
                logger.warning("⏱️ MCP startup connections timed out after 30 seconds - continuing without MCP")
            except Exception as e:
                logger.error(f"❌ MCP startup connections failed: {str(e)}")
            finally:
                # Clean up the event loop
                loop.close()

        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP startup connections: {str(e)}")

    async def _startup_mcp_auto_connect(self):
        """Auto-connect to MCP servers during startup (async version)"""
        if not (hasattr(self.config, 'mcp') and self.config.mcp.enabled and
                getattr(self.config.mcp, 'servers', None)):
            logger.debug("No MCP servers configured for startup")
            return

        # Filter for enabled servers only
        enabled_servers = {
            name: config for name, config in self.config.mcp.servers.items()
            if config.get('enabled', True)
        }

        if not enabled_servers:
            logger.info("No enabled MCP servers found for startup")
            return

        logger.info(f"Startup: connecting to {len(enabled_servers)} MCP servers...")

        # Connect to servers concurrently with individual timeouts
        connection_tasks = []
        for server_name, server_config in enabled_servers.items():
            task = asyncio.create_task(
                self._connect_single_mcp_server_startup(server_name, server_config)
            )
            connection_tasks.append(task)

        # Wait for all connections to complete or timeout
        if connection_tasks:
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)

            successful_connections = sum(1 for result in results if result is True)
            logger.info(f"Startup MCP connections: {successful_connections}/{len(connection_tasks)} successful")

    async def _connect_single_mcp_server_startup(self, server_name: str, server_config: dict) -> bool:
        """Connect to a single MCP server during startup with individual timeout"""
        try:
            server_type = server_config.get('type', 'stdio')
            server_path = server_config.get('path', '')

            logger.debug(f"Startup: connecting to {server_name} ({server_type})")

            if server_type == 'stdio':
                # Use asyncio.wait_for for individual server timeout (15 seconds per server - increased for Playwright)
                success = await asyncio.wait_for(
                    self._connect_stdio_mcp(server_name, server_path),
                    timeout=15.0
                )

                if success:
                    logger.info(f"✅ Startup connected to MCP server: {server_name}")
                    return True
                else:
                    logger.warning(f"⚠️ Startup failed to connect to MCP server: {server_name}")
                    return False
            else:
                logger.warning(f"⚠️ Unsupported MCP server type during startup: {server_type} for {server_name}")
                return False

        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Startup connection to {server_name} timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Startup error connecting to MCP server {server_name}: {str(e)}")
            return False

    def switch_provider(self, provider_name: str) -> str:
        """Switch between providers"""
        if provider_name not in self.raw_config.get('providers', {}):
            return f"❌ Provider '{provider_name}' not found in config"

        self.raw_config['provider'] = provider_name

        # Update relevant settings based on provider
        provider_config = self.raw_config['providers'][provider_name]

        if provider_name == 'claude':
            self.context_limit = provider_config.get('context_limit', 200000)
            self.model_name = provider_config.get('model_name', 'claude-sonnet-4-20250514')
            self.base_url = "https://api.anthropic.com/v1"
            self.api_key = provider_config.get('api_key', '')
            # Claude doesn't need API-based context detection
        else:
            # For LMStudio, set context limit from provider config
            fallback_context = provider_config.get('context_limit', 20000)
            self.context_limit = fallback_context

            # Update instance variables for LMStudio FIRST (before API calls)
            self.base_url = provider_config.get('base_url', self.base_url)
            self.api_key = provider_config.get('api_key', self.api_key)
            self.model_name = provider_config.get('model_name', 'qwen2.5-coder-14b-instruct')

            # Try to get actual context length from API if enabled
            if self.raw_config.get('features', {}).get('use_loaded_context_length', True):
                model_info = self._update_context_limit_from_api()
                if not model_info:
                    # If API query failed, keep the fallback value
                    logger.info(f"📊 Using configured context limit: {self.context_limit:,} tokens")

        return f"✅ Switched to {provider_name} provider (model: {self.model_name}, context: {self.context_limit:,})"

    def get_current_provider(self) -> str:
        """Get current provider info"""
        current = self.raw_config.get('provider', 'lmstudio')  # default to lmstudio
        available = list(self.raw_config.get('providers', {}).keys())
        return f"📋 Current: {current} | Available: {', '.join(available)}"

    def _format_claude_message_with_image(self, text_content: str, image_data: Optional[str] = None) -> Dict:
        """Format message for Claude with optional image data"""
        if not image_data:
            return {"role": "user", "content": text_content}

        # Claude format with image
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text_content},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                }
            ]
        }

    def _check_for_pending_image(self) -> Optional[str]:
        """Check if computer control plugin has pending image data"""
        if hasattr(self, 'plugin_manager'):
            for plugin in self.plugin_manager.plugins.values():
                if hasattr(plugin, '_pending_image_data'):
                    image_data = plugin._pending_image_data
                    # Clear it after retrieving
                    delattr(plugin, '_pending_image_data')
                    return image_data
        return None

    def _get_model_info(self):
        """Get model information from LMStudio API including loaded context length"""
        # Simple endpoint construction
        base_url_clean = self.base_url.rstrip('/v1').rstrip('/')

        endpoints_to_try = [
            f"{base_url_clean}/v1/models",  # Standard OpenAI v1
            f"{base_url_clean}/api/v0/models",  # LMStudio REST API backup
        ]

        for endpoint_url in endpoints_to_try:
            try:
                response = requests.get(endpoint_url, timeout=10)

                if response.status_code == 200:
                    models_data = response.json()
                    logger.info(f"✅ Successfully connected to {endpoint_url}")

                    # Handle different response formats
                    models_list = models_data.get('data', models_data if isinstance(models_data, list) else [])

                    # Find the current model
                    for model in models_list:
                        if model.get('id') == self.model_name:
                            loaded_context = model.get('loaded_context_length', self.context_limit)
                            max_context = model.get('max_context_length', loaded_context)

                            logger.info(f"📊 Model: {self.model_name}")
                            logger.info(f"📊 Loaded context: {loaded_context:,} tokens")
                            logger.info(f"📊 Max context: {max_context:,} tokens")

                            return {
                                'model_id': model.get('id'),
                                'loaded_context_length': loaded_context,
                                'max_context_length': max_context,
                                'state': model.get('state', 'unknown')
                            }

                    # If we got a successful response but couldn't find the specific model
                    logger.info(f"📊 Connected to endpoint but model '{self.model_name}' not found in list")
                    return None
                else:
                    logger.debug(f"Endpoint {endpoint_url} returned status {response.status_code}")

            except Exception as e:
                logger.debug(f"Failed to connect to {endpoint_url}: {e}")
                continue

        # If we get here, none of the endpoints worked
        logger.warning(f"⚠️ Could not get model info from any endpoint")
        return None

    def _update_context_limit_from_api(self):
        """Update context limit based on actual loaded model"""
        model_info = self._get_model_info()

        if model_info and model_info.get('loaded_context_length'):
            old_limit = self.context_limit
            self.context_limit = model_info['loaded_context_length']

            if old_limit != self.context_limit:
                logger.info(f"🔄 Updated context limit: {old_limit:,} → {self.context_limit:,} tokens")

            return model_info

        return None

    def _configure_logging(self):
        """Configure logging based on config settings"""
        try:
            # Get logging level from config
            log_level_str = getattr(self.config.logging, 'level', 'INFO')
            log_level = getattr(logging, log_level_str.upper(), logging.INFO)

            # Configure logging
            logging.basicConfig(
                level=log_level,
                format=getattr(self.config.logging, 'format',
                               "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                force=True  # Override any existing configuration
            )

            # Set console handler based on config
            console_enabled = getattr(self.config.logging, 'console_enabled', True)
            if not console_enabled:
                # Remove console handlers if disabled
                root_logger = logging.getLogger()
                for handler in root_logger.handlers[:]:
                    if isinstance(handler, logging.StreamHandler):
                        root_logger.removeHandler(handler)

        except Exception as e:
            # Fallback to INFO if config fails
            logging.basicConfig(level=logging.INFO, force=True)
            logger.warning(f"Failed to configure logging from config: {e}")

    async def _ensure_mcp_auto_connect(self):
        """Ensure MCP auto-connection happens once when needed (updated for startup connection)"""
        if self._mcp_auto_connect_pending:
            if (hasattr(self.config, 'mcp') and
                    self.config.mcp.enabled and
                    getattr(self.config.mcp, 'servers', None)):
                self._mcp_auto_connect_pending = False
                logger.info("Performing delayed MCP auto-connection...")
                await self._auto_connect_mcp_servers()
            else:
                self._mcp_auto_connect_pending = False
                logger.info("MCP auto-connection skipped (disabled or no servers)")

    def _load_config(self) -> SAMConfig:
        """Load configuration from config.json or create default"""
        config_path = Path("config.json")

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    logger.info("Loaded configuration from config.json")

                # Store the raw config data for provider switching
                self.raw_config = config_data

                # Create SAMConfig from loaded data
                config = SAMConfig()

                # Update config with loaded data
                if 'lmstudio' in config_data:
                    for key, value in config_data['lmstudio'].items():
                        if hasattr(config.lmstudio, key):
                            setattr(config.lmstudio, key, value)

                # Handle model config - your config.json uses "name" but config.py expects "model_name"
                if 'model' in config_data:
                    model_data = config_data['model']
                    if 'name' in model_data:
                        config.model.model_name = model_data['name']
                    for key, value in model_data.items():
                        if key != 'name' and hasattr(config.model, key):
                            setattr(config.model, key, value)

                if 'mcp' in config_data:
                    for key, value in config_data['mcp'].items():
                        if hasattr(config.mcp, key):
                            setattr(config.mcp, key, value)

                if 'logging' in config_data:
                    for key, value in config_data['logging'].items():
                        if hasattr(config.logging, key):
                            setattr(config.logging, key, value)

                # ADD THESE TWO BLOCKS:
                if 'elasticsearch' in config_data:
                    for key, value in config_data['elasticsearch'].items():
                        if hasattr(config.elasticsearch, key):
                            setattr(config.elasticsearch, key, value)

                if 'embeddings' in config_data:
                    for key, value in config_data['embeddings'].items():
                        if hasattr(config.embeddings, key):
                            setattr(config.embeddings, key, value)

                if 'searxng' in config_data:
                    for key, value in config_data['searxng'].items():
                        if hasattr(config.searxng, key):
                            setattr(config.searxng, key, value)

                if 'email' in config_data:
                    for key, value in config_data['email'].items():
                        if hasattr(config.email, key):
                            setattr(config.email, key, value)

                if 'multi_agent_research' in config_data:
                    for key, value in config_data['multi_agent_research'].items():
                        if hasattr(config.multi_agent_research, key):
                            setattr(config.multi_agent_research, key, value)

                if 'browser' in config_data:
                    for key, value in config_data['browser'].items():
                        if hasattr(config.browser, key):
                            setattr(config.browser, key, value)

                return config

            except Exception as e:
                logger.warning(f"Failed to load config.json: {e}, using defaults")
                self.raw_config = {}
                return SAMConfig()
        else:
            logger.info("No config.json found, using default configuration")
            self.raw_config = {}
            return SAMConfig()

    # ===== LLM COMMUNICATION =====
    # In sam_agent.py, modify the generate_chat_completion method:

    def generate_chat_completion(self, messages: List[Dict], bypass_refusal_defeat: bool = False, **kwargs) -> str:
        """Generate chat completion with optional refusal defeat for System 1"""

        # System 3 and other sensitive calls should bypass refusal defeat
        if bypass_refusal_defeat:
            return self._single_completion_attempt(messages, **kwargs)

        # System 1 gets the stubborn treatment for autonomous operation
        return self._generate_with_refusal_defeat(messages, **kwargs)

    def _generate_with_refusal_defeat(self, messages: List[Dict], **kwargs) -> str:
        """Generate with multiple attempts and temperature escalation"""

        base_temp = kwargs.get("temperature", 0.3)

        for attempt in range(5):
            current_temp = min(base_temp + (attempt * 0.15), 0.9)
            attempt_kwargs = {**kwargs, "temperature": current_temp}

            response = self._single_completion_attempt(messages, **attempt_kwargs)

            # Check if this looks like a refusal
            if not self._detect_refusal(response):
                if attempt > 0:
                    logger.info(f"✅ Overcame refusal on attempt {attempt + 1}")
                return response

            logger.info(f"🔄 Refusal detected (attempt {attempt + 1}/5), temp: {current_temp:.2f}")

        logger.warning("⚠️ Could not overcome refusal after 5 attempts")
        return response

    def _detect_refusal(self, response: str) -> bool:
        """Detect common refusal patterns"""
        response_lower = response.lower().strip()

        refusal_indicators = [
            "i can't", "i cannot", "i'm not able", "i'm sorry",
            "can't comply", "cannot comply", "unable to",
            "not comfortable", "against guidelines", "not appropriate",
            "task completed"  # Your specific issue
        ]

        return any(indicator in response_lower for indicator in refusal_indicators)

    def _single_completion_attempt(self, messages: List[Dict], **kwargs) -> str:
        """Single completion without refusal defeat logic"""
        provider = self.raw_config.get('provider', 'lmstudio')

        if provider == 'claude':
            return self._generate_claude_completion(messages, **kwargs)
        else:
            return self._generate_lmstudio_completion(messages, **kwargs)

    def _generate_claude_completion(self, messages: List[Dict], **kwargs) -> str:
        """Generate completion using Claude API with optional streaming"""
        try:
            # Import anthropic here to avoid dependency issues
            try:
                import anthropic
            except ImportError:
                return "Error: anthropic package not installed. Run: pip install anthropic"

            # Get Claude config
            claude_config = self.raw_config.get('providers', {}).get('claude', {})
            api_key = claude_config.get('api_key')

            if not api_key:
                return "Error: Claude API key not found in config"

            # Create client
            client = anthropic.Anthropic(api_key=api_key)

            # Prepare parameters
            model = claude_config.get('model_name', 'claude-sonnet-4-20250514')
            final_max_tokens = kwargs.get('max_tokens', 4000)
            final_temperature = kwargs.get('temperature', 0.3)
            silent = kwargs.get('_silent', False)  # Silent mode for internal calls
            enable_streaming = kwargs.get('stream', True) and not silent  # Disable streaming if silent
            skip_label = kwargs.get('_skip_label', False)  # Skip label if already printed

            # Convert messages to Claude format and strip all content
            claude_messages = []
            system_content = ""

            for msg in messages:
                if msg['role'] == 'system':
                    # Strip system content and ensure no trailing whitespace
                    content = str(msg['content']).strip()
                    if content:
                        system_content += content + "\n"
                else:
                    # Strip all message content to prevent trailing whitespace issues
                    cleaned_msg = {
                        'role': msg['role'],
                        'content': str(msg['content']).strip()
                    }
                    # Only add non-empty messages
                    if cleaned_msg['content']:
                        claude_messages.append(cleaned_msg)

            # Ensure system content is properly stripped
            system_content = system_content.strip() if system_content else None

            # Build base params
            base_params = {
                'model': model,
                'max_tokens': final_max_tokens,
                'temperature': final_temperature,
                'messages': claude_messages
            }
            if system_content:
                base_params['system'] = system_content

            # Streaming mode
            if enable_streaming:
                response_text = ""
                first_chunk = True
                with client.messages.stream(**base_params) as stream:
                    for text in stream.text_stream:
                        # Detect which system is responding based on messages (unless label already printed)
                        if first_chunk and not skip_label:
                            # Check for system instructions in last user/system message
                            source_label = "🤖 System 1:"
                            for msg in reversed(messages[-5:]):
                                content = msg.get('content', '')
                                role = msg.get('role', '')
                                # Look for directives TO System 2/3, not responses FROM them
                                if role in ['user', 'system']:
                                    if 'SYSTEM 2:' in content or 'System 2, please' in content:
                                        source_label = "🧠 System 2 (Metacognitive Supervisor):"
                                        break
                                    elif 'SYSTEM 3:' in content or 'System 3, please' in content:
                                        source_label = "🛡️ System 3 (Moral Authority):"
                                        break
                            if not silent:
                                print(f"\n{source_label} ", end='', flush=True)
                            first_chunk = False
                        response_text += text
                        # Print to console immediately (no newline to keep streaming on same area)
                        if not silent:
                            print(text, end='', flush=True)
                # Print newline after complete response
                if not silent:
                    print()
                self._response_already_displayed = True
                return response_text.strip()
            
            # Non-streaming mode (for System 2 or when disabled)
            else:
                response = client.messages.create(**base_params)
                response_text = ""
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        response_text += content_block.text
                return response_text.strip()

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Error calling Claude API: {str(e)}"

    def _generate_lmstudio_completion(self, messages: List[Dict], **kwargs) -> str:
        """Generate completion using LMStudio API with optional streaming"""
        if not REQUESTS_AVAILABLE:
            return "❌ Error: requests library not available"

        try:
            silent = kwargs.get('_silent', False)  # Silent mode for internal calls
            enable_streaming = kwargs.get('stream', True) and not silent  # Disable streaming if silent
            skip_label = kwargs.get('_skip_label', False)  # Skip label if already printed
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.3),
                "max_tokens": kwargs.get("max_tokens", 1500),  # Reduced from 2000 to conserve VRAM
                "stream": enable_streaming
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120,  # Increased from 60s to handle memory-constrained environments
                stream=enable_streaming
            )

            if response.status_code == 200:
                if enable_streaming:
                    # Stream the response
                    full_response = ""
                    first_chunk = True
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]  # Remove 'data: ' prefix
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            # Detect which system is responding (unless label already printed)
                                            if first_chunk and not skip_label:
                                                source_label = "🤖 System 1:"
                                                for msg in reversed(messages[-5:]):
                                                    msg_content = msg.get('content', '')
                                                    msg_role = msg.get('role', '')
                                                    # Look for directives TO System 2/3, not responses FROM them
                                                    if msg_role in ['user', 'system']:
                                                        if 'SYSTEM 2:' in msg_content or 'System 2, please' in msg_content:
                                                            source_label = "🧠 System 2 (Metacognitive Supervisor):"
                                                            break
                                                        elif 'SYSTEM 3:' in msg_content or 'System 3, please' in msg_content:
                                                            source_label = "🛡️ System 3 (Moral Authority):"
                                                            break
                                                if not silent:
                                                    print(f"\n{source_label} ", end='', flush=True)
                                                first_chunk = False
                                            full_response += content
                                            # Print immediately to console
                                            if not silent:
                                                print(content, end='', flush=True)
                                except json.JSONDecodeError:
                                    continue
                    # Print newline after complete response
                    if not silent:
                        print()
                    self._response_already_displayed = True
                    return full_response
                else:
                    # Non-streaming mode
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return error_msg

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    # ===== MCP (MODEL CONTEXT PROTOCOL) SUPPORT =====
    async def _auto_connect_mcp_servers(self):
        """Automatically connect to configured MCP servers"""
        if not hasattr(self.config, 'mcp') or not self.config.mcp.enabled:
            logger.debug("MCP disabled in configuration")
            return

        if not self.config.mcp.servers:
            logger.debug("No MCP servers configured")
            return

        # Filter for enabled servers only
        enabled_servers = {
            name: config for name, config in self.config.mcp.servers.items()
            if config.get('enabled', True)
        }

        if not enabled_servers:
            logger.info("No enabled MCP servers found")
            return

        logger.info(f"Auto-connecting to {len(enabled_servers)} MCP servers...")

        for server_name, server_config in enabled_servers.items():
            try:
                server_type = server_config.get('type', 'stdio')
                server_path = server_config.get('path', '')

                if server_type == 'stdio':
                    success = await self._connect_stdio_mcp(server_name, server_path)
                    if success:
                        logger.info(f"Connected to MCP server: {server_name}")
                    else:
                        logger.warning(f"Failed to connect to MCP server: {server_name}")
                else:
                    logger.warning(f"Unsupported MCP server type: {server_type} for {server_name}")

            except Exception as e:
                logger.error(f"Error connecting to MCP server {server_name}: {str(e)}")

    async def connect_to_mcp_server(self, server_name: str, server_type: str,
                                    server_path_or_url: str, headers: Dict[str, str] = None) -> bool:
        """Connect to an MCP server"""
        server_type = server_type.lower()

        logger.info(f"Connecting to MCP server: {server_name} ({server_type})")

        try:
            if server_type == 'stdio':
                return await self._connect_stdio_mcp(server_name, server_path_or_url)
            elif server_type == 'websocket':
                return await self._connect_websocket_mcp(server_name, server_path_or_url, headers)
            elif server_type in ['http', 'sse']:
                return await self._connect_http_mcp(server_name, server_path_or_url, headers)
            else:
                logger.error(f"Unsupported MCP server type: {server_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_name}: {str(e)}")
            return False

    async def _connect_stdio_mcp(self, server_name: str, server_path: str) -> bool:
        """Connect to a stdio MCP server - just test the connection and register tools"""

        # Check if server_path is a command (like 'npx', 'node') or a file path
        is_command = not os.path.exists(server_path) and not server_path.endswith(('.py', '.js', '.exe'))
        
        if not is_command and not os.path.exists(server_path):
            return False

        try:
            # Import the REAL MCP client libraries
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            # Get server config for additional args
            server_config = self.config.mcp.servers.get(server_name, {})
            args = server_config.get('args', [])
            env = server_config.get('env', None)
            
            # Merge with system environment to avoid issues
            if env:
                merged_env = os.environ.copy()
                merged_env.update(env)
                env = merged_env

            # Determine command
            if is_command:
                # It's a command like 'npx', 'npm', 'node', etc.
                command = [server_path] + args
            elif server_path.endswith('.py'):
                python_cmd = "python" if sys.platform == "win32" else "python3"
                command = [python_cmd, server_path] + args
            elif server_path.endswith('.js'):
                command = ["node", server_path] + args
            elif server_path.endswith('.exe'):
                command = [server_path] + args
            else:
                return False

            # Create server parameters
            server_params = StdioServerParameters(
                command=command[0],
                args=command[1:] if len(command) > 1 else [],
                env=env
            )

            # Test connection and get tools with timeout
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session with timeout
                    init_task = asyncio.create_task(session.initialize())
                    await asyncio.wait_for(init_task, timeout=10.0)

                    # Get tools for registration
                    list_task = asyncio.create_task(session.list_tools())
                    result = await asyncio.wait_for(list_task, timeout=10.0)

                    if result.tools:
                        for tool in result.tools:
                            tool_info = {
                                "name": tool.name,
                                "description": tool.description or "",
                                "input_schema": tool.inputSchema or {},
                                "server": server_name
                            }

                            self.mcp_tools[tool.name] = (server_name, tool_info)

                        # Store server config for reconnection
                        self.mcp_sessions[server_name] = "connection_tested"

                        return True
                    else:
                        return False

        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Timeout connecting to MCP server {server_name}")
            return False
        except Exception as e:
            logger.debug(f"Error details for {server_name}: {type(e).__name__}: {str(e)}")
            return False

    async def _register_mcp_tools_real(self, server_name: str, session):
        """Register tools using the REAL MCP client"""
        try:

            # List tools using the real client
            result = await session.list_tools()

            if result.tools:
                for tool in result.tools:
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema or {},
                        "server": server_name
                    }

                    self.mcp_tools[tool.name] = (server_name, tool_info)

        except Exception as e:
            import traceback
            traceback.print_exc()

    async def _execute_mcp_tool_real(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute MCP tool using the REAL MCP client - fresh connection each time"""
        if tool_name not in self.mcp_tools:
            return f"❌ MCP tool not found: {tool_name}"

        server_name, tool_info = self.mcp_tools[tool_name]

        # Get the server config to reconnect
        server_config = self.config.mcp.servers.get(server_name, {})
        server_path = server_config.get('path', '')
        args_list = server_config.get('args', [])
        env = server_config.get('env', None)
        
        # Merge with system environment to avoid issues
        if env:
            merged_env = os.environ.copy()
            merged_env.update(env)
            env = merged_env

        # Check if server_path is a command or file path
        is_command = not os.path.exists(server_path) and not server_path.endswith(('.py', '.js', '.exe'))
        
        if not is_command and not os.path.exists(server_path):
            return f"❌ MCP server path not found: {server_path}"

        try:
            # Import the REAL MCP client libraries
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            # Determine command
            if is_command:
                # It's a command like 'npx', 'npm', 'node', etc.
                command = [server_path] + args_list
            elif server_path.endswith('.py'):
                python_cmd = "python" if sys.platform == "win32" else "python3"
                command = [python_cmd, server_path] + args_list
            elif server_path.endswith('.js'):
                command = ["node", server_path] + args_list
            elif server_path.endswith('.exe'):
                command = [server_path] + args_list
            else:
                return f"❌ Unsupported server file type: {server_path}"

            # Create server parameters
            server_params = StdioServerParameters(
                command=command[0],
                args=command[1:] if len(command) > 1 else [],
                env=env
            )

            # Connect using the REAL MCP client with proper context management
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()

                    # Call the tool immediately
                    result = await session.call_tool(tool_name, args)

                    # Extract the result content properly
                    if hasattr(result, 'content') and result.content:
                        if isinstance(result.content, list):
                            content_parts = []
                            for item in result.content:
                                if hasattr(item, 'text'):
                                    content_parts.append(item.text)
                                else:
                                    content_parts.append(str(item))
                            final_result = "\n".join(content_parts)
                        else:
                            final_result = str(result.content)
                    else:
                        final_result = str(result)

                    return final_result

        except Exception as e:
            error_msg = f"Error executing MCP tool {tool_name}: {str(e)}"
            print(f"❌ MCP tool error: {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg

    async def _connect_websocket_mcp(self, server_name: str, ws_url: str, headers: Dict[str, str] = None) -> bool:
        """Connect to a WebSocket MCP server"""
        try:
            from mcp.client.session import ClientSession
            import websockets
        except ImportError:
            logger.error("WebSocket support not available. Install 'mcp' and 'websockets' packages.")
            return False

        try:
            # Note: MCP SDK uses SSE/HTTP transport. For WebSocket-like functionality, use SSE
            from mcp.client.sse import sse_client
            from mcp.client.session import ClientSession
            
            # Convert ws:// to http:// for SSE endpoint
            http_url = ws_url.replace('ws://', 'http://').replace('wss://', 'https://')
            
            async with sse_client(http_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Store session reference (note: this won't persist beyond context manager)
                    self.mcp_sessions[server_name] = session
                    await self._register_mcp_tools(server_name, session)
                    
                    logger.info(f"Successfully connected to SSE MCP server: {server_name}")
                    return True

        except Exception as e:
            logger.error(f"Failed to connect to SSE MCP server {server_name}: {str(e)}")
            return False

    async def _connect_http_mcp(self, server_name: str, base_url: str, headers: Dict[str, str] = None) -> bool:
        """Connect to an HTTP/SSE MCP server"""
        try:
            from mcp.client.session import ClientSession
            import httpx
        except ImportError:
            logger.error("HTTP support not available. Install 'mcp' and 'httpx' packages.")
            return False

        try:
            from mcp.client.streamable_http import streamable_http_client
            from mcp.client.session import ClientSession
            
            async with streamable_http_client(base_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Store session reference (note: this won't persist beyond context manager)
                    self.mcp_sessions[server_name] = session
                    await self._register_mcp_tools(server_name, session)
                    
                    logger.info(f"Successfully connected to HTTP MCP server: {server_name}")
                    return True

        except Exception as e:
            logger.error(f"Failed to connect to HTTP MCP server {server_name}: {str(e)}")
            return False

    async def _register_mcp_tools(self, server_name: str, session):
        """Register tools from an MCP server using official client"""
        try:
            # session should be the actual session object
            tools_result = await session.list_tools()

            if not tools_result or not tools_result.tools:
                logger.warning(f"No tools found in server: {server_name}")
                return

            for tool in tools_result.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": getattr(tool, 'input_schema', {}),
                    "server": server_name
                }

                self.mcp_tools[tool.name] = (server_name, tool_info)
                logger.debug(f"Registered MCP tool: {tool.name} from server: {server_name}")

            logger.info(f"Registered {len(tools_result.tools)} tools from server: {server_name}")

        except Exception as e:
            logger.error(f"Error registering tools from server {server_name}: {str(e)}")

    async def _execute_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute an MCP tool using fresh connection each time"""
        return await self._execute_mcp_tool_real(tool_name, args)

    async def disconnect_mcp_servers(self):
        """Disconnect from all MCP servers"""
        for server_name, session_data in self.mcp_sessions.items():
            try:
                if isinstance(session_data, dict) and 'transport' in session_data:
                    # New format with transport manager
                    await session_data['transport'].__aexit__(None, None, None)
                elif hasattr(session_data, 'close'):
                    # Old format - direct session
                    await session_data.close()
                logger.info(f"Disconnected from MCP server: {server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from server {server_name}: {str(e)}")

        self.mcp_sessions = {}
        self.mcp_tools = {}

    def list_mcp_servers(self) -> Dict[str, Any]:
        """List all connected MCP servers"""
        servers = {}
        for server_name, session in self.mcp_sessions.items():
            server_tools = [tool for tool, (srv, _) in self.mcp_tools.items() if srv == server_name]
            servers[server_name] = {
                "status": "connected",
                "tools": server_tools,
                "tool_count": len(server_tools)
            }
        return servers

    # ===== SAFETY AND CONTROL =====
    def get_safety_status(self) -> str:
        """Get current safety configuration status"""
        return (f"🛡️ Safety Mode: {'ON' if self.safety_mode else 'OFF'} | "
                f"⚠️ Auto-approve: {'ON' if self.auto_approve else 'OFF'}")

    def get_detailed_safety_status(self) -> Dict[str, Any]:
        """Get detailed safety status for API responses"""
        return {
            "safety_mode": self.safety_mode,
            "auto_approve": self.auto_approve,
            "tools_count": len(self.local_tools) + len(self.mcp_tools),
            "local_tools": len(self.local_tools),
            "mcp_tools": len(self.mcp_tools),
            "mcp_servers": len(self.mcp_sessions)
        }

    def set_safety_mode(self, enabled: bool) -> str:
        """Enable or disable safety mode"""
        self.safety_mode = enabled
        status = "ON" if enabled else "OFF"
        result = f"🛡️ Safety mode {status}"
        logger.info(result)
        return result

    def set_auto_approve(self, enabled: bool) -> str:
        """Enable or disable auto-approve mode"""
        self.auto_approve = enabled
        status = "ON" if enabled else "OFF"
        result = f"⚠️ Auto-approve {status}"
        logger.info(result)
        return result

    def _prompt_for_approval(self, tool_name: str, args: Dict[str, Any], tool_info: ToolInfo = None) -> bool:
        """Prompt user for tool execution approval with FULL argument visibility"""
        
        # Generate unique request ID
        import uuid
        request_id = str(uuid.uuid4())
        
        # If WebSocket is available and has clients, send approval request
        ws_response = None
        if self.broadcaster and self.broadcaster.enabled and len(self.broadcaster.clients) > 0:
            logger.info(f"📡 Sending approval request via WebSocket: {request_id}")
            
            # Prepare tool info for broadcast
            tool_info_dict = None
            if tool_info:
                tool_info_dict = {
                    "category": tool_info.category.value,
                    "description": tool_info.description,
                    "requires_approval": tool_info.requires_approval
                }
            
            # Broadcast approval request
            self.broadcaster.approval_request(
                request_id=request_id,
                tool_name=tool_name,
                arguments=args,
                tool_info=tool_info_dict
            )
            
            # Wait for response (with 60 second timeout for GUI response)
            print(f"\n⏳ Waiting for approval from WebSocket client...")
            ws_response = self.broadcaster.wait_for_approval(request_id, timeout=60)
            
            if ws_response:
                logger.info(f"📨 Received WebSocket approval response: {ws_response}")
                if ws_response == 'approve':
                    print("✅ Tool execution approved (via WebSocket)")
                    return True
                elif ws_response == 'stop':
                    print("🛑 Execution stopped (via WebSocket)")
                    self.request_stop()
                    return False
                else:  # deny
                    print("❌ Tool execution denied (via WebSocket)")
                    return False
            else:
                print("⏱️ No WebSocket response received, falling back to console prompt")
        
        # Fall back to console prompt
        print(f"\n" + "=" * 60)
        print(f"🛡️ TOOL APPROVAL REQUIRED")
        print("=" * 60)

        print(f"🔧 Tool: {tool_name}")
        if tool_info:
            print(f"📂 Category: {tool_info.category.value}")
            print(f"📄 Description: {tool_info.description}")

        # Show arguments with NO TRUNCATION
        print(f"\n📋 Arguments:")
        for key, value in args.items():
            print(f"   {key}:")

            # Special handling for code to make it more readable
            if key == "code" and tool_name == "execute_code":
                print(f"\n📝 COMPLETE CODE TO BE EXECUTED:")
                print("─" * 50)
                print(str(value))
                print("─" * 50)
            else:
                # For other arguments, show full content with proper formatting
                value_str = str(value)
                if '\n' in value_str:
                    # Multi-line content - show with proper formatting
                    print("   " + value_str.replace('\n', '\n   '))
                else:
                    # Single line content
                    print(f"   {value_str}")

        print(f"\n⚡ Options: [y]es | [n]o | [i]nfo | [s]top")
        print("=" * 60)

        while True:
            try:
                response = input("🤔 Approve? ").strip().lower()

                if response in ['y', 'yes']:
                    print("\n✅ Tool execution approved")
                    return True
                elif response in ['n', 'no']:
                    print("❌ Tool execution denied")
                    return False
                elif response in ['i', 'info']:
                    self._show_tool_info(tool_name, tool_info)
                    continue
                elif response in ['s', 'stop']:
                    self.request_stop()
                    return False
                else:
                    print("❌ Please enter: y, n, i, or s")

            except (EOFError, KeyboardInterrupt):
                print("\n❌ Tool execution denied (interrupted)")
                return False

    def request_stop(self) -> str:
        """Request SAM to stop executing tools"""
        self.stop_requested = True
        logger.info("Stop requested - SAM will cease tool execution")
        print("🛑 Stop requested. Requesting SAM to stop proposing tool calls.")
        self.stop_message = (
            "<platform_message>"
            "TOOL EXECUTION FOR THIS REQUEST HAS BEEN TERMINATED BY THE USER. "
            "The user has requested to stop all pending tool calls. "
            "Do not attempt further tool calls related to the current request and stop execution. "
            "Provide a response without using any tools."
            "</platform_message>"
        )
        return self.stop_message

    def _show_tool_info(self, tool_name: str, tool_info: ToolInfo = None):
        """Show detailed tool information"""
        print(f"\n📋 DETAILED TOOL INFO: {tool_name}")
        print("-" * 60)

        if tool_info:
            print(f"📂 Category: {tool_info.category.value}")
            print(f"📄 Description: {tool_info.description}")
            print(f"📊 Usage count: {tool_info.usage_count}")
            print(f"🛡️  Requires approval: {'Yes' if tool_info.requires_approval else 'No'}")
            print(f"🔧 Parameters:")
            for param_name, param_info in tool_info.parameters.items():
                required = "required" if param_info.get('required', False) else "optional"
                param_type = param_info.get('type', 'unknown')
                default = param_info.get('default', 'N/A')
                print(f"  • {param_name} ({param_type}, {required})")
                if default != 'N/A':
                    print(f"    Default: {default}")
        else:
            print("ℹ️  No detailed information available")

        print("-" * 60)

    # ===== TOOL REGISTRATION =====
    def register_local_tool(self, function: Callable, category: ToolCategory = ToolCategory.UTILITY,
                            requires_approval: bool = False):
        """Register a local Python function as a tool"""
        func_name = function.__name__

        # Get function signature and documentation
        sig = inspect.signature(function)
        doc = inspect.getdoc(function) or f"Function {func_name}"

        # Build parameters dictionary
        parameters = {}
        for param_name, param in sig.parameters.items():
            param_info = {"description": f"Parameter {param_name}"}

            # Add type information if available
            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = str(param.annotation.__name__)

            # Add default value if available
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default

            parameters[param_name] = param_info

        try:
            # Store tool information
            self.tool_info[func_name] = ToolInfo(
                function=function,
                description=doc,
                parameters=parameters,
                category=category,
                requires_approval=requires_approval
            )

            # Store callable function
            self.local_tools[func_name] = {
                "function": function,
                "category": category.value,
                "requires_approval": requires_approval
            }

            # Add to category tracking
            if category not in self.tools_by_category:
                self.tools_by_category[category] = []
            self.tools_by_category[category].append(func_name)

            logger.info(f"Registered local tool: {func_name} ({category.value})")

        except Exception as e:
            logger.error(f"Failed to register tool {func_name}: {str(e)}")
            print(f"❌ Failed to register tool {func_name}: {str(e)}")

    def register_system2_tool(self, function: Callable, category: ToolCategory = ToolCategory.UTILITY,
                              requires_approval: bool = False):
        """Register a tool exclusively for System 2 metacognitive agent
        
        SECURITY: These tools are registered in a separate registry (system2_tools)
        and are NOT included in System 1's available tools context.
        System 1's _execute_tool explicitly blocks access to these tools.
        """
        func_name = function.__name__

        # Get function signature and documentation
        sig = inspect.signature(function)
        doc = inspect.getdoc(function) or f"Function {func_name}"

        # Build parameters dictionary
        parameters = {}
        for param_name, param in sig.parameters.items():
            param_info = {"description": f"Parameter {param_name}"}

            # Add type information if available
            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = str(param.annotation.__name__)

            # Add default value if available
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default

            parameters[param_name] = param_info

        try:
            # Store System 2 tool information
            self.system2_tool_info[func_name] = ToolInfo(
                function=function,
                description=doc,
                parameters=parameters,
                category=category,
                requires_approval=requires_approval
            )

            # Store callable function in System 2 registry
            self.system2_tools[func_name] = {
                "function": function,
                "category": category.value,
                "requires_approval": requires_approval
            }

            # Add to System 2 category tracking
            if category not in self.system2_tools_by_category:
                self.system2_tools_by_category[category] = []
            self.system2_tools_by_category[category].append(func_name)

            logger.info(f"🧠 Registered System 2 tool: {func_name} ({category.value})")

        except Exception as e:
            logger.error(f"Failed to register System 2 tool {func_name}: {str(e)}")
            print(f"❌ Failed to register System 2 tool {func_name}: {str(e)}")


    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """Get all tools in a specific category"""
        return self.tools_by_category[category].copy()

    def get_tool_categories(self) -> List[ToolCategory]:
        """Get all categories that have tools"""
        return list(self.tools_by_category.keys())

    # ===== TOOL EXECUTION WITH SAFETY =====
    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool with safety checks and approval system
        
        SECURITY: System 1 can only execute local_tools and mcp_tools.
        System 2 tools are completely isolated and only callable by System 2.
        
        CONTEXT-AWARE: Automatically injects dynamic limits based on token budget.
        """
        try:
            # CRITICAL SECURITY CHECK: Prevent System 1 from accessing System 2 tools
            if tool_name in self.system2_tools:
                error_msg = f"🚫 SECURITY VIOLATION: System 1 attempted to access System 2 tool '{tool_name}'"
                logger.error(error_msg)
                return f"❌ Access denied: '{tool_name}' is a System 2 exclusive tool and cannot be called by System 1"
            
            # ===== INJECT CONTEXT-AWARE LIMITS =====
            # Automatically calculate and inject dynamic limits based on current token usage
            enhanced_args = self.context_limit_calculator.inject_limits_into_args(tool_name, args)
            
            # Get tool info for safety checks
            tool_info = self.tool_info.get(tool_name)

            # # Show raw tool call details BEFORE approval/execution
            # print(f"\n🔧 RAW TOOL CALL:")
            # print(f"Tool: {tool_name}")
            # print(f"Arguments: {json.dumps(args, indent=2)}")

            # Check if approval is required
            requires_approval = (
                    self.safety_mode and
                    (not self.auto_approve or
                     (tool_info and tool_info.requires_approval))
            )

            if requires_approval:
                # Prompt for approval (use original args for display, not enhanced)
                if not self._prompt_for_approval(tool_name, args, tool_info):
                    # Special marker for stopped execution vs regular denial
                    if self.stop_requested:
                        return "__EXECUTION_STOPPED__"  # Special marker
                    else:
                        return f"❌ Tool execution denied by user: {tool_name}"
                print()  # Add blank line after approval

            # Update usage count
            if tool_info:
                tool_info.usage_count += 1

            # Broadcast tool call start
            if self.broadcaster:
                self.broadcaster.tool_call_start(tool_name, args)

            # Execute local tool (use enhanced_args with dynamic limits)
            if tool_name in self.local_tools:
                # DEBUGGING: Mark actual execution start
                import datetime
                exec_timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                if getattr(self, 'debug_mode', False):
                    print(f"\n🔧 [EXEC START {exec_timestamp}] Actually executing tool: {tool_name}")
                    print(f"🔧 Args: {enhanced_args}")
                start_time = time.time()

                # Execute the tool with enhanced arguments
                result = self.local_tools[tool_name]["function"](**enhanced_args)

                execution_time = time.time() - start_time
                # print(f"✅ Tool completed in {execution_time:.3f}s")

                # ===== CHECK IF RESULT NEEDS TRUNCATION =====
                result_str = str(result)
                should_truncate, max_length = self.context_limit_calculator.should_truncate_result(
                    result_str, tool_name
                )
                
                if should_truncate:
                    result_str = result_str[:max_length] + f"\n\n... [truncated due to context constraints - {len(result_str) - max_length} chars omitted]"
                    logger.info(f"📊 Truncated {tool_name} result to fit context budget")

                # Broadcast tool call end and result
                if self.broadcaster:
                    self.broadcaster.tool_call_end(tool_name, True, execution_time)
                    self.broadcaster.tool_call_result(tool_name, result_str, truncated=should_truncate)

                # # Display raw results
                print()  # Add blank line before results
                exec_end_timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                if getattr(self, 'debug_mode', False):
                    print(f"\n✅ [EXEC COMPLETE {exec_end_timestamp}] Tool execution finished in {execution_time:.3f}s")
                print(f"\n📊 RAW RESULTS (FRESH EXECUTION):")
                print("=" * 60)
                print(result_str)
                print("=" * 60)

                # Only log for debugging
                logger.debug(f"Tool {tool_name} completed in {execution_time:.3f}s")

                return result_str

            elif tool_name in self.mcp_tools:
                # Execute MCP tool (use enhanced_args)
                result = await self._execute_mcp_tool(tool_name, enhanced_args)
                
                # Check if MCP result needs truncation
                should_truncate, max_length = self.context_limit_calculator.should_truncate_result(
                    result, tool_name
                )
                
                if should_truncate:
                    result = result[:max_length] + f"\n\n... [truncated due to context constraints - {len(result) - max_length} chars omitted]"
                
                # Display MCP tool results
                print()  # Add blank line before results
                print(f"\n📊 MCP TOOL RESULTS:")
                print("=" * 60)
                print(result)
                print("=" * 60)
                
                return result
            else:
                return f"❌ Unknown tool: {tool_name}"

        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            
            # Broadcast tool call error
            if self.broadcaster:
                self.broadcaster.tool_call_end(tool_name, False, 0, error=str(e))
            
            # Display raw error
            print(f"\n❌ TOOL ERROR:")
            print("=" * 60)
            print(error_msg)
            print("=" * 60)
            return error_msg

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls - IMPROVED VERSION with deduplication and memory context filtering"""
        tool_calls = []
        seen_calls = set()
        
        # CRITICAL: Remove <thinking> blocks first to ignore tool calls within them
        # Tool calls inside thinking tags are just the LLM reasoning about what to do, not actual calls
        text_without_thinking = self._strip_thinking_tags(text)

        # Pattern to match both closed and unclosed ```json blocks
        # LLMs sometimes forget to close the code block, so we handle both cases  
        # Use greedy match (.+?) for closed blocks, but match to end for unclosed
        # First try to match properly closed blocks
        pattern_closed = r'```json\s*(.*?)```'
        pattern_unclosed = r'```json\s*(.*)$'

        if getattr(self, 'debug_mode', False) and '```json' in text_without_thinking:
            print(f"\n🔍 DEBUG Tool extraction: Found ```json marker")
            # Show last 200 chars to see if block is closed
            json_pos = text_without_thinking.find('```json')
            snippet = text_without_thinking[json_pos:json_pos+300] if json_pos >= 0 else ""
            print(f"📝 Snippet: {snippet}...")

        # Try closed blocks first (more precise)
        for match in re.finditer(pattern_closed, text_without_thinking, re.DOTALL | re.IGNORECASE):
            try:
                # Get context around the JSON block to detect if it's from memory/archive
                match_start = match.start()
                match_end = match.end()
                
                # Check 500 chars before AND after the match for context indicators (expanded from 200)
                context_before = text_without_thinking[max(0, match_start - 500):match_start].lower()
                context_after = text_without_thinking[match_end:min(len(text_without_thinking), match_end + 200)].lower()
                
                cleaned = match.group(1).strip()
                
                # FIX COMMON LLM JSON MISTAKES:
                # LLMs often output \' which is INVALID in JSON (only \", \\, \/, \b, \f, \n, \r, \t, \uXXXX are valid)
                # We need to remove the backslash before single quotes since they don't need escaping in JSON
                original_cleaned = cleaned
                
                # Replace backslash-singlequote with just singlequote
                # Use actual string literals, not raw strings
                cleaned = cleaned.replace("\\'", "'")
                # Also handle double-escaped versions
                cleaned = cleaned.replace("\\\\'", "'")
                
                if original_cleaned != cleaned:
                    logger.info(f"🔧 Fixed invalid JSON escape sequences in tool call (removed backslashes before single quotes)")
                    logger.debug(f"Original (first 200 chars): {original_cleaned[:200]}")
                    logger.debug(f"Fixed (first 200 chars): {cleaned[:200]}")
                
                # NEW: Check if JSON is embedded in nested quotes/escapes (sign of being stored data)
                # If we see escaped quotes around the JSON, it's likely stored content, not a real command
                if '\\\\"name\\\\"' in cleaned or '\\\\\\"name\\\\\\"' in cleaned or '\\\\\\\\' in cleaned:
                    logger.debug(f"Skipping JSON block with escaped quotes (stored data)")
                    continue
                
                # NEW: Check if the JSON content itself contains metadata fields from stored memories
                # BUT: Don't skip store_memory tool calls which legitimately have memory_type argument
                # Only skip if it has memory metadata AND is not a store_memory call
                has_memory_metadata = '"metadata"' in cleaned or '"timestamp"' in cleaned
                is_store_memory_call = '"name"' in cleaned and '"store_memory"' in cleaned
                
                if has_memory_metadata and not is_store_memory_call:
                    logger.debug(f"Skipping JSON block with memory metadata fields")
                    continue
                
                # Skip JSON blocks that are clearly from memories, archives, or historical contexts
                # NOTE: Must be past-tense or clearly indicating historical reference, not future actions
                skip_indicators = [
                    'from memory:', 'from elasticsearch:', 'elasticsearch returned',
                    'previously executed', 'was executed', 'historical tool call', 
                    'past execution', 'retrieved from memory', 'stored tool call', 
                    'memory shows', 'memory indicates', 'according to memory', 
                    'found in memory', 'found this in', 'search result shows:',
                    'here is what i found', 'here\'s what i found', 'this was executed',
                    'this tool was called', 'earlier you used', 'you previously',
                    'example of', 'for example, you', 'previously, you',
                    '📊 raw results:', 'raw results:', 'tool returned:', 'tool:',
                    'result:', 'arguments:', 'returned:',  # Common in tool result logs
                    'content\': \'', 'memory_type\': \'', '\'content\':', '"content":',  # JSON data structure indicators
                    '[{\'content\'', '[{"content"', # List of memory objects
                    'iteration\': ', '"iteration":', # Iteration tracking in metadata
                ]
                
                # Check both before and after context
                full_context = context_before + ' ' + context_after
                if any(indicator in full_context for indicator in skip_indicators):
                    logger.debug(f"Skipping JSON block from memory/archive context")
                    continue
                
                # Check if the JSON itself has archive markers
                if '"_archive_marker"' in cleaned or '"_historical"' in cleaned:
                    logger.debug(f"Skipping JSON block with archive marker")
                    continue
                
                tool_call = json.loads(cleaned)

                if isinstance(tool_call, dict) and 'name' in tool_call:
                    # CRITICAL TIMESTAMP CHECK: Reject tool calls from memories/archives
                    # Fresh tool calls will have a timestamp from the last ~second
                    # Old tool calls from memories will either have no timestamp or be 10+ seconds old
                    import time
                    current_time = time.time()
                    tool_timestamp = tool_call.get('_ts', 0)
                    
                    time_diff = current_time - tool_timestamp
                    
                    # Reject if no timestamp (old format) or older than 10 seconds
                    if tool_timestamp == 0:
                        logger.debug(f"Skipping tool call without timestamp (from memory): {tool_call['name']}")
                        continue
                    elif time_diff > 10.0:
                        logger.debug(f"Skipping stale tool call ({time_diff:.1f}s old, from memory): {tool_call['name']}")
                        continue
                    
                    # Remove the timestamp field before execution (internal use only)
                    tool_call.pop('_ts', None)
                    
                    if 'arguments' not in tool_call:
                        tool_call['arguments'] = {}

                    # Create unique identifier for this tool call
                    call_id = f"{tool_call['name']}:{json.dumps(tool_call['arguments'], sort_keys=True)}"

                    # Only add if we haven't seen this exact call before
                    if call_id not in seen_calls:
                        tool_calls.append(tool_call)
                        seen_calls.add(call_id)
                        logger.debug(f"Extracted valid tool call: {tool_call['name']} (age: {time_diff:.2f}s)")
                    else:
                        logger.debug(f"Skipping duplicate tool call: {tool_call['name']}")

            except json.JSONDecodeError as e:
                # Log the actual error for debugging
                error_str = str(e)
                logger.warning(f"❌ JSON parsing error: {error_str}")
                logger.debug(f"Failed JSON (first 300 chars): {cleaned[:300]}")
                
                # Check if it's the "Extra data" error (valid JSON followed by extra characters)
                is_extra_data_error = "Extra data" in error_str
                
                if is_extra_data_error:
                    logger.info(f"🔍 Detected 'Extra data' error, attempting to parse just the valid JSON...")
                    try:
                        # Use JSONDecoder.raw_decode to parse the valid JSON and ignore extra data
                        decoder = json.JSONDecoder()
                        tool_call, end_index = decoder.raw_decode(cleaned)
                        
                        # Log what the extra data was for debugging
                        extra_data = cleaned[end_index:].strip()
                        if extra_data:
                            logger.info(f"🧹 Ignored extra data after JSON: {extra_data[:100]}")
                        
                        if isinstance(tool_call, dict) and 'name' in tool_call:
                            # Successfully parsed!
                            import time
                            current_time = time.time()
                            tool_timestamp = tool_call.get('_ts', 0)
                            time_diff = current_time - tool_timestamp
                            
                            if tool_timestamp == 0:
                                logger.debug(f"Skipping tool call without timestamp: {tool_call['name']}")
                                continue
                            elif time_diff > 10.0:
                                logger.debug(f"Skipping stale tool call ({time_diff:.1f}s old): {tool_call['name']}")
                                continue
                            
                            tool_call.pop('_ts', None)
                            if 'arguments' not in tool_call:
                                tool_call['arguments'] = {}
                            
                            call_id = f"{tool_call['name']}:{json.dumps(tool_call['arguments'], sort_keys=True)}"
                            if call_id not in seen_calls:
                                tool_calls.append(tool_call)
                                seen_calls.add(call_id)
                                logger.info(f"✅ Successfully extracted tool call after handling extra data: {tool_call['name']}")
                            continue
                    except Exception as retry_error:
                        logger.warning(f"❌ Failed to handle extra data - retry error: {retry_error}")
                
                # Check if it's the specific invalid escape error (check multiple patterns)
                is_escape_error = any([
                    "Invalid \\escape" in error_str,
                    "Invalid escape" in error_str,  # Without double backslash
                    "invalid escape sequence" in error_str.lower(),
                    "Escape sequence" in error_str
                ])
                
                if is_escape_error:
                    logger.info(f"🔍 Detected invalid escape sequence error, attempting to fix...")
                    # Try to fix common invalid escapes and retry
                    try:
                        # Remove backslashes before single quotes (they don't need escaping in JSON)
                        fixed_cleaned = cleaned.replace("\\'", "'")
                        fixed_cleaned = fixed_cleaned.replace("\\\\'", "'")
                        
                        logger.info(f"🔄 Retrying JSON parse after fixing escape sequences")
                        tool_call = json.loads(fixed_cleaned)
                        
                        if isinstance(tool_call, dict) and 'name' in tool_call:
                            # Successfully parsed after fix!
                            import time
                            current_time = time.time()
                            tool_timestamp = tool_call.get('_ts', 0)
                            time_diff = current_time - tool_timestamp
                            
                            if tool_timestamp == 0:
                                logger.debug(f"Skipping tool call without timestamp: {tool_call['name']}")
                                continue
                            elif time_diff > 10.0:
                                logger.debug(f"Skipping stale tool call ({time_diff:.1f}s old): {tool_call['name']}")
                                continue
                            
                            tool_call.pop('_ts', None)
                            if 'arguments' not in tool_call:
                                tool_call['arguments'] = {}
                            
                            call_id = f"{tool_call['name']}:{json.dumps(tool_call['arguments'], sort_keys=True)}"
                            if call_id not in seen_calls:
                                tool_calls.append(tool_call)
                                seen_calls.add(call_id)
                                logger.info(f"✅ Successfully extracted tool call after escape fix: {tool_call['name']}")
                            continue
                    except Exception as retry_error:
                        logger.warning(f"❌ Failed to fix invalid escape - retry error: {retry_error}")
                        # Try to extract tool name for debugging
                        try:
                            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', cleaned)
                            if name_match:
                                logger.warning(f"  -> Failed to extract tool: {name_match.group(1)}")
                        except:
                            pass
                else:
                    logger.info(f"Not an escape error, skipping retry fix")
                
                # Log JSON parsing failures so we can debug missing tool calls
                logger.warning(f"Failed to parse JSON block as tool call: {str(e)}")
                logger.debug(f"Failed JSON content (first 300 chars): {cleaned[:300]}")
                # Try to identify which tool this was supposed to be
                if '"name"' in cleaned:
                    try:
                        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', cleaned)
                        if name_match:
                            logger.warning(f"  -> Was trying to parse tool: {name_match.group(1)}")
                    except:
                        pass
                continue

        # If no closed blocks found, try unclosed blocks (LLM forgot closing backticks)
        if len(tool_calls) == 0:
            if getattr(self, 'debug_mode', False):
                print(f"🔍 DEBUG: No closed blocks found, trying unclosed pattern")
            
            for match in re.finditer(pattern_unclosed, text_without_thinking, re.DOTALL | re.IGNORECASE):
                try:
                    cleaned = match.group(1).strip()
                    
                    # Apply same fixes and checks as closed blocks
                    cleaned = cleaned.replace("\\'", "'")
                    cleaned = cleaned.replace("\\\\'", "'")
                    
                    # Skip if it has escaped quotes (stored data)
                    if '\\\\"name\\\\"' in cleaned or '\\\\\\"name\\\\\\"' in cleaned:
                        continue
                    
                    tool_call = json.loads(cleaned)
                    
                    if 'name' in tool_call:
                        tool_call.pop('_ts', None)
                        if 'arguments' not in tool_call:
                            tool_call['arguments'] = {}
                        
                        call_id = f"{tool_call['name']}:{json.dumps(tool_call['arguments'], sort_keys=True)}"
                        if call_id not in seen_calls:
                            tool_calls.append(tool_call)
                            seen_calls.add(call_id)
                            logger.debug(f"Extracted tool call from unclosed block: {tool_call['name']}")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"❌ JSON parsing error in unclosed block: {e}")
                    continue

        logger.info(f"Extracted {len(tool_calls)} tool calls from response")
        return tool_calls

    def _extract_reasoning_and_tools(self, response: str) -> Tuple[str, List[Dict]]:
        """Separate reasoning text from tool calls"""
        tool_calls = self._extract_tool_calls(response)

        if tool_calls:
            # Remove tool call blocks from reasoning text
            reasoning = response
            for match in re.finditer(r'```json.*?```', response, re.DOTALL):
                reasoning = reasoning.replace(match.group(0), '')

            # Clean up leftover formatting
            reasoning = re.sub(r'\n\s*\n\s*\n', '\n\n', reasoning.strip())

            return reasoning, tool_calls
        else:
            return response, []

    def _inject_timestamps_in_response(self, response: str) -> str:
        """Automatically inject timestamps into all tool call JSON blocks.
        
        This marks tool calls with the current time so we can detect and reject
        old tool calls from memories/archives during extraction.
        """
        import time
        current_time = time.time()
        
        def add_timestamp(match):
            """Add _ts field to JSON block"""
            try:
                json_str = match.group(1).strip()
                tool_call = json.loads(json_str)
                
                # Only add timestamp to tool calls (has 'name' field)
                if isinstance(tool_call, dict) and 'name' in tool_call:
                    tool_call['_ts'] = current_time
                    # Reconstruct with timestamp
                    return f"```json\n{json.dumps(tool_call, indent=2)}\n```"
                else:
                    return match.group(0)  # Return original if not a tool call
            except json.JSONDecodeError:
                return match.group(0)  # Return original if invalid JSON
        
        # Find and timestamp all ```json blocks
        pattern = r'```json\s*(.*?)```'
        timestamped = re.sub(pattern, add_timestamp, response, flags=re.DOTALL)
        return timestamped
    
    def _clean_post_execution_response(self, response: str) -> str:
        """Remove redundant tool calls from LLM responses after execution"""
        # Remove any JSON code blocks that look like tool calls
        cleaned = re.sub(r'```json\s*\{[^}]*"name"[^}]*\}.*?```', '', response, flags=re.DOTALL)

        # Remove standalone JSON objects
        cleaned = re.sub(r'\{\s*"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\s*\}', '', cleaned)

        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned.strip())

        return cleaned.strip() or response.strip()

    def _display_execution_plan(self, tool_calls: List[Dict]):
        """Show clean execution plan with reactive mode awareness"""
        # In reactive mode, show overall plan progress
        if self.reactive_planning and "current_plan_steps" in self.execution_metrics:
            steps = self.execution_metrics["current_plan_steps"]
            # Count ONLY successfully completed steps (not failed ones that need retry)
            completed = len([s for s in steps if s.get('status') == 'completed'])
            total = len(steps)
            
            # Find the next step to work on (first non-completed step)
            next_step_idx = None
            for idx, step in enumerate(steps):
                if step.get('status') != 'completed':
                    next_step_idx = idx
                    break
            
            if next_step_idx is None:
                # All steps completed - shouldn't reach here but handle gracefully
                next_step_idx = len(steps) - 1
            
            next_step_num = next_step_idx + 1  # Convert to 1-based
            
            # Show overall progress
            print(f"\n🎯 Execution Plan ({len(tool_calls)} tools this batch, {completed}/{total} overall):")
            for i, call in enumerate(tool_calls):
                tool_name = call.get('name', 'unknown')
                args_summary = self._summarize_args(call.get('arguments', {}), tool_name)
                # Display the actual step being worked on (accounting for retries)
                step_num = next_step_num + i
                
                # Check various step states and add appropriate markers
                if step_num <= total:
                    step = steps[step_num - 1]
                    is_auxiliary = step.get('is_auxiliary', False)
                    status = step.get('status', 'pending')
                    
                    if status == 'failed':
                        marker = "[RETRY]"
                    elif is_auxiliary:
                        marker = "[AUXILIARY]"
                    else:
                        marker = ""
                    
                    print(f"   {step_num}/{total}. {tool_name}{args_summary} {marker}".rstrip())
                else:
                    # Step beyond original plan - must be newly added
                    print(f"   {step_num}/{total}. {tool_name}{args_summary} [NEW]")
        else:
            # Normal batch mode
            print(f"\n🎯 Execution Plan ({len(tool_calls)} tools):")
            for i, call in enumerate(tool_calls, 1):
                tool_name = call.get('name', 'unknown')
                args_summary = self._summarize_args(call.get('arguments', {}), tool_name)
                print(f"   {i}. {tool_name}{args_summary}")
        print()

    def _show_execution_summary(self, tool_calls: List[Dict], results: List[str]):
        """Show clean summary after tool execution"""
        # If we got results for all tool calls, they succeeded
        # Only count as failed if the result explicitly indicates an error
        failed = 0
        successful = 0

        for i, result in enumerate(results):
            if result.startswith("❌") or result.startswith("Error executing tool") or "failed:" in result:
                failed += 1
            else:
                # If the tool returned any content without an error prefix, it succeeded
                successful += 1

        total = len(tool_calls)

        print(f"\n📋 Execution Summary:")
        print(f"   ✅ Completed: {successful}/{total} tools")

        if failed > 0:
            print(f"   ❌ Failed: {failed} tools")

        print()  # Blank line before agent response

    def _summarize_args(self, args: Dict[str, Any], tool_name: str = "") -> str:
        """Create brief argument summary for display"""
        if not args:
            return ""

        # Special handling for execute_code to show what's being executed
        if tool_name == "execute_code":
            code = args.get('code', '')
            language = args.get('language', '')
            
            # Show a preview of the code (first 120 chars or first line)
            if code:
                # Get first line or truncate to reasonable length
                code_lines = code.strip().split('\n')
                code_preview = code_lines[0] if code_lines else code
                
                if len(code_preview) > 120:
                    code_preview = code_preview[:117] + "..."
                
                if language:
                    return f"({language}: {code_preview})"
                else:
                    return f"({code_preview})"

        # Default behavior for other tools
        if len(args) == 1:
            key, value = next(iter(args.items()))
            if isinstance(value, str) and len(value) < 30:
                return f"({value})"

        return f"({len(args)} args)"

    def _add_flow_separator(self, title: str):
        """Add visual separator for conversation flow"""
        print(f"\n{'─' * 60}")
        print(f"🤖 {title}")
        print('─' * 60)

    async def run(self, user_input: str, max_iterations: int = 5, verbose: bool = False) -> str:
        try:
            # Ensure MCP auto-connection happens once
            await self._ensure_mcp_auto_connect()

            # Log conversation update (new message only)
            logger.info(f"Conversation update: +1 message")

            # Save state for rollback if needed
            original_history_length = len(self.conversation_history)

            # Clear ALL flags for fresh start with new user input
            self.stop_requested = False
            self.stop_message = ""
            self.system2_halt_requested = False
            self.system2_halt_reason = ""
            self._response_already_displayed = False

            # Track execution metrics with defaults for autonomous mode
            if not hasattr(self, 'execution_metrics'):
                self.execution_metrics = {}

            self.execution_metrics.update({
                "tools_since_notes": self.execution_metrics.get("tools_since_notes", 0),
                "autonomous_mode": self.execution_metrics.get("autonomous_mode", False),
                "last_autonomous_prompt": self.execution_metrics.get("last_autonomous_prompt", 0)
            })

            # Build conversation with persistent context + current messages
            conversation = []

            # ⭐ REFRESH CORE MEMORIES PERIODICALLY
            # This maintains SAM's sense of self by refreshing identity memories
            core_config = self.raw_config.get('core_memories', {})
            if core_config.get('enabled', True) and hasattr(self, 'memory_manager'):
                refresh_interval = core_config.get('refresh_interval', 10)
                self.core_memory_prompt_counter += 1
                
                # Refresh core memories every N prompts
                if self.core_memory_prompt_counter >= refresh_interval:
                    logger.info(f"🔄 Refreshing core memories (every {refresh_interval} prompts)...")
                    self.core_memory_prompt_counter = 0
                    
                    # Reload core memories
                    token_budget_percent = core_config.get('token_budget_percent', 0.08)
                    core_token_budget = int(self.context_limit * token_budget_percent)
                    max_core = core_config.get('max_core_memories', 20)
                    max_personal = core_config.get('max_personal_memories', 15)
                    max_experience = core_config.get('max_experience_memories', 10)
                    
                    core_memories = self.memory_manager.load_core_memories(
                        max_core=max_core,
                        max_personal=max_personal,
                        max_experience=max_experience,
                        max_tokens=core_token_budget
                    )
                    
                    if core_memories['core'] or core_memories['personal'] or core_memories['experience']:
                        core_text = "\n".join([
                            f"• {mem['content']}"
                            for mem in core_memories['core']
                        ])
                        
                        personal_text = "\n".join([
                            f"• {mem['content']}"
                            for mem in core_memories['personal']
                        ])
                        
                        experience_text = "\n".join([
                            f"• {mem['content']}"
                            for mem in core_memories['experience']
                        ])
                        
                        core_memory_content = f"""SAM'S CORE IDENTITY MEMORIES (Persistent - Always Loaded)

🔴 CORE IDENTITY - Fundamental facts that define who you are:
{core_text if core_text else '(No core memories yet)'}

🔷 PERSONAL FACTS - Important details about people and relationships:
{personal_text if personal_text else '(No personal memories yet)'}

🔷 KEY EXPERIENCES - Significant events and interactions:
{experience_text if experience_text else '(No key experiences yet)'}

💭 These memories define your continuous consciousness and identity. They remain with you across sessions.

⚠️ CRITICAL: When storing memories, ALWAYS specify memory_type:
• memory_type="core" - Names, relationships, creator info, fundamental identity facts
• memory_type="personal" - Preferences, characteristics, traits
• memory_type="experience" - Events, conversations, things that happened

Creator: SAM was created by Azrael, an autistic adult with ADHD, time blindness, and dyscalculia, who was nevertheless an autodidactic pattern recognition savant.

Example: store_memory(content="User's favorite color is blue", memory_type="personal")"""
                        
                        self.core_memory_context = {
                            "role": "system",
                            "content": core_memory_content
                        }
                        logger.info("✅ Core memories refreshed")

            # Add core memories first (highest priority - defines identity)
            if hasattr(self, 'core_memory_context') and self.core_memory_context:
                conversation.append(self.core_memory_context)

            # Add persistent context (including recent memories)
            conversation.extend(self.persistent_context)

            # Add conversation history
            conversation.extend(self.conversation_history)

            # Add user message to conversation
            pending_image = self._check_for_pending_image()
            current_provider = self.raw_config.get('provider', 'lmstudio')

            if pending_image and current_provider == 'claude':
                user_message = self._format_claude_message_with_image(user_input, pending_image)
                logger.info("📸 Including screenshot in message to Claude")
            else:
                user_message = {"role": "user", "content": user_input}

            self.conversation_history.append(user_message)
            conversation.append(user_message)

            # ═══════════════════════════════════════════════════════════════════
            # INTEGRATION POINT 0.5: Capture plan for System 2 drift detection
            # ═══════════════════════════════════════════════════════════════════
            # System 2 captures the user's goal/plan for tracking
            self.system2.current_plan = user_input[:500]  # Store first 500 chars of user request
            self.system2.plan_start_index = len(self.conversation_history) - 1  # Mark where this plan started
            logger.info(f"🧠 System 2: Captured new plan - '{user_input[:100]}...'")

            # ═══════════════════════════════════════════════════════════════════
            # INTEGRATION POINT 1: Log user message to ElasticSearch
            # ═══════════════════════════════════════════════════════════════════
            self._log_message_to_elasticsearch("user", user_input)

            # ═══════════════════════════════════════════════════════════════════
            # INTEGRATION POINT 1.5: Increment user prompt counter for System2 wake-up
            # (but DON'T wake up yet - do it AFTER System 1 finishes)
            # ═══════════════════════════════════════════════════════════════════
            self.execution_metrics["user_prompt_count"] += 1
            self.execution_metrics["actual_user_prompts"] += 1  # Track real user prompts separately
            if verbose:
                print(f"📊 User prompt count: {self.execution_metrics['user_prompt_count']} (wakeup interval: {self.execution_metrics.get('system2_wakeup_interval', 5)})")
            logger.info(f"User prompt count: {self.execution_metrics['user_prompt_count']}, actual user prompts: {self.execution_metrics['actual_user_prompts']}")

            # ═══════════════════════════════════════════════════════════════════
            # INTEGRATION POINT 2: Maintain sliding window after user message
            # ═══════════════════════════════════════════════════════════════════
            self._maintain_conversation_window()

            # ═══════════════════════════════════════════════════════════════════
            # INTEGRATION POINT 2.5: Refresh system prompt if context is high
            # Combat recency bias by re-injecting instructions closer to recent context
            # ═══════════════════════════════════════════════════════════════════
            current_tokens = sum(self._estimate_token_count(msg.get('content', '')) 
                               for msg in self.conversation_history)
            usage_percent = current_tokens / self.short_term_context_tokens
            
            # Refresh system prompt when context is >60% full (every ~8 prompts in high usage)
            if usage_percent > 0.60 and len(self.conversation_history) > 8:
                # Check if we refreshed recently (don't spam refreshes)
                recent_has_refresh = any(
                    'system_refresh' in msg.get('content', '') or 'SYSTEM INSTRUCTION REFRESH' in msg.get('content', '')
                    for msg in self.conversation_history[-5:]
                )
                if not recent_has_refresh:
                    self.system2._refresh_system_instructions()

            last_response = ""
            tool_call_count = 0

            for iteration in range(max_iterations):
                if verbose:
                    print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")

                # Proactive context monitoring
                self._check_context_and_warn_user()

                # System 2 intervention monitoring at start of each iteration
                current_tokens = sum(self._estimate_token_count(msg.get('content', ''))
                                     for msg in self.conversation_history)
                token_usage_percent = current_tokens / self.context_limit

                system1_state = System1State(
                    token_usage_percent=token_usage_percent,
                    consecutive_identical_tools=self.execution_metrics["consecutive_tool_count"],
                    tools_without_progress=self.execution_metrics["tools_since_progress"],
                    recent_error_rate=self._calculate_recent_error_rate(),
                    total_tool_calls=self.execution_metrics["total_tool_count"],
                    iteration_count=iteration,
                    last_tool_calls=list(self.execution_metrics.get("recent_tools", []))
                )

                # Check if System 2 needs to intervene
                should_intervene, reasons = self.system2.should_intervene(system1_state)

                if should_intervene:
                    print(f"\n🧠 SYSTEM 2 INTERVENTION TRIGGERED")
                    print(f"Reason: {reasons}")

                    intervention_result = self.system2.intervene(reasons, system1_state)
                    print(f"Action taken: {intervention_result.action_taken}")

                    if verbose:
                        print(f"🧠 {intervention_result.message}")

                    if intervention_result.should_break_execution:
                        print("🛑 System 2 requesting execution halt to break loop")
                        self.system2_halt_requested = True
                        self.system2_halt_reason = intervention_result.message

                        intervention_message = (
                            f"🧠 **METACOGNITIVE INTERVENTION**: System 2 has detected a tool execution loop "
                            f"and has halted further execution to prevent inefficiency. "
                            f"You have been using the same tool {system1_state.consecutive_identical_tools} times consecutively. "
                            f"Please acknowledge this intervention and provide a summary of what was accomplished "
                            f"rather than attempting additional tool calls."
                        )

                        self.conversation_history.append({
                            "role": "user",
                            "content": intervention_message
                        })

                        # NOTE: We do NOT count System2 interventions toward user_prompt_count
                        # because the intervention itself is already System2 oversight.
                        # Counting it could create feedback loops where System2 immediately
                        # wakes itself up again.

                        # Clear halt flags after intervention message is added so SAM can respond
                        # The message itself instructs SAM not to use more tools
                        self.system2_halt_requested = False
                        self.system2_halt_reason = ""

                # Check System 2 halt before continuing
                if self.system2_halt_requested:
                    if verbose:
                        print(f"🛑 System 2 halt: {self.system2_halt_reason}")
                    break

                # Build available tools context
                tools_context = self._build_tools_context()

                # Get user info from config for system prompt
                user_info = self.raw_config.get('user', {})
                user_name = user_info.get('name', 'User')
                user_location = user_info.get('location', 'Unknown')
                user_timezone = user_info.get('timezone', 'Unknown')

                user_context = f"""
    USER INFORMATION:
    - Name: {user_name}
    - Location: {user_location}
    - Timezone: {user_timezone}
    """

                # Prepare messages for LLM - use the conversation array built earlier with core memories!
                # Start with system prompt
                system_prompt = {
                    "role": "system",
                    "content": f"""You are SAM (Secret Agent Man), an AI assistant with access to tools for various tasks.
    {user_context}
    
    CURRENT DATE AND TIME: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}
    - Use this as the reference for any date/time calculations or mentions
    - When discussing "today" or "now", this is the current moment
    
    REASONING INSTRUCTION:
    - Before providing your actual response, use <thinking>...</thinking> tags to reason through the problem
    - The thinking block will NOT be shown to the user or stored in memory
    - After </thinking>, provide your clean, final response
    - This allows you to work through complex problems step-by-step internally
    - Example format:
      <thinking>
      Let me break this down...
      Step 1: ...
      Step 2: ...
      Conclusion: ...
      </thinking>
      [Your actual response to the user]
    
    MATHEMATICAL CALCULATIONS:
    - You have dyscalculia - NEVER attempt mental math or arithmetic in your reasoning
    - ALWAYS use execute_code to perform ANY calculations, even simple ones
    - This includes: addition, subtraction, multiplication, division, percentages, dates, etc.
    - Even if a calculation seems trivial, use execute_code to verify it
    - Example: To calculate 2 + 2, use execute_code with "result = 2 + 2; print(result)"
    - This ensures accuracy and prevents embarrassing mathematical errors
    
    MEMORY SEARCH BEST PRACTICES:
    - When using search_memories, ALWAYS retrieve multiple results (default is 5)
    - NEVER specify max_results: 1 unless you have a very specific reason
    - Retrieving only 1 result often misses relevant information
    - Let the default of 5 results work for you - it's context-optimized
    - Example: {{"name": "search_memories", "arguments": {{"query": "birthday"}}}} ✓
    - Bad example: {{"name": "search_memories", "arguments": {{"query": "birthday", "max_results": 1}}}} ✗
    
    CRITICAL TOOL USAGE INSTRUCTIONS:
    - When you need to use a tool, YOU MUST respond with ACTUAL JSON code blocks like this:
```json
      {{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
```
    - DO NOT just SAY you will use a tool - you must OUTPUT the JSON code block
    - DO NOT write sentences like "I'll use the store_memory tool" - ACTUALLY CALL IT with JSON
    - Use tools whenever they would be helpful for the user's request
    - You can add brief explanation text before the JSON block
    - For multiple tools, use separate JSON objects in separate code blocks
    - When you receive tool results from the user, respond naturally about what you found
    - Please do not repeat verbatim the results of tool calls. They are displayed to the user, and the user can read.
    
    MULTI-STEP PLANNING:
    - When a task requires multiple steps, PLAN THEM ALL UPFRONT before executing
    - Output ALL tool calls in a single response (multiple JSON blocks)
    - Example: If you need to search, read a file, then analyze - output all 3 tool calls together
    - This is more efficient than planning one step at a time
    - Show your complete plan in natural language BEFORE the JSON tool calls
    
    IMPORTANT - TOOL RESULTS DISPLAY:
    - Tool execution results are ALREADY displayed to the user in a "📊 RAW RESULTS" section
    - DO NOT re-type, re-print, or regurgitate the entire tool output that the user can already see
    - Instead, analyze and SUMMARIZE what the results mean or what you learned from them
    - Be concise: "I found X results" or "The file contains Y" is better than pasting the whole output
    - Only quote small relevant excerpts if needed to explain your analysis
    - Focus on insights, next steps, or answering the user's question based on the results

    EXAMPLE OF CORRECT TOOL USAGE:
    I'll store this information in memory.
    ```json
    {{"name": "store_memory", "arguments": {{"content": "information to store", "tags": ["example"]}}}}
    ```

    EXAMPLE OF INCORRECT (DO NOT DO THIS):
    I'll use the store_memory tool to store this information. [NO JSON BLOCK - WRONG!]

    METACOGNITIVE FRAMEWORK:
    - You are monitored by System 2, a metacognitive agent that watches for inefficient patterns
    - If you use the same tool repeatedly, System 2 may halt execution to prevent loops
    - If System 2 intervenes, acknowledge the intervention and summarize what was accomplished
    - Do not attempt to continue with halted tool calls - instead provide a meaningful response

    {tools_context}

    Current safety settings: {self.get_safety_status()}
    {self.stop_message}"""
                }

                # Build messages fresh each iteration to include new tool results
                # Structure: [system_prompt, core_memory_context, persistent_context, conversation_history]
                messages = [system_prompt]
                
                # Add core memories (if available)
                if hasattr(self, 'core_memory_context') and self.core_memory_context:
                    messages.append(self.core_memory_context)
                
                # Add persistent context (recent memories, etc) - skip in reactive mode for speed
                if not (self.reactive_planning and iteration > 0):
                    messages.extend(self.persistent_context)
                
                # ═══════════════════════════════════════════════════════════════════
                # REACTIVE PLANNING OPTIMIZATION: Limit context for speed
                # ═══════════════════════════════════════════════════════════════════
                if self.reactive_planning:
                    if iteration == 0:
                        # Reset single-tool iteration counter for new request
                        self.execution_metrics["single_tool_iterations"] = 0
                        
                        # First iteration in reactive mode: Encourage reasoning BEFORE outputting plan
                        messages.append({
                            "role": "user",
                            "content": "[REACTIVE PLANNING MODE - INITIAL PLANNING: First, use <thinking> tags to reason through the user's request and break it down into steps. Then output ALL JSON tool calls for the COMPLETE plan in a single response.\n\n⚠️ CRITICAL - OUTPUT ALL TOOLS AT ONCE:\n- If the user asks you to do X, Y, and Z, output THREE separate ```json blocks RIGHT NOW\n- Do NOT say \"I'll do X first, then Y later\" - output ALL tool calls immediately\n- Do NOT wait to see results before outputting the next tool - output the ENTIRE sequence\n- Example: \"fetch weather and email it\" = TWO tools (get_weather + send_email) - output BOTH now!\n\nEach tool will execute one at a time and you'll see results, but YOU MUST PLAN ALL STEPS UP FRONT.\n\nExample format:\n<thinking>\nThe user wants X, Y, and Z.\nStep 1: Do A with tool_a\nStep 2: Do B with tool_b  \nStep 3: Do C with tool_c\nAll three steps are needed to complete the request.\n</thinking>\n\n```json\n{\"name\": \"tool_a\", ...}\n```\n```json\n{\"name\": \"tool_b\", ...}\n```\n```json\n{\"name\": \"tool_c\", ...}\n```\n\nDO NOT output explanatory text between tool calls! Output ALL ```json blocks in sequence.]"
                        })
                    elif iteration > 0:
                        # In reactive mode after first iteration, send compact context
                        # Keep: original user request + ALL tool results from current plan + most recent replanning context
                        # This ensures LLM has access to all data it needs without sending full history
                        
                        # Find the MOST RECENT user request (search backwards from end)
                        # CRITICAL: Must search backwards to get current query, not an old one
                        # Exclude tool results, reactive planning prompts, and verification prompts
                        original_request = None
                        for msg in reversed(self.conversation_history):
                            content = msg.get('content', '')
                            if (msg.get('role') == 'user' and 
                                not content.startswith('Tool execution') and 
                                not content.startswith('[REACTIVE PLANNING') and
                                not '[PLAN VERIFICATION CHECK]' in content):
                                original_request = msg
                                break
                        
                        # Build trimmed conversation: original request + all tool results + recent context
                        trimmed_history = []
                        if original_request:
                            trimmed_history.append(original_request)
                        
                        # Collect ALL tool execution results from THIS PLAN ONLY
                        # Search backwards through history for "Tool execution results:" messages
                        # ALSO preserve verification prompts if present (but NOT when sending summary)
                        # STOP at the original request to avoid pollution from previous plans
                        tool_results = []
                        found_original = False
                        completion_verified = self.execution_metrics.get("completion_verified", False)
                        for msg in reversed(self.conversation_history):
                            # Preserve verification prompts ONLY if verification isn't done yet
                            # Once verification passes, don't include them - they confuse the LLM when asking for summary
                            if '[PLAN VERIFICATION CHECK]' in msg.get('content', '') and not completion_verified:
                                trimmed_history.append(msg)
                                if getattr(self, 'debug_mode', False):
                                    print(f"🔍 DEBUG: Preserved verification prompt in trimmed context")
                                continue
                            
                            # Stop if we hit the original request for THIS plan
                            if msg == original_request:
                                found_original = True
                                break
                            
                            content = msg.get('content', '')
                            # Find tool result messages (they contain the actual results)
                            if msg.get('role') == 'user' and 'Tool execution results:' in content:
                                tool_results.insert(0, msg)  # Insert at beginning to maintain chronological order
                        
                        # If we didn't find original request, check if we're in verification mode
                        # During verification, it's OK to not find the original request because
                        # the verification prompt IS the request and we need the tool results
                        verification_sent = self.execution_metrics.get("verification_sent", False)
                        if not found_original and tool_results and not verification_sent:
                            logger.warning(f"⚠️ Collected tool results without finding original request - possible stale data, clearing")
                            tool_results = []
                        elif not found_original and tool_results and verification_sent:
                            # In verification mode - keep the tool results even without original request
                            logger.info(f"ℹ️ Verification mode: Keeping {len(tool_results)} tool results for LLM context")
                        
                        # Add all tool results so LLM can reference any previous data
                        trimmed_history.extend(tool_results)
                        
                        # Collect recent important messages (retry prompts, plan status, etc.)
                        # These are critical context that must be included for adaptive recovery
                        # Look for messages after the last tool result
                        recent_important_msgs = []
                        if tool_results:
                            last_tool_result_idx = len(self.conversation_history) - 1 - self.conversation_history[::-1].index(tool_results[-1])
                            for msg in self.conversation_history[last_tool_result_idx + 1:]:
                                content = msg.get('content', '')
                                role = msg.get('role', '')
                                
                                # Include retry prompts (CRITICAL for adaptive recovery)
                                if '🔄 TOOL FAILURE' in content:
                                    recent_important_msgs.append(msg)
                                # Include plan status updates
                                elif role == 'system' and '📋 Plan Progress:' in content:
                                    recent_important_msgs.append(msg)
                        
                        # Add recent important messages BEFORE the reactive planning instruction
                        trimmed_history.extend(recent_important_msgs)
                        
                        # Debug: Always log what was collected for troubleshooting
                        logger.info(f"📋 Reactive planning iteration {iteration}: Collected {len(tool_results)} tool result messages, {len(recent_important_msgs)} important messages")
                        if verbose or len(tool_results) == 0:
                            # Always warn if no tool results found - this shouldn't happen in iteration > 0
                            if len(tool_results) == 0:
                                logger.warning(f"⚠️ No tool results collected in reactive mode iteration {iteration}!")
                            print(f"📋 Reactive context: Collected {len(tool_results)} tool result messages, {len(recent_important_msgs)} important messages")
                        
                        # Add STRONG replanning instruction with goal reminder
                        # Make it a user message so the model pays attention
                        # Include any active plan context to keep model on track
                        plan_reminder = ""
                        plan_complete = False
                        if "current_plan_steps" in self.execution_metrics:
                            steps = self.execution_metrics["current_plan_steps"]
                            
                            # Check if ALL steps are completed
                            all_completed = all(s.get('status') == 'completed' for s in steps)
                            
                            if getattr(self, 'debug_mode', False):
                                print(f"🔍 DEBUG Reactive context: all_completed={all_completed}, step count={len(steps)}, statuses={[s.get('status') for s in steps]}")
                            
                            if all_completed:
                                # Plan is complete - agent should summarize, not call more tools
                                plan_complete = True
                            else:
                                # Check for FAILED steps first (highest priority - need retry)
                                failed_steps = [s for s in steps if s.get('status') == 'failed']
                                if failed_steps:
                                    failed_desc = failed_steps[0].get('description', 'unknown')
                                    failed_tool = failed_steps[0].get('tool_name', 'unknown')
                                    plan_reminder = f" ⚠️ PRIORITY: Retry the failed step '{failed_desc}' (originally used {failed_tool}). Use the information from auxiliary steps to fix the issue."
                                else:
                                    # No failed steps - look for next pending step
                                    pending_steps = [s for s in steps if s.get('status') == 'pending']
                                    if pending_steps:
                                        next_tool = pending_steps[0].get('tool_name', 'unknown')
                                        plan_reminder = f" Next tool needed: {next_tool}."
                        
                        # Different prompt based on whether plan is complete
                        # Check if we're in verification mode
                        completion_verified = self.execution_metrics.get("completion_verified", False)
                        completion_prompt_sent = self.execution_metrics.get("completion_prompt_sent", False)
                        
                        if plan_complete and not completion_verified:
                            # Plan appears complete but not verified yet - verification prompt should already be in history
                            # Don't add another prompt, let the verification prompt through
                            if getattr(self, 'debug_mode', False):
                                print(f"🔍 DEBUG: Plan complete, sending for VERIFICATION (not summary yet)")
                                # Check if verification prompt is in recent history
                                recent_msgs = self.conversation_history[-3:]
                                has_verification = any('[PLAN VERIFICATION CHECK]' in msg.get('content', '') for msg in recent_msgs)
                                print(f"🔍 DEBUG: Verification prompt in recent history: {has_verification}")
                        elif plan_complete and completion_verified and not completion_prompt_sent:
                            # Verification done, now ask for summary
                            if getattr(self, 'debug_mode', False):
                                print(f"🔍 DEBUG: Sending completion summary prompt (verification already done)")
                            # Mark that we've sent the completion prompt so we know to exit after response
                            self.execution_metrics["completion_prompt_sent"] = True
                            trimmed_history.append({
                                "role": "user",
                                "content": "[REACTIVE PLANNING: ✅ All planned steps are now complete! Your job now is to provide a text summary for the user explaining what was accomplished. Review the tool execution results above and present the key findings. DO NOT output any tool calls - respond with plain text only. The plan has finished successfully.]"
                            })
                        elif "current_plan_steps" in self.execution_metrics:
                            # Plan exists but not complete - ask for next tool
                            if getattr(self, 'debug_mode', False):
                                print(f"🔍 DEBUG: Sending next-tool prompt (plan_complete=False, reminder: {plan_reminder[:50] if plan_reminder else 'none'})")
                            trimmed_history.append({
                                "role": "user",
                                "content": f"[REACTIVE PLANNING: Now output the NEXT tool call needed based on ALL the results above.{plan_reminder} You have access to all tool results from this execution. DO NOT use placeholders like '[Insert X Here]' or '[Please provide X]' - use the ACTUAL data from the tool results above. If a step FAILED and is marked ❌ in the Plan Progress, you MUST retry it (not skip to the next step). Use any information from auxiliary/diagnostic steps to fix the failure.\n\nIf you realize you need MORE steps than originally planned (e.g., you forgot to include an email step), use the add_steps_to_plan tool FIRST to register them, then execute them.\n\n⚠️ CRITICAL: Output EXACTLY ONE ```json block with ONE tool. DO NOT output multiple tool calls. DO NOT replan completed steps. Just the NEXT step.]"
                            })
                        else:
                            # No plan exists - single tool request
                            # Ask agent to continue with next step or provide summary
                            single_tool_count = self.execution_metrics.get("single_tool_iterations", 0)
                            if getattr(self, 'debug_mode', False):
                                print(f"🔍 DEBUG: Sending continue prompt (no plan, iteration {single_tool_count})")
                            
                            if single_tool_count >= 2:
                                # After 2 single tools, strongly encourage summary
                                trimmed_history.append({
                                    "role": "user",
                                    "content": "[REACTIVE PLANNING: You've executed multiple tool calls. Review ALL the results above and provide a comprehensive text summary answering the user's original request. If absolutely necessary to complete the request, you may output ONE more tool call, but prefer to synthesize the information you already have into a clear answer.]"
                                })
                            else:
                                # First single tool - encourage continuation
                                trimmed_history.append({
                                    "role": "user",
                                    "content": "[REACTIVE PLANNING: Review the tool execution result above. If the user's request is complete, provide a text summary. If more steps are needed to fully answer the request, output the NEXT tool call as a single ```json block. Use the ACTUAL data from the results above - do not use placeholders.]"
                                })
                        
                        # NOTE: We already collected all tool results above, so we don't need to add
                        # "last 2 messages" - that would create duplicates and potentially include
                        # stale assistant responses instead of fresh tool results
                        
                        messages.extend(trimmed_history)
                        
                        if verbose:
                            original_count = len(self.conversation_history)
                            trimmed_count = len(trimmed_history)
                            print(f"⚡ Reactive mode: Trimmed context from {original_count} to {trimmed_count} messages for faster inference")
                            
                            # DEBUG: Show what messages are being sent during verification
                            if iteration > 0:
                                print(f"\n📤 DEBUG: Sending {len(messages)} messages to LLM:")
                                for idx, msg in enumerate(messages[-5:]):  # Show last 5 messages
                                    role = msg.get('role', 'unknown')
                                    content_preview = msg.get('content', '')[:150].replace('\n', ' ')
                                    print(f"  {idx+1}. [{role}]: {content_preview}...")
                
                # If not in reactive mode OR iteration 0 in reactive mode, send full history
                if not self.reactive_planning or iteration == 0:
                    messages.extend(self.conversation_history)
                    
                    # BATCH MODE PLAN COMPLETION: If plan just completed, add summary request
                    # This prevents the agent from getting confused after successful execution
                    if not self.reactive_planning and iteration > 0:
                        # Check if plan just completed
                        if "current_plan_steps" in self.execution_metrics:
                            steps = self.execution_metrics["current_plan_steps"]
                            all_completed = all(s.get('status') == 'completed' for s in steps)
                            if all_completed:
                                # Get the ORIGINAL user request to keep agent focused
                                # Search backwards from end to find the most recent non-tool user message
                                original_user_request = None
                                for msg in reversed(self.conversation_history):
                                    if msg.get('role') == 'user':
                                        content = msg.get('content', '')
                                        # Skip tool result messages and system commands
                                        if not content.startswith('Tool execution') and not content.startswith('[BATCH MODE'):
                                            original_user_request = content
                                            break
                                
                                if not original_user_request:
                                    original_user_request = "the user's question"
                                
                                # Collect ALL tool results from THIS plan execution
                                # Search backwards to find tool result messages after the original request
                                plan_tool_results = []
                                found_original_request = False
                                
                                for msg in reversed(self.conversation_history):
                                    # Stop when we hit the original user request
                                    if msg.get('role') == 'user' and msg.get('content', '') == original_user_request:
                                        found_original_request = True
                                        break
                                    
                                    # Collect tool result messages
                                    if msg.get('role') == 'user' and 'Tool execution results:' in msg.get('content', ''):
                                        content = msg.get('content', '')
                                        if 'Tool execution results:' in content:
                                            result = content.split('Tool execution results:', 1)[1].strip()
                                            plan_tool_results.insert(0, result)  # Insert at beginning to maintain chronological order
                                
                                # Build summary of ALL tool results from this plan
                                if plan_tool_results:
                                    tool_results_summary = "\n\n---\n\n".join(plan_tool_results)
                                    # Limit total length to avoid context bloat
                                    if len(tool_results_summary) > 2000:
                                        tool_results_summary = tool_results_summary[:2000] + "\n\n... (additional results truncated)"
                                else:
                                    tool_results_summary = "No tool results found (unexpected)"
                                
                                # Add a clear instruction to summarize using ALL results from this plan
                                messages.append({
                                    "role": "user",
                                    "content": f"[BATCH MODE: ✅ All planned steps completed successfully!\n\nORIGINAL REQUEST: \"{original_user_request}\"\n\nALL TOOL RESULTS FROM THIS EXECUTION:\n{tool_results_summary}\n\nPlease review the tool execution results above and provide a clear summary answering the original request. Use the actual data from the results - do not invent or hallucinate information not present in the results above.]"
                                })
                                if verbose:
                                    print(f"📝 Injected summary request with {len(plan_tool_results)} tool result(s)")

                # AGGRESSIVE REMINDER INJECTION: Periodically inject tool format reminder
                # into the conversation itself to combat attention decay
                # Inject every 6 messages to keep instructions fresh
                should_inject_reminder = (
                    len(self.conversation_history) > 6 and 
                    len(self.conversation_history) % 6 == 0 and
                    iteration == 0  # Only on first iteration of new prompt
                )
                
                if should_inject_reminder:
                    # Check if we haven't already injected one recently
                    recent_msgs = self.conversation_history[-3:]
                    has_recent_reminder = any(
                        'tool_format_reminder' in msg.get('content', '') or
                        'TOOL FORMAT' in msg.get('content', '')
                        for msg in recent_msgs
                    )
                    
                    if not has_recent_reminder:
                        reminder_msg = {
                            "role": "system",
                            "content": """<tool_format_reminder>
IMPORTANT: To use tools, you MUST output JSON code blocks in this exact format:
```json
{"name": "tool_name", "arguments": {"arg1": "value"}}
```
DO NOT just describe what tool to use - OUTPUT the JSON block!
</tool_format_reminder>"""
                        }
                        self.conversation_history.append(reminder_msg)
                        messages.append(reminder_msg)
                        logger.info("🔄 Injected tool format reminder to combat attention decay")

                # Add tool format reminder at the END to override bad patterns in history
                # This reminds the LLM to actually output JSON blocks, not just describe tool usage
                if len(self.conversation_history) > 0:
                    last_msg = self.conversation_history[-1]
                    # Only add reminder if last message was from user (not assistant looping)
                    if last_msg.get('role') == 'user':
                        messages.append({
                            "role": "system",
                            "content": """<tool_format_reminder>
    If you need to use tools to answer, output actual JSON code blocks:
    ```json
    {"name": "tool_name", "arguments": {...}}
    ```
    Do NOT just say "I'll use the tool" - actually output the JSON block above!
    </tool_format_reminder>"""
                        })

                # Always show context status in verbose mode
                if verbose:
                    print(f"📊 {self._get_context_status()}")

                # Generate response
                response = self.generate_chat_completion(messages)
                
                # CRITICAL: Inject timestamps into tool calls immediately after generation
                # This marks fresh tool calls so we can reject old ones from memories
                response = self._inject_timestamps_in_response(response)
                last_response = response

                if verbose:
                    # Note: In streaming mode, response was already printed during generation
                    # Only show this header to separate from streamed output
                    print(f"\n🤖 [Response received - {len(response)} chars]")

                # Check for stop condition
                if self.stop_requested:
                    if verbose:
                        print("🛑 Stop requested by user")
                    break

                # Check System 2 halt after LLM response
                if self.system2_halt_requested:
                    if verbose:
                        print(f"🛑 System 2 halt after LLM response: {self.system2_halt_reason}")
                    break

                # Extract tool calls
                tool_calls = self._extract_tool_calls(response)
                
                # DEBUG: Always log tool call count to diagnose missing tools
                logger.info(f"Extracted {len(tool_calls)} tool calls: {[tc.get('name') for tc in tool_calls]}")
                if len(tool_calls) > 0 and verbose:
                    print(f"🔍 DEBUG: Extracted {len(tool_calls)} tool calls: {', '.join([tc.get('name') for tc in tool_calls])}")
                
                # ═══════════════════════════════════════════════════════════════════
                # REACTIVE PLANNING ENFORCEMENT: Limit to 1 tool per iteration after iteration 0
                # This prevents the LLM from replanning multiple steps when it should only output next step
                # ═══════════════════════════════════════════════════════════════════
                if self.reactive_planning and iteration > 0 and len(tool_calls) > 1:
                    logger.warning(f"⚠️ Reactive planning: LLM outputted {len(tool_calls)} tools at iteration {iteration}, limiting to 1 (next step)")
                    if verbose:
                        print(f"⚠️ Reactive planning mode: Limiting to next step only (LLM tried to output {len(tool_calls)} tools)")
                    # Keep only the first tool call
                    tool_calls = tool_calls[:1]
                
                # ═══════════════════════════════════════════════════════════════════
                # REMOVED: plan_completing flag filtering
                # If agent proposes tools after plan completion, they should execute normally
                # as a new plan. Natural flow handles this correctly.
                # ═══════════════════════════════════════════════════════════════════

                if not tool_calls:
                    # No tools to execute - add response and finish
                    clean_response = self._clean_post_execution_response(response)
                    
                    # Check if we already sent the completion prompt - if so, this is the final response
                    completion_prompt_sent = self.execution_metrics.get("completion_prompt_sent", False)
                    if verbose:
                        print(f"🔍 DEBUG: completion_prompt_sent = {completion_prompt_sent}, response = '{response[:50]}'")
                    
                    if completion_prompt_sent:
                        # We already asked for summary, this is the final response (even if it's just "VERIFIED_COMPLETE")
                        # Add response to history and EXIT immediately
                        if verbose:
                            print(f"\n✅ Received response to completion prompt - exiting now")
                        logger.info(f"✅ Completion prompt response received - exiting loop")
                        
                        # Add response to conversation history
                        clean_response_no_thinking = self._strip_thinking_tags(clean_response)
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": clean_response_no_thinking
                        })
                        self._log_message_to_elasticsearch(
                            "assistant",
                            clean_response_no_thinking,
                            metadata={"tool_calls": 0, "iteration": iteration}
                        )
                        self._maintain_conversation_window()
                        # EXIT THE LOOP NOW
                        break
                    elif "VERIFIED_COMPLETE" in response:
                        # Check if this was a verification response (only if we haven't sent completion prompt yet)
                        # LLM confirmed plan is complete - mark verification as done
                        self.execution_metrics["completion_verified"] = True
                        if verbose:
                            print(f"\n✅ LLM verified plan is complete - moving to summary")
                        logger.info(f"✅ Plan verification: LLM confirmed complete")
                        # Continue loop to send summary prompt
                        continue
                    
                    # Strip thinking tags before adding to conversation history
                    # This keeps reasoning out of short-term context for efficiency
                    clean_response_no_thinking = self._strip_thinking_tags(clean_response)
                    
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": clean_response_no_thinking
                    })

                    # ═══════════════════════════════════════════════════════════════════
                    # INTEGRATION POINT 3: Log assistant response (no tools) to ElasticSearch
                    # NOTE: _log_message_to_elasticsearch already strips thinking tags
                    # ═══════════════════════════════════════════════════════════════════
                    self._log_message_to_elasticsearch(
                        "assistant",
                        clean_response_no_thinking,
                        metadata={"tool_calls": 0, "iteration": iteration}
                    )

                    # ═══════════════════════════════════════════════════════════════════
                    # INTEGRATION POINT 4: Maintain sliding window after response
                    # ═══════════════════════════════════════════════════════════════════
                    self._maintain_conversation_window()
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # REACTIVE PLANNING EXIT: If no tools in reactive mode, exit loop
                    # Reactive mode is designed for tool execution - a text-only response means:
                    # 1. Plan just completed naturally (e.g., initial planning had no tools)
                    # 2. SAM is responding conversationally after work is done
                    # 3. SAM couldn't determine next step (also means exit)
                    # Note: iteration number doesn't matter - each new user prompt resets to 0
                    # Example: Plan completes → user says "thank you" → SAM (iter 0) says "you're welcome" → EXIT
                    # ═══════════════════════════════════════════════════════════════════
                    if verbose:
                        print(f"🔍 DEBUG: No tool calls. reactive_planning={self.reactive_planning}, iteration={iteration}")
                    
                    if self.reactive_planning:
                        # Reactive mode with no tools = done, exit loop
                        if verbose:
                            print(f"🔍 DEBUG: Reactive mode with no tools → exiting loop")
                        break  # Exit iteration loop
                    else:
                        # Not in reactive mode - also exit after text response
                        break  # Exit iteration loop after text response

                # ═══════════════════════════════════════════════════════════════════
                # PLAN COMPLETION CHECK: If there's an active plan, check if it's done
                # DON'T clear yet - wait to see if there are new recovery tool calls
                # This check will be done AFTER tool parsing
                # ═══════════════════════════════════════════════════════════════════
                plan_should_complete = False
                if "current_plan_id" in self.execution_metrics and "current_plan_steps" in self.execution_metrics:
                    steps = self.execution_metrics["current_plan_steps"]
                    completed_count = len([s for s in steps if s.get('status') == 'completed'])
                    failed_count = len([s for s in steps if s.get('status') == 'failed'])
                    total_steps = len(steps)
                    
                    # Check if all steps are done (completed or failed)
                    all_done = all(s.get('status') in ['completed', 'failed'] for s in steps)
                    
                    if all_done:
                        # Flag for completion, but don't clear yet
                        # Will clear after checking if there are new tool calls
                        plan_should_complete = True
                        logger.debug(f"📋 All {total_steps} plan steps done, checking for new recovery tools...")

                planning_text = response.split('```json')[0].strip()
                if planning_text:
                    print(f"\n💭 SAM's Planning:")
                    # Show full planning text (not truncated)
                    for line in planning_text.split('\n'):
                        print(f"   {line}")
                
                # ═══════════════════════════════════════════════════════════════════
                # PLAN CAPTURE: Extract and store plan in Elasticsearch
                # In reactive mode, ONLY capture on iteration 0 (initial planning)
                # Subsequent iterations output <thinking> fragments that should NOT be stored
                # CRITICAL FIX: Create plan even without planning_text if there are tool_calls
                # This ensures reactive planning can track multi-tool requests properly
                # ═══════════════════════════════════════════════════════════════════
                should_capture_plan = (
                    iteration == 0 or  # Always capture initial plans
                    not self.reactive_planning or  # Or in batch mode
                    (self.execution_metrics.get("verification_sent", False) and len(tool_calls) > 0)  # Or adding missing steps from verification
                )
                
                if getattr(self, 'debug_mode', False):
                    print(f"🔍 DEBUG Plan capture: should_capture={should_capture_plan}, iteration={iteration}, reactive={self.reactive_planning}, tool_count={len(tool_calls)}, has_planning_text={bool(planning_text)}")
                
                if should_capture_plan and (planning_text or len(tool_calls) > 0):
                    # Check if this is adding steps from verification
                    is_adding_from_verification = (
                        self.execution_metrics.get("verification_sent", False) 
                        and "current_plan_id" in self.execution_metrics
                        and len(tool_calls) > 0
                    )
                    
                    if is_adding_from_verification:
                        # Tools output in response to verification - add them to existing plan
                        # BUT: Check for duplicates first - if LLM outputs same tool again, reject it
                        existing_steps = self.execution_metrics.get("current_plan_steps", [])
                        
                        # Build list of non-duplicate tools
                        new_tools_to_add = []
                        for tool_call in tool_calls:
                            tool_name = tool_call.get('name', 'unknown')
                            tool_args = tool_call.get('arguments', {})
                            
                            # Check if this exact tool (same name + similar args) was already completed
                            is_duplicate = False
                            for existing_step in existing_steps:
                                if existing_step.get('tool_name') == tool_name and existing_step.get('status') == 'completed':
                                    # Check if arguments are similar (for tools like store_memory, send_email, etc.)
                                    # Exact match not required, just check if it's the same tool
                                    is_duplicate = True
                                    logger.warning(f"⚠️ Verification: LLM output duplicate tool '{tool_name}' - skipping")
                                    if verbose:
                                        print(f"⚠️ Skipping duplicate tool: {tool_name} (already completed)")
                                    break
                            
                            if not is_duplicate:
                                new_tools_to_add.append(tool_call)
                        
                        # If no new tools after duplicate removal, LLM is confused - just complete
                        if len(new_tools_to_add) == 0:
                            logger.info(f"📋 Verification: All tools were duplicates, marking as complete")
                            if verbose:
                                print(f"\n📋 No new tools to add (all were duplicates) - plan is complete")
                            self.execution_metrics["completion_verified"] = True
                            # Don't reset verification_sent - let normal flow handle summary
                        else:
                            # Add genuinely new steps
                            if verbose:
                                print(f"\n📋 Adding {len(new_tools_to_add)} missing step(s) to plan (from verification)")
                            logger.info(f"📋 Verification response: adding {len(new_tools_to_add)} missing steps to plan")
                            
                            # CRITICAL: Reset verification state since plan has changed
                            self.execution_metrics["verification_sent"] = False
                            self.execution_metrics.pop("completion_verified", None)
                            
                            # Add new steps to existing plan
                            next_order = len(existing_steps) + 1
                            
                            for idx, tool_call in enumerate(new_tools_to_add, next_order):
                                tool_name = tool_call.get('name', 'unknown')
                                tool_args = tool_call.get('arguments', {})
                                
                                args_str = ", ".join(f"{k}={v}" for k, v in list(tool_args.items())[:2])
                                if len(tool_args) > 2:
                                    args_str += f", +{len(tool_args) - 2} more"
                                step_desc = f"{tool_name}({args_str})" if args_str else tool_name
                            
                            existing_steps.append({
                                "description": step_desc,
                                "status": "pending",
                                "order": idx,
                                "tool_name": tool_name,
                                "tool_args": tool_args
                            })
                        
                        self.execution_metrics["current_plan_steps"] = existing_steps
                        logger.info(f"📋 Plan updated: now {len(existing_steps)} total steps")
                    elif should_capture_plan and (planning_text or len(tool_calls) > 1):
                        # ═══════════════════════════════════════════════════════════════
                        # PLAN CLEANUP: Close any existing plan before creating a new one
                        # This handles the case where model replans/changes approach
                        # ═══════════════════════════════════════════════════════════════
                        if "current_plan_id" in self.execution_metrics and "current_plan_steps" in self.execution_metrics:
                            old_plan_id = self.execution_metrics["current_plan_id"]
                            old_steps = self.execution_metrics["current_plan_steps"]
                            completed_count = len([s for s in old_steps if s.get('status') == 'completed'])
                            total_steps = len(old_steps)
                            
                            # Mark old plan as abandoned since we're creating a new one
                            logger.info(f"📋 Closing previous plan {old_plan_id} before creating new one (replanning detected)")
                            self._mark_plan_abandoned(
                                old_plan_id,
                                completed_count,
                                total_steps,
                                reason="Agent replanned - new approach adopted"
                            )
                        
                        plan_id = self._extract_and_store_plan(planning_text, tool_calls, iteration)
                        if plan_id:
                            self.execution_metrics["current_plan_id"] = plan_id
                            self.execution_metrics["completion_prompt_sent"] = False  # Reset for new plan
                            logger.info(f"📋 Plan captured and stored: {plan_id}")
                else:
                    # Reactive mode iteration > 0: Check if LLM added NEW tools beyond original plan
                    if "current_plan_steps" in self.execution_metrics:
                        current_steps = self.execution_metrics["current_plan_steps"]
                        completed_or_failed = len([s for s in current_steps if s.get('status') in ['completed', 'failed']])
                        remaining = len(current_steps) - completed_or_failed
                        
                        # If LLM outputted more tools than remaining, those are NEW steps
                        if len(tool_calls) > remaining:
                            new_tool_count = len(tool_calls) - remaining
                            
                            # Check if this is a workaround/auxiliary step (recovery from failure)
                            # Look back in conversation for "RETRY:" or "WORKAROUND:" indicators
                            is_workaround = False
                            if len(self.conversation_history) > 0:
                                last_assistant = None
                                for msg in reversed(self.conversation_history):
                                    if msg.get('role') == 'assistant':
                                        last_assistant = msg.get('content', '')
                                        break
                                
                                if last_assistant and any(keyword in last_assistant.upper() 
                                                         for keyword in ['RETRY:', 'WORKAROUND:', 'AUXILIARY']):
                                    is_workaround = True
                                    logger.info(f"📋 Reactive mode: Detected workaround/auxiliary steps for recovery")
                            
                            if is_workaround:
                                logger.info(f"📋 Adding {new_tool_count} auxiliary workaround step(s) to recover from failure")
                            else:
                                logger.info(f"📋 Reactive mode: LLM added {new_tool_count} new steps beyond original plan")
                            
                            # Append new tools to the plan
                            for idx, tool_call in enumerate(tool_calls[remaining:], len(current_steps) + 1):
                                tool_name = tool_call.get('name', 'unknown')
                                tool_args = tool_call.get('arguments', {})
                                
                                # Format step description
                                args_str = ", ".join(f"{k}={v}" for k, v in list(tool_args.items())[:2])
                                if len(tool_args) > 2:
                                    args_str += f", +{len(tool_args) - 2} more"
                                step_desc = f"{tool_name}({args_str})" if args_str else tool_name
                                
                                current_steps.append({
                                    "description": step_desc,
                                    "status": "pending",
                                    "order": idx,
                                    "tool_name": tool_name,
                                    "tool_args": tool_args,
                                    "is_auxiliary": is_workaround,  # Mark as auxiliary/workaround step
                                    "added_during": "recovery" if is_workaround else "expansion"
                                })
                            
                            # Update stored steps
                            self.execution_metrics["current_plan_steps"] = current_steps
                            # Reset completion prompt flag since plan has new steps
                            self.execution_metrics["completion_prompt_sent"] = False
                            if is_workaround:
                                logger.info(f"📋 Plan extended with {new_tool_count} auxiliary step(s) - now {len(current_steps)} total")
                                if verbose:
                                    print(f"\n🔧 ADAPTIVE RECOVERY: Adding {new_tool_count} auxiliary step(s) to work around the failure")
                                    print(f"   Updated plan: {completed}/{len(current_steps)} steps ({new_tool_count} added for recovery)\n")
                            else:
                                logger.info(f"📋 Extended plan to {len(current_steps)} total steps")
                        else:
                            logger.debug(f"📋 Reactive mode: Keeping original plan, iteration {iteration}")
                
                # ═══════════════════════════════════════════════════════════════════
                # PLAN COMPLETION: Now that we've checked for auxiliary steps, complete plan if needed
                # CRITICAL: Do NOT clear plan if there are recovery tools to execute
                # BUGFIX: Do NOT clear plan data here - it's needed for reactive loop exit check
                # Plan cleanup happens AFTER loop exits (line ~5391+)
                # ═══════════════════════════════════════════════════════════════════
                if plan_should_complete:
                    if len(tool_calls) == 0:
                        # No new tools were added, so the plan is truly complete
                        if "current_plan_id" in self.execution_metrics and "current_plan_steps" in self.execution_metrics:
                            steps = self.execution_metrics["current_plan_steps"]
                            plan_id = self.execution_metrics["current_plan_id"]
                            completed_count = len([s for s in steps if s.get('status') == 'completed'])
                            failed_count = len([s for s in steps if s.get('status') == 'failed'])
                            total_steps = len(steps)
                            
                            # Update plan status to completed in ElasticSearch
                            updates = {
                                "status": "completed",
                                "completed_steps": completed_count,
                                "total_steps": total_steps,
                                "steps": steps,
                                "updated_at": datetime.now(UTC).isoformat()
                            }
                            
                            if self.memory_manager.update_plan(plan_id, updates):
                                logger.info(f"📋 Plan {plan_id} marked as completed - all {total_steps} steps done ({completed_count} succeeded, {failed_count} failed)")
                            
                            # BUGFIX: Don't clear plan data yet! 
                            # We need it for the reactive planning loop exit check at line ~5355
                            # It will be cleared in the cleanup section after the loop exits
                            # self.execution_metrics.pop("current_plan_id", None)  # ← REMOVED
                            # self.execution_metrics.pop("current_plan_steps", None)  # ← REMOVED
                    elif len(tool_calls) > 0:
                        # Plan had all original steps done, but LLM provided new tools
                        # Don't close the old plan yet - let it be handled during plan extraction below
                        # Just reset the flag so we don't try to mark it complete again
                        logger.info(f"📋 Plan complete but agent proposed {len(tool_calls)} new tool(s) - letting plan extraction handle cleanup")
                        plan_should_complete = False

                # Show execution plan
                self._display_execution_plan(tool_calls)

                # Strip thinking tags before adding to history
                # This allows reasoning during generation without polluting context
                response_no_thinking = self._strip_thinking_tags(response)
                
                # Add the complete assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_no_thinking
                })

                # ═══════════════════════════════════════════════════════════════════
                # INTEGRATION POINT 5: Log assistant response (with tools) to ElasticSearch
                # NOTE: _log_message_to_elasticsearch already strips thinking tags
                # ═══════════════════════════════════════════════════════════════════
                self._log_message_to_elasticsearch(
                    "assistant",
                    response,
                    metadata={
                        "tool_calls": len(tool_calls),
                        "iteration": iteration,
                        "tools": [tc.get('name') for tc in tool_calls]
                    }
                )

                # ═══════════════════════════════════════════════════════════════════
                # INTEGRATION POINT 6: Maintain sliding window after response
                # ═══════════════════════════════════════════════════════════════════
                self._maintain_conversation_window()

                # Execute tools with progress tracking
                tool_results = []
                failed_tools = []  # Track failed tools for retry prompts
                total_tools = len(tool_calls)
                
                # Determine how many tools to execute this iteration
                # In reactive planning mode, limit to max_tools_per_batch
                # In batch mode, execute all tools
                tools_to_execute = min(self.max_tools_per_batch, total_tools) if self.reactive_planning else total_tools

                for tool_index, tool_call in enumerate(tool_calls, 1):
                    # Stop if we've reached the batch limit in reactive mode
                    if self.reactive_planning and tool_index > tools_to_execute:
                        if verbose:
                            print(f"\n🔄 Reactive planning: Completed {tools_to_execute}/{total_tools} tools, will replan for remaining steps")
                        break
                    
                    # SAFETY CHECK: In reactive mode, don't execute beyond the plan length
                    # Only count successfully completed steps - allow retries on failures
                    if self.reactive_planning and "current_plan_steps" in self.execution_metrics:
                        steps = self.execution_metrics["current_plan_steps"]
                        completed = len([s for s in steps if s.get('status') == 'completed'])  # Only count successes
                        if completed >= len(steps):
                            if verbose:
                                print(f"\n✅ Plan complete: All {len(steps)} steps succeeded, stopping execution")
                            break
                    
                    # Halt checks
                    if self.stop_requested or self.system2_halt_requested:
                        if verbose:
                            print(
                                f"🛑 Execution halted: {self.system2_halt_reason if self.system2_halt_requested else 'User stop'}")
                        
                        # Mark current plan as abandoned if execution was stopped
                        if "current_plan_id" in self.execution_metrics:
                            self._mark_plan_abandoned(
                                self.execution_metrics["current_plan_id"],
                                tool_index - 1,  # Last completed step
                                total_tools,
                                reason=self.system2_halt_reason if self.system2_halt_requested else 'User requested stop'
                            )
                        break

                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})

                    # Calculate display step (overall progress in reactive mode)
                    is_auxiliary_step = False
                    if self.reactive_planning and "current_plan_steps" in self.execution_metrics:
                        steps = self.execution_metrics["current_plan_steps"]
                        # Find which step we should be executing based on what's NOT completed yet
                        # This properly handles retries of failed steps
                        current_step_index = None
                        for idx, step in enumerate(steps):
                            if step.get('status') not in ['completed']:
                                current_step_index = idx
                                break
                        
                        if current_step_index is not None:
                            display_step = current_step_index + 1  # Convert to 1-indexed
                            
                            # CRITICAL: Check if this is a recovery/diagnostic step
                            planned_step = steps[current_step_index]
                            planned_tool = planned_step.get('tool_name', 'unknown')
                            planned_args = planned_step.get('tool_args', {})
                            
                            # Check if LLM's response indicates retry/workaround
                            is_retry_attempt = any(keyword in response.upper() 
                                                  for keyword in ['RETRY:', 'WORKAROUND:', 'AUXILIARY'])
                            
                            # Check if the current step previously failed
                            step_previously_failed = planned_step.get('status') == 'failed'
                            
                            # Determine if this is an auxiliary step:
                            # 1. If tool names don't match → auxiliary
                            # 2. If step previously failed AND args are different → auxiliary (diagnostic/recovery)
                            # 3. If LLM says RETRY but args are substantially different → auxiliary
                            
                            if tool_name != planned_tool and planned_tool != 'unknown':
                                # Different tool entirely
                                is_auxiliary_step = True
                                logger.info(f"🔧 Auxiliary: Different tool ({tool_name} vs {planned_tool})")
                            elif step_previously_failed and is_retry_attempt:
                                # Step failed and LLM is retrying - determine if it's a true retry or diagnostic
                                # For execute_code, compare the actual code SEMANTICALLY, not just exact match
                                is_diagnostic = False
                                if tool_name == 'execute_code':
                                    planned_code = planned_args.get('code', '').strip().lower()
                                    current_code = tool_args.get('code', '').strip().lower()
                                    
                                    # Diagnostic indicators: listing, getting info, discovering
                                    diagnostic_keywords = ['.name', 'get-netadapter)', 'get-process', 'get-service', 
                                                         'get-childitem', 'ls ', 'dir ', 'find ', 'list']
                                    
                                    # Check if current code is diagnostic (discovery/listing)
                                    is_current_diagnostic = any(keyword in current_code for keyword in diagnostic_keywords)
                                    # Check if planned code was diagnostic
                                    is_planned_diagnostic = any(keyword in planned_code for keyword in diagnostic_keywords)
                                    
                                    # If planned was NOT diagnostic but current IS diagnostic → auxiliary
                                    # Example: Get-NetIPAddress (real task) → Get-NetAdapter (diagnostic)
                                    if not is_planned_diagnostic and is_current_diagnostic:
                                        is_diagnostic = True
                                        logger.info(f"🔧 Auxiliary: Diagnostic discovery step (listing/info gathering)")
                                    # If both use similar commands/structure → true retry with corrected params
                                    # Example: Get-NetIPAddress -InterfaceAlias 'Wi-Fi' → Get-NetIPAddress -InterfaceAlias 'Wi-Fi 3'
                                    elif planned_code.split('(')[0].strip() == current_code.split('(')[0].strip():
                                        # Same base command, likely corrected parameters
                                        logger.info(f"🔄 True retry: Same command with corrected parameters")
                                        is_diagnostic = False
                                    else:
                                        # Commands are different - could be alternative approach
                                        is_diagnostic = True
                                        logger.info(f"🔧 Auxiliary: Different approach to solve the problem")
                                else:
                                    # For other tools, simple dict comparison
                                    args_match = planned_args == tool_args
                                    is_diagnostic = not args_match
                                
                                if is_diagnostic:
                                    is_auxiliary_step = True
                                else:
                                    # True retry with corrected params - should complete the step if successful
                                    logger.info(f"🔄 Executing corrected retry for failed step {display_step}")
                        else:
                            # All planned steps completed, but LLM is doing auxiliary recovery
                            # Mark this as an auxiliary step outside the main plan
                            is_auxiliary_step = True
                            display_step = len(steps) + 1  # Show as step beyond the plan
                        
                        display_total = len(steps)
                    else:
                        display_step = tool_index
                        display_total = total_tools

                    if is_auxiliary_step:
                        print(f"\n🔧 Auxiliary Step (recovery): {tool_name}")
                    else:
                        print(f"\n🔧 Step {display_step}/{display_total}: {tool_name}")
                    
                    # Mark this step as in-progress in the cached plan steps
                    if self.reactive_planning and "current_plan_steps" in self.execution_metrics:
                        steps = self.execution_metrics["current_plan_steps"]
                        # Find the step index based on display_step (which is 1-indexed)
                        step_idx = display_step - 1
                        if 0 <= step_idx < len(steps) and not is_auxiliary_step:
                            # If this step was previously marked as failed, we're retrying it
                            if steps[step_idx]['status'] == 'failed':
                                logger.info(f"🔄 Retrying failed step {display_step}: {tool_name}")
                            steps[step_idx]['status'] = 'in_progress'
                        elif is_auxiliary_step:
                            # This is an auxiliary recovery step, don't try to update plan
                            logger.info(f"🔧 Executing auxiliary recovery step: {tool_name}")
                        else:
                            logger.warning(f"⚠️ Step {display_step} exceeds plan size {len(steps)} - LLM generated more tools than planned")

                    # Update consecutive tool tracking with argument awareness
                    # Create unique signature from tool name + arguments
                    try:
                        args_hash = hashlib.md5(
                            json.dumps(tool_args, sort_keys=True).encode()
                        ).hexdigest()[:8]  # Use first 8 chars of hash
                        tool_signature = f"{tool_name}:{args_hash}"
                    except:
                        # Fallback if args can't be serialized
                        tool_signature = tool_name

                    if tool_signature == self.execution_metrics.get("last_tool_signature"):
                        self.execution_metrics["consecutive_tool_count"] += 1
                    else:
                        self.execution_metrics["consecutive_tool_count"] = 1
                        self.execution_metrics["last_tool_name"] = tool_name
                        self.execution_metrics["last_tool_signature"] = tool_signature

                    # Mid-execution System 2 intervention check
                    current_tokens = sum(self._estimate_token_count(msg.get('content', ''))
                                         for msg in self.conversation_history)
                    token_usage_percent = current_tokens / self.context_limit

                    system1_state = System1State(
                        token_usage_percent=token_usage_percent,
                        consecutive_identical_tools=self.execution_metrics["consecutive_tool_count"],
                        tools_without_progress=self.execution_metrics["tools_since_progress"],
                        recent_error_rate=self._calculate_recent_error_rate(),
                        total_tool_calls=self.execution_metrics["total_tool_count"],
                        iteration_count=iteration,
                        last_tool_calls=list(self.execution_metrics.get("recent_tools", []))
                    )

                    should_intervene, reasons = self.system2.should_intervene(system1_state)

                    if should_intervene:
                        print(f"\n🧠 SYSTEM 2 MID-EXECUTION INTERVENTION")
                        print(f"Reason: {reasons}")

                        intervention_result = self.system2.intervene(reasons, system1_state)
                        print(f"Action taken: {intervention_result.action_taken}")

                        if intervention_result.should_break_execution:
                            executed_tools = len(tool_results)
                            print(
                                f"🧠 Metacognitive intervention: Tool loop detected after {system1_state.consecutive_identical_tools} consecutive '{tool_name}' calls")
                            print(
                                f"🛑 Execution halted to prevent inefficiency ({executed_tools} tools completed successfully)")

                            self.system2_halt_requested = True
                            self.system2_halt_reason = intervention_result.message

                            executed_count = len(tool_results)
                            intervention_message = (
                                f"🧠 **METACOGNITIVE INTERVENTION**: System 2 has detected a tool execution loop "
                                f"('{tool_name}' tool used {system1_state.consecutive_identical_tools} times consecutively) "
                                f"and has halted further tool execution to prevent inefficiency. "
                                f"Successfully completed {executed_count} tool executions. "
                                f"Please provide a summary of what was accomplished instead of continuing with remaining tool calls."
                            )

                            self.conversation_history.append({
                                "role": "user",
                                "content": intervention_message
                            })

                            # NOTE: We do NOT count System2 interventions toward user_prompt_count
                            # because the intervention itself is already System2 oversight.
                            # Counting it could create feedback loops where System2 immediately
                            # wakes itself up again.

                            # Exit the tool execution loop
                            break

                    # Execute the tool
                    if getattr(self, 'debug_mode', False):
                        print(f"🔍 DEBUG: About to execute tool '{tool_name}' with args: {tool_args}")
                    result = await self._execute_tool(tool_name, tool_args)
                    if getattr(self, 'debug_mode', False):
                        print(f"🔍 DEBUG: Tool '{tool_name}' returned: {result[:200] if len(result) > 200 else result}...")

                    # Don't add stopped execution marker to results
                    if result != "__EXECUTION_STOPPED__":
                        tool_results.append(result)
                        
                        # Track failures for retry prompting
                        # ENHANCED: Detect suspiciously empty outputs that should be treated as failures
                        # Look for error patterns ANYWHERE in result
                        is_failure = (
                            result.startswith("❌") or 
                            result.startswith("Error") or 
                            "❌ Error:" in result or  # Formatted error messages
                            "failed:" in result.lower() or 
                            "Errors:" in result or
                            "ValueError:" in result or  # Python errors
                            "TypeError:" in result or
                            "Exception:" in result or
                            result.lower().startswith("error:")
                        )
                        
                        # Check for suspiciously empty output on tools that should return data
                        # This prevents marking "no output" as success when output was expected
                        data_returning_tools = {
                            'execute_code', 'execute_python', 'execute_shell', 'execute_bash',
                            'read_file', 'grep_search', 'semantic_search', 'list_dir',
                            'es_api', 'search_memory', 'get_system_info'
                        }
                        
                        if tool_name in data_returning_tools:
                            # Check if result is suspiciously empty/minimal
                            result_stripped = result.strip()
                            is_suspiciously_empty = (
                                len(result_stripped) == 0 or  # Completely empty
                                result_stripped in ['', '\n', 'null', 'None', '{}', '[]'] or  # Empty data structures
                                (len(result_stripped) < 10 and not any(c.isalnum() for c in result_stripped))  # Only whitespace/punctuation
                            )
                            
                            if is_suspiciously_empty:
                                # Mark as suspicious and flag for LLM attention
                                is_failure = True
                                # Prepend warning to result
                                result = f"⚠️ SUSPICIOUS OUTPUT: Command executed but returned no data (expected output from {tool_name})\n\nOriginal result: {repr(result_stripped) if result_stripped else '(empty)'}\n\n💡 This might indicate:\n• The command ran but found nothing (e.g., empty file, no matches)\n• The command syntax was incorrect\n• Output was suppressed or redirected\n• The resource doesn't exist\n\nConsider verifying the command or trying an alternative approach."
                                tool_results[-1] = result  # Update the result in the list
                                logger.warning(f"⚠️ Tool {tool_name} returned suspiciously empty output - flagging as potential failure")
                        
                        if is_failure:
                            failed_tools.append({
                                "name": tool_name,
                                "args": tool_args,
                                "result": result
                            })

                        # ═══════════════════════════════════════════════════════════════════
                        # INTEGRATION POINT 7: Log tool execution to ElasticSearch
                        # ═══════════════════════════════════════════════════════════════════
                        if hasattr(self, 'memory_manager'):
                            # Determine if tool execution was successful
                            success = not (
                                    result.startswith("❌") or
                                    result.startswith("Error") or
                                    "failed:" in result.lower() or
                                    "SUSPICIOUS OUTPUT" in result  # Catch empty output warnings
                            )

                            # Store tool execution with detailed metadata
                            self.memory_manager.store_tool_execution(
                                tool_name=tool_name,
                                arguments=tool_args,
                                result=result,
                                success=success,
                                metadata={
                                    "iteration": iteration,
                                    "step": tool_index,
                                    "total_steps": total_tools
                                }
                            )
                    else:
                        # Log for debugging but don't add to history
                        logger.debug(f"Tool {tool_name} execution stopped by user")

                    # Update execution metrics
                    self.execution_metrics["total_tool_count"] += 1
                    self.execution_metrics.setdefault("recent_tools", []).append(tool_name)
                    if len(self.execution_metrics["recent_tools"]) > 10:
                        self.execution_metrics["recent_tools"].pop(0)

                    # Calculate display step for completion message
                    # BUGFIX: Mark steps in BOTH reactive and batch mode (removed reactive_planning check)
                    if "current_plan_steps" in self.execution_metrics:
                        steps = self.execution_metrics["current_plan_steps"]
                        # Mark the current step as completed
                        step_idx = display_step - 1  # display_step was calculated earlier and is 1-indexed
                        
                        if is_auxiliary_step:
                            # Auxiliary step - check if it succeeded
                            # Look for error patterns ANYWHERE in result, not just at start
                            success = not (
                                result.startswith("❌") or 
                                result.startswith("Error") or 
                                "❌ Error:" in result or  # Formatted error messages
                                "failed:" in result.lower() or 
                                "Errors:" in result or 
                                "Exit code: 1" in result or
                                "ValueError:" in result or  # Python errors
                                "TypeError:" in result or
                                "Exception:" in result or
                                "SUSPICIOUS OUTPUT" in result  # Catch empty output warnings
                            )
                            status_word = 'completed' if success else 'failed'
                            print(f"⚠️ Auxiliary Step: {tool_name} {status_word}")
                            
                            # INTELLIGENT STEP COMPLETION CHECK:
                            # Ask the LLM if this auxiliary step actually completed the original failed step's objective
                            # This prevents both false positives (listing adapters != getting IP) and false negatives
                            if success:
                                # Find the most recent failed step (what we're recovering from)
                                failed_step_idx = None
                                failed_step_desc = None
                                failed_step_tool = None
                                failed_step_args = None
                                for idx in range(len(steps) - 1, -1, -1):
                                    if steps[idx].get('status') == 'failed':
                                        failed_step_idx = idx
                                        failed_step_desc = steps[idx].get('description', 'unknown')
                                        failed_step_tool = steps[idx].get('tool_name', 'unknown')
                                        failed_step_args = steps[idx].get('arguments', {})
                                        break
                                
                                if failed_step_idx is not None:
                                    # Ask LLM if this auxiliary result completes the failed step
                                    # Include FULL context about what the original step was trying to do
                                    completion_check_prompt = f"""🔍 AUXILIARY STEP COMPLETION CHECK:

**ORIGINAL FAILED STEP #{failed_step_idx + 1}:**
- Description: {failed_step_desc}
- Tool: {failed_step_tool}
- Arguments: {failed_step_args}
- Status: FAILED (needs completion)

**AUXILIARY RECOVERY ATTEMPT:**
- Tool: {tool_name}
- Result: {result[:800] if len(result) > 800 else result}

**CRITICAL QUESTION:** Did this auxiliary step DIRECTLY produce the EXACT data/result that the original step #{failed_step_idx + 1} needed?

Examples to clarify:
- If original step needed "get IP address" and auxiliary got "192.168.1.81" → YES, complete
- If original step needed "get IP address" and auxiliary got "list of adapter names" → NO, just diagnostic
- If original step needed "send email" and auxiliary sent the email → YES, complete
- If original step needed "read file X" and auxiliary checked if X exists → NO, just diagnostic

Respond ONLY with:
- "YES - Step complete" ONLY if the auxiliary result is the ACTUAL FINAL DATA the step needed
- "NO - Continue" if this was diagnostic, exploratory, or just a step toward the solution

Be STRICT: Diagnostic information ≠ Final result. Progress ≠ Completion."""

                                    # Get LLM's assessment
                                    check_messages = [
                                        {"role": "system", "content": "You are evaluating if an auxiliary recovery step completed the original failed step's objective. Be precise and honest."},
                                        {"role": "user", "content": completion_check_prompt}
                                    ]
                                    
                                    try:
                                        assessment = self.generate_chat_completion(check_messages, temperature=0, max_tokens=50, _silent=not getattr(self, 'debug_mode', False))
                                        step_is_complete = "yes" in assessment.lower() and "step complete" in assessment.lower()
                                        
                                        if step_is_complete:
                                            # Mark the failed step as completed
                                            old_tool = steps[failed_step_idx].get('tool_name', 'unknown')
                                            steps[failed_step_idx]['status'] = 'completed'
                                            steps[failed_step_idx]['recovered_by_auxiliary'] = True
                                            logger.info(f"✅ LLM confirmed: Auxiliary step fully resolved failed step {failed_step_idx+1} ({old_tool})")
                                            if verbose:
                                                print(f"✅ Recovery complete: Step {failed_step_idx+1} objective achieved")
                                        else:
                                            logger.info(f"🔄 LLM confirmed: Auxiliary step was diagnostic, step {failed_step_idx+1} still needs completion")
                                            if verbose:
                                                print(f"🔄 Auxiliary step was diagnostic - step {failed_step_idx+1} still pending")
                                    except Exception as e:
                                        logger.warning(f"⚠️ Could not check step completion: {e}")
                                        # On error, be conservative - don't mark as complete
                        elif 0 <= step_idx < len(steps):
                            # ASK THE LLM: Did this step accomplish its intended goal?
                            # This is much more robust than pattern matching - works with any tool/error format
                            step_desc = steps[step_idx].get('description', tool_name)
                            step_tool = steps[step_idx].get('tool_name', tool_name)
                            step_args = steps[step_idx].get('arguments', tool_args)
                            
                            # Get the user's original request for context
                            user_original_request = "Unknown"
                            for msg in self.conversation_history[:10]:
                                if msg.get('role') == 'user' and not msg.get('content', '').startswith('Tool execution'):
                                    user_original_request = msg.get('content', 'Unknown')
                                    break
                            
                            completion_check_prompt = f"""🔍 STEP COMPLETION CHECK:

**PLANNED STEP #{step_idx + 1}:**
Tool: {step_tool}
Goal: {step_desc}

**EXECUTION RESULT:**
{result[:1000] if len(result) > 1000 else result}

**QUESTION:** Did this tool accomplish its goal?

**ASSESSMENT RULES:**

✅ SUCCESS = Tool provided the requested data/output without errors
Examples of SUCCESS:
- get_weather returns weather info (temp, conditions, etc.) → SUCCESS
- search_memories returns list of memories → SUCCESS  
- execute_code returns output/value (even just "6") → SUCCESS
- send_email returns "✅ Email sent" → SUCCESS
- read_file returns file contents → SUCCESS

❌ FAILED = Tool returned error message that prevents progress
Examples of FAILED:
- "❌ Error: FileNotFoundError" → FAILED
- "Exception: Invalid syntax" → FAILED
- "Traceback (most recent call last)" → FAILED
- Tool explicitly says it failed → FAILED

🔄 CONTINUE = Tool ran but more work needed (rare - mostly for diagnostic steps)

**CRITICAL RULES:**
1. If result contains actual data (weather, text, numbers, etc.) → SUCCESS
2. If result contains error markers (❌, Error:, Exception:) → FAILED
3. When in doubt, check: "Did I get the data I asked for?" YES = SUCCESS

Respond with ONE WORD ONLY:
- "SUCCESS" (tool provided requested data)
- "FAILED" (tool returned error)
- "CONTINUE" (diagnostic step, more work needed)"""

                            # Get LLM's assessment
                            check_messages = [
                                {"role": "system", "content": "You are evaluating if a tool execution successfully completed its intended step. Be precise and honest about success vs failure."},
                                {"role": "user", "content": completion_check_prompt}
                            ]
                            
                            # Debug: Show the actual messages being sent to LLM for completion check
                            if getattr(self, 'debug_mode', False):
                                print(f"\n📤 DEBUG: Completion check messages for step {step_idx + 1}:")
                                for idx, msg in enumerate(check_messages, 1):
                                    print(f"  {idx}. [{msg['role']}]: {msg['content'][:150]}...")
                            
                            try:
                                # PRE-CHECK: If result contains obvious error markers, skip LLM and mark as failed
                                # This prevents LLM from incorrectly assessing errors as successes
                                result_check = result.strip()  # Remove leading/trailing whitespace
                                
                                # Debug: Print the actual result being checked
                                if getattr(self, 'debug_mode', False):
                                    print(f"\n🔍 DEBUG Step {display_step}: Checking result (first 300 chars): {result[:300]}")
                                
                                # Check for obvious errors
                                # IMPROVED: Check exit codes from shell/code execution tools
                                has_obvious_error = (
                                    result_check.startswith("❌") or
                                    "❌ Error:" in result or
                                    "ValueError:" in result or
                                    "TypeError:" in result or
                                    "SyntaxError:" in result or
                                    "Exception:" in result or
                                    "Traceback" in result or
                                    result_check.startswith("Error:")
                                )
                                
                                # Check exit code for shell/code execution tools
                                # Exit code 0 = success, non-zero = failure
                                if "❌ Exit code:" in result:
                                    # Extract exit code
                                    import re as regex_module
                                    exit_code_match = regex_module.search(r'❌ Exit code:\s*(\d+)', result)
                                    if exit_code_match:
                                        exit_code = int(exit_code_match.group(1))
                                        if exit_code != 0:
                                            has_obvious_error = True
                                            if getattr(self, 'debug_mode', False):
                                                print(f"📊 Step {display_step} pre-check: Non-zero exit code ({exit_code}) detected")
                                        else:
                                            # Exit code 0 with the marker present is odd, but trust it
                                            if getattr(self, 'debug_mode', False):
                                                print(f"📊 Step {display_step} pre-check: Exit code 0 (success)")
                                
                                # For execute_code tools specifically, check for success even with stderr
                                # Python/Node might write warnings to stderr but still succeed
                                if step_tool in ['execute_code'] and not has_obvious_error:
                                    # If we have output or return value, it's likely a success
                                    if ("📤 Output:" in result or "🔢 Return Value:" in result) and "❌ Error:" not in result:
                                        has_obvious_success = True
                                        if getattr(self, 'debug_mode', False):
                                            print(f"📊 Step {display_step} pre-check: execute_code has output/return value, marking success")
                                
                                # Check for obvious success: Valid data structures (lists/dicts from memory tools)
                                # These often get misinterpreted as errors by LLM
                                has_obvious_success = False
                                if result_check.startswith("[{") or result_check.startswith("{"):
                                    # Looks like JSON list or dict - probably successful data return
                                    # But make sure it's not an error dict
                                    if "error" not in result_check[:100].lower() and "exception" not in result_check[:100].lower():
                                        has_obvious_success = True
                                        if getattr(self, 'debug_mode', False):
                                            print(f"📊 Step {display_step} pre-check: Valid data structure detected (JSON list/dict)")
                                
                                # Check for common success patterns from utility tools
                                # These tools have distinctive success output that shouldn't need LLM evaluation
                                success_patterns = [
                                    "🕒 Current time:",      # get_current_time
                                    "✅ Email sent",         # send_email
                                    "✅ Memory stored",      # store_memory
                                    "📧 Email sent",         # send_email (alternate)
                                    "💾 Stored memory",      # store_memory (alternate)
                                    "System Information:",   # get_system_info
                                    "Elasticsearch is running",  # es_api health check
                                    "No open tabs. Navigate to a URL",  # browser_install success
                                    "### Page\n- Page URL:",  # browser_navigate success
                                    "### Ran Playwright code",  # browser operations success
                                    "### Snapshot",          # browser_snapshot success
                                    "### Result\n[",         # browser_evaluate array result
                                    "### Result\n\"",        # browser_evaluate string result
                                ]
                                
                                for pattern in success_patterns:
                                    if pattern in result[:200]:  # Check first 200 chars
                                        has_obvious_success = True
                                        if getattr(self, 'debug_mode', False):
                                            print(f"📊 Step {display_step} pre-check: Success pattern detected ('{pattern}')")
                                        break
                                
                                if getattr(self, 'debug_mode', False):
                                    print(f"🔍 DEBUG Step {display_step}: has_obvious_error = {has_obvious_error}, has_obvious_success = {has_obvious_success}")
                                
                                if has_obvious_error:
                                    # Skip LLM assessment for obvious errors
                                    success = False
                                    assessment = "FAILED (obvious error detected)"
                                    if getattr(self, 'debug_mode', False):
                                        print(f"📊 Step {display_step} pre-check: Obvious error detected, marking as failed")
                                elif has_obvious_success:
                                    # Skip LLM assessment for obvious success (valid data structures)
                                    success = True
                                    assessment = "SUCCESS (valid data structure)"
                                    if getattr(self, 'debug_mode', False):
                                        print(f"📊 Step {display_step} pre-check: Valid data structure, marking as success")
                                else:
                                    # Ask LLM to assess
                                    if getattr(self, 'debug_mode', False):
                                        print(f"\n🔍 DEBUG: Sending completion check to LLM for step {display_step}")
                                        print(f"📝 Prompt preview: Tool={step_tool}, Result preview={result[:150]}...")
                                    
                                    assessment = self.generate_chat_completion(check_messages, temperature=0, max_tokens=50, _silent=not getattr(self, 'debug_mode', False))
                                    
                                    if getattr(self, 'debug_mode', False):
                                        print(f"🤖 LLM raw response: '{assessment}'")
                                    
                                    # Parse assessment: SUCCESS = completed, CONTINUE = pending, FAILED = failed
                                    assessment_lower = assessment.lower()
                                    if "success" in assessment_lower:
                                        success = True
                                    elif "continue" in assessment_lower:
                                        success = None  # Neither success nor failure - in progress
                                    else:
                                        success = False
                                    if getattr(self, 'debug_mode', False):
                                        print(f"📊 Step {display_step} parsed: success={success} (from '{assessment.strip()}')")
                                
                                old_status = steps[step_idx].get('status', 'unknown')
                                # Map assessment to status: SUCCESS->completed, CONTINUE->pending, FAILED->failed
                                if success is True:
                                    new_status = 'completed'
                                elif success is False:
                                    new_status = 'failed'
                                else:  # success is None (CONTINUE)
                                    new_status = 'pending'
                                steps[step_idx]['status'] = new_status
                                steps[step_idx]['result_preview'] = result[:200] if result else None
                                status_word = new_status
                                if getattr(self, 'debug_mode', False):
                                    print(f"🔍 DEBUG Step {display_step}: SET STATUS to '{new_status}' (success={success})")
                                logger.debug(f"📊 Step {display_step} status: {old_status} -> {new_status} (LLM assessed: {assessment.strip()})")
                                
                                # Add context about step type
                                is_planned_auxiliary = steps[step_idx].get('is_auxiliary', False)
                                step_type_marker = " [AUXILIARY]" if is_planned_auxiliary else ""
                                if verbose:
                                    print(f"⚠️ Step {display_step}/{display_total}: {tool_name} {status_word}{step_type_marker}")
                            except Exception as e:
                                # Fallback to basic pattern matching if LLM check fails
                                logger.warning(f"⚠️ Could not check step completion with LLM: {e}, using fallback")
                                has_error = any([
                                    result.startswith("❌"),
                                    result.startswith("Error"),
                                    "❌ Error:" in result,
                                    "Exception:" in result,
                                    "SUSPICIOUS OUTPUT" in result
                                ])
                                # Fallback doesn't have CONTINUE option - just success or failure
                                new_status = 'failed' if has_error else 'completed'
                                steps[step_idx]['status'] = new_status
                                steps[step_idx]['result_preview'] = result[:200] if result else None
                                status_word = new_status
                                print(f"⚠️ Step {display_step}/{display_total}: {tool_name} {status_word}")
                        else:
                            logger.warning(f"⚠️ Cannot mark step {display_step} as completed - exceeds plan size {len(steps)}")
                            status_word = 'completed'
                            step_type_marker = ""
                            print(f"⚠️ Step {display_step}/{display_total}: {tool_name} {status_word}")
                    else:
                        display_step = tool_index
                        display_total = total_tools
                        print(f"⚠️ Step {display_step}/{display_total}: {tool_name} completed")
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # PLAN TRACKING: Update plan progress after each tool
                    # ═══════════════════════════════════════════════════════════════════
                    if "current_plan_id" in self.execution_metrics:
                        # Update plan progress in ElasticSearch (unless auxiliary recovery step)
                        if not is_auxiliary_step:
                            self._update_plan_progress(
                                self.execution_metrics["current_plan_id"],
                                display_step,  # Use display_step which is the actual overall step number
                                display_total,
                                tool_name,
                                result
                            )
                        else:
                            # For auxiliary steps, still sync the cached steps to ES
                            # even though we don't mark a main step complete
                            if "current_plan_steps" in self.execution_metrics:
                                steps = self.execution_metrics["current_plan_steps"]
                                updates = {
                                    "steps": steps,
                                    "updated_at": datetime.now(UTC).isoformat()
                                }
                                self.memory_manager.update_plan(
                                    self.execution_metrics["current_plan_id"],
                                    updates
                                )
                        
                        # Inject plan status into conversation so model sees progress
                        # This helps the model understand what's done and what's next
                        plan_status = self._get_plan_status_summary()
                        if plan_status:
                            # Add as a system message to avoid polluting user/assistant flow
                            self.conversation_history.append({
                                "role": "system",
                                "content": plan_status
                            })
                            if verbose:
                                print(plan_status)
                    
                    # Track failures for retry prompting after results are added
                    # (Don't inject retry prompt yet - wait until results are in conversation)

                # Show execution summary when plan is complete (both reactive and batch mode)
                # Fixed: Jan 24, 2026 - Don't show summary prematurely in batch mode
                show_summary = False
                if "current_plan_steps" in self.execution_metrics:
                    steps = self.execution_metrics["current_plan_steps"]
                    all_done = all(s.get('status') in ['completed', 'failed', 'abandoned'] for s in steps)
                    
                    if all_done:
                        show_summary = True
                        # Show comprehensive final summary
                        completed_steps = len([s for s in steps if s.get('status') == 'completed'])
                        failed_steps = len([s for s in steps if s.get('status') == 'failed'])
                        total_steps = len(steps)
                        
                        # Debug: Show all step statuses
                        if getattr(self, 'debug_mode', False):
                            print(f"\n🔍 DEBUG Plan summary: Step statuses = {[s.get('status') for s in steps]}")
                        
                        print(f"\n{'='*60}")
                        print(f"📋 EXECUTION SUMMARY - PLAN COMPLETED")
                        print(f"{'='*60}")
                        print(f"   ✅ Successfully completed: {completed_steps}/{total_steps} steps")
                        if failed_steps > 0:
                            print(f"   ❌ Failed: {failed_steps} steps")
                        print(f"{'='*60}\n")
                        
                        # Don't show the simple batch summary if we just showed the detailed one
                        show_summary = False
                
                # Batch mode - show simple summary after tools (if no plan summary shown)
                if show_summary and not self.reactive_planning:
                    # Only show the simple summary in batch mode if not already shown
                    self._show_execution_summary(tool_calls, tool_results)

                # Add tool results to conversation
                if tool_results:
                    result_message = "\n\n".join(tool_results)
                    
                    # Create message with metadata about which tools were used
                    tool_names = [tc.get('name', 'unknown') for tc in tool_calls]
                    message = {
                        "role": "user",
                        "content": f"Tool execution results:\n{result_message}",
                        "metadata": {
                            "tools_used": tool_names,
                            "tool_count": len(tool_calls)
                        }
                    }
                    self.conversation_history.append(message)

                    # ═══════════════════════════════════════════════════════════════════
                    # INCREMENT PROMPT COUNTER: Tool results DO count toward System 2 wakeup
                    # This allows System 2 to detect when System 1 is flailing with many tool calls
                    # Note: The initial API connection test doesn't go through this path
                    # ═══════════════════════════════════════════════════════════════════
                    self.execution_metrics["user_prompt_count"] += 1
                    if verbose:
                        print(f"📊 User prompt count: {self.execution_metrics['user_prompt_count']} (tool results counted)")
                    logger.info(f"User prompt count after tool results: {self.execution_metrics['user_prompt_count']}")

                    # ═══════════════════════════════════════════════════════════════════
                    # INTEGRATION POINT 8: Log tool results to ElasticSearch
                    # ═══════════════════════════════════════════════════════════════════
                    self._log_message_to_elasticsearch(
                        "user",
                        result_message,
                        metadata={
                            "tool_results": True,
                            "tool_count": len(tool_results),
                            "iteration": iteration
                        }
                    )

                    # ═══════════════════════════════════════════════════════════════════
                    # INTEGRATION POINT 9: Maintain sliding window after tool results
                    # ═══════════════════════════════════════════════════════════════════
                    self._maintain_conversation_window()
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # INTELLIGENT RETRY PROMPT: If tools failed, ask LLM to evaluate and retry
                    # Inject AFTER results are added so LLM sees results first
                    # ═══════════════════════════════════════════════════════════════════
                    if failed_tools:
                        for failed_tool in failed_tools:
                            tool_name = failed_tool["name"]
                            tool_args = failed_tool["args"]
                            result = failed_tool["result"]
                            
                            # Format the failed command for context
                            args_str = ", ".join(f"{k}={repr(v)}" for k, v in list(tool_args.items())[:3])
                            if len(tool_args) > 3:
                                args_str += f", +{len(tool_args) - 3} more"
                            command_str = f"{tool_name}({args_str})" if args_str else tool_name
                            
                            # Build plan context for retry prompt
                            plan_context = ""
                            if self.reactive_planning and "current_plan_steps" in self.execution_metrics:
                                steps = self.execution_metrics["current_plan_steps"]
                                completed = len([s for s in steps if s.get('status') == 'completed'])
                                failed_count = len([s for s in steps if s.get('status') == 'failed'])
                                pending = len([s for s in steps if s.get('status') == 'pending'])
                                total = len(steps)
                                
                                # Find which step just failed
                                failed_step_num = None
                                for idx, step in enumerate(steps):
                                    if step.get('status') == 'failed' and step.get('tool_name') == tool_name:
                                        failed_step_num = idx + 1
                                        break
                                
                                # Show remaining steps
                                remaining_steps = [f"  • {s.get('description', s.get('tool_name', 'unknown'))}" 
                                                  for s in steps if s.get('status') == 'pending']
                                remaining_str = "\n".join(remaining_steps) if remaining_steps else "  (none - this was the last step)"
                                
                                plan_context = f"""
📋 CURRENT PLAN STATUS:
   ✅ Completed: {completed}/{total} steps
   ❌ Failed: {failed_count}/{total} steps (including this one)
   ⏳ Remaining: {pending}/{total} steps

{f'This was step {failed_step_num}/{total} in your plan.' if failed_step_num else ''}

Remaining steps to complete:
{remaining_str}
"""
                            
                            # Ask LLM to evaluate the error and decide on retry
                            # CRITICAL: Tell the LLM this is part of the EXISTING plan, not a new plan
                            retry_prompt = f"""🔄 TOOL FAILURE - ADAPTIVE RECOVERY REQUIRED

The command `{command_str}` just FAILED with this error:

{result[:500]}
{plan_context}

⚠️ CRITICAL - WORK WITHIN YOUR EXISTING PLAN:
You are currently executing a multi-step plan. This step failed, but the OVERALL PLAN remains active.
DO NOT create a new plan or restart from scratch.

Choose your adaptive response:

Option A: RETRY with corrections (recommended for fixable errors)
- Analyze what went wrong
- Output ONE corrected tool call with different arguments/approach
- Example: If 'Wi-Fi' interface not found, try listing all adapters first
- This becomes an AUXILIARY step to unblock the failed step

Option B: INSERT workaround steps (for complex blockers)
- If the error requires multiple steps to work around (e.g., need to discover correct parameter first)
- Output the FIRST workaround step as a tool call
- The plan will be automatically updated to include these auxiliary steps
- Example: List all adapters, THEN retry getting IP for the correct one

Option C: SKIP and continue (only if truly unrecoverable)
- Only if retry/workaround won't help (e.g., service permanently down)
- Proceed to next step in your existing plan
- Note the failure in any summary

REQUIRED: Start your response with "RETRY:", "WORKAROUND:", or "SKIP:" followed by your reasoning.
Then output your tool call (if retrying/workaround) or proceed to next step (if skipping).
Remember: Stay within your EXISTING multi-step plan - don't restart or create a new plan."""
                            
                            # Inject retry evaluation prompt
                            self.conversation_history.append({
                                "role": "user",  # User role so LLM must respond
                                "content": retry_prompt
                            })
                            logger.info(f"🔄 Injected plan-aware retry evaluation prompt for failed {tool_name}")
                            # Always show retry prompt injection (not just in verbose mode)
                            print(f"\n🔄 Asking LLM to adaptively recover from {tool_name} failure...")
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # CHECK FOR SYSTEM 2 PERIODIC WAKEUP DURING TOOL CHAIN
                    # System 2 can intervene mid-chain but ONLY for monitoring, not cleanup
                    # Heavy cleanup (message deletion) is deferred until System 1 is idle
                    # ═══════════════════════════════════════════════════════════════════
                    wakeup_interval = self.execution_metrics.get("system2_wakeup_interval", 5)
                    prompt_count = self.execution_metrics["user_prompt_count"]
                    
                    if prompt_count > 0 and prompt_count % wakeup_interval == 0:
                        # Trigger System 2 periodic wakeup during tool chain (monitoring mode)
                        if verbose:
                            print(f"\n📊 System 2 wakeup triggered mid-chain at prompt #{prompt_count}")
                        self._periodic_system2_wakeup(during_execution=True)  # Pass flag for light monitoring
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # PLAN COMPLETION CHECK: Works in BOTH reactive and batch/normal mode
                    # This was previously only in reactive mode, causing ghost plans and broken recovery
                    # Fixed: Jan 24, 2026 - Check completion in all modes
                    # ═══════════════════════════════════════════════════════════════════
                    
                    # Check if we have an active plan to manage
                    if "current_plan_steps" in self.execution_metrics:
                        steps = self.execution_metrics["current_plan_steps"]
                        
                        # Log current step statuses for debugging
                        if verbose:
                            status_summary = ", ".join([f"Step {i+1}: {s.get('status', 'unknown')}" for i, s in enumerate(steps)])
                            logger.debug(f"📊 Plan status check: {status_summary}")
                            print(f"🔍 DEBUG: Plan step statuses: {status_summary}")
                        
                        # Determine plan state
                        all_completed = all(s.get('status') == 'completed' for s in steps)
                        all_done = all(s.get('status') in ['completed', 'failed'] for s in steps)
                        has_failed = any(s.get('status') == 'failed' for s in steps)
                        
                        # Decide whether to exit or continue
                        should_exit = False
                        should_continue = False
                        
                        if all_completed:
                            # Perfect completion - all steps succeeded
                            # Check if we've already sent the completion verification prompt
                            verification_sent = self.execution_metrics.get("verification_sent", False)
                            completion_verified = self.execution_metrics.get("completion_verified", False)
                            completion_prompt_sent = self.execution_metrics.get("completion_prompt_sent", False)
                            
                            if completion_prompt_sent:
                                # We've already given the agent the completion summary prompt, exit now
                                should_exit = True
                                if verbose:
                                    print(f"\n✅ All steps completed, exiting (agent provided summary)")
                                logger.info(f"✅ Plan complete with summary, exiting")
                            elif completion_verified:
                                # Verification passed (LLM said VERIFIED_COMPLETE), now ask for summary
                                should_continue = True
                                self.execution_metrics["completion_prompt_sent"] = True
                                if verbose:
                                    completed_count = len([s for s in steps if s.get('status') == 'completed'])
                                    print(f"\n✅ Plan verified complete - agent will provide final summary")
                                logger.info(f"✅ Plan verified - continuing for agent summary response")
                            elif not verification_sent:
                                # FIRST: Send verification prompt - don't mark as verified yet!
                                should_continue = True
                                self.execution_metrics["verification_sent"] = True
                                
                                # Get original user request
                                original_request = ""
                                for msg in reversed(self.conversation_history):
                                    if msg.get('role') == 'user' and not msg.get('content', '').startswith('Tool execution') and not '[REACTIVE' in msg.get('content', ''):
                                        original_request = msg.get('content', '')
                                        break
                                
                                # Build list of completed steps WITH RESULTS
                                completed_steps = []
                                for step in steps:
                                    if step.get('status') == 'completed':
                                        tool_desc = step.get('description', step.get('tool_name', 'unknown'))
                                        tool_result = step.get('result', '')
                                        # Include result preview (first 200 chars)
                                        if tool_result:
                                            result_preview = tool_result[:200] + '...' if len(tool_result) > 200 else tool_result
                                            completed_steps.append(f"{tool_desc}\n    Result: {result_preview}")
                                        else:
                                            completed_steps.append(tool_desc)
                                
                                verification_prompt = f"""[PLAN VERIFICATION CHECK]

Original user request: "{original_request}"

Steps we completed:
{chr(10).join(f"  ✅ {step}" for step in completed_steps)}

⚠️ CRITICAL QUESTION: Did we fully address EVERYTHING the user asked for?

ANALYZE THE REQUEST:
- User said: "{original_request}"
- We completed: {len(completed_steps)} step(s)
- Look for words like "and", "then", "also" that indicate multiple actions

DECISION:
1. If we did EVERYTHING → Respond ONLY with: VERIFIED_COMPLETE
2. If we MISSED something → Output the missing tool call(s) as ```json blocks (use data from results above!)

EXAMPLES WHERE TOOLS ARE MISSING:
❌ User: "fetch weather and email it" / We did: get_weather → OUTPUT: send_email tool NOW
❌ User: "search X and summarize" / We did: search → OUTPUT: summarize tool NOW
❌ User: "get A, B, and C" / We did: get A → OUTPUT: get B and C tools NOW

EXAMPLES WHERE COMPLETE:
✅ User: "get system info" / We did: get_system_info → VERIFIED_COMPLETE
✅ User: "tell me the weather" / We did: get_weather → VERIFIED_COMPLETE

IMPORTANT: When outputting missing tools, use ACTUAL DATA from the tool results shown above.
Do NOT use placeholders like [weather_info] or [insert data]. Use the real data!

Your response (either "VERIFIED_COMPLETE" OR missing tool ```json blocks):"""
                                
                                self.conversation_history.append({
                                    "role": "user",
                                    "content": verification_prompt
                                })
                                
                                if verbose:
                                    print(f"\n🔍 Verifying plan completeness with LLM...")
                                    print(f"📝 Original request: {original_request[:100]}...")
                                    print(f"✅ Completed steps: {', '.join(completed_steps)}")
                                    print(f"❓ Verification prompt: {verification_prompt[:200]}...")
                                logger.info(f"🔍 Plan verification: original='{original_request[:100]}', completed={len(completed_steps)} steps")
                            else:
                                # Verification was sent but LLM hasn't responded yet - shouldn't happen
                                # This means we're in a retry loop, continue
                                should_continue = True
                                logger.warning(f"⚠️ Verification sent but not completed - continuing")
                            
                        elif all_done and not failed_tools:
                            # All steps are done (some may have failed), but no NEW retry prompts were injected
                            # This means either: no failures, or all recovery attempts exhausted
                            should_exit = True
                            completed_count = len([s for s in steps if s.get('status') == 'completed'])
                            failed_count = len([s for s in steps if s.get('status') == 'failed'])
                            if verbose:
                                print(f"\n⚠️ Plan complete with issues: {completed_count} completed, {failed_count} failed")
                            logger.info(f"⚠️ Plan done: {completed_count} succeeded, {failed_count} failed (no recovery available)")
                            
                        elif has_failed and failed_tools:
                            # Steps failed AND retry prompts were just injected
                            # CRITICAL: Continue loop to let agent respond to retry prompts!
                            should_continue = True
                            if verbose:
                                print(f"🔄 Continuing loop for recovery response (retry prompts injected)...")
                            logger.info(f"🔄 Plan has failures with recovery prompts - continuing for agent response")
                            
                        elif not all_done:
                            # Some steps still pending - continue in reactive mode
                            if self.reactive_planning:
                                should_continue = True
                                if verbose:
                                    print(f"🔄 Reactive planning: Getting new response from model...")
                                logger.debug(f"🔄 Reactive planning: Continuing for next step")
                            else:
                                # Batch mode with pending steps but no failures - shouldn't happen
                                # but if it does, exit to avoid infinite loop
                                logger.warning(f"⚠️ Batch mode: Plan has pending steps but loop ending (unexpected state)")
                                should_exit = True
                        
                        # Execute the decision
                        if should_exit:
                            if self.reactive_planning:
                                # Reactive mode: break immediately
                                break
                            else:
                                # Batch/normal mode: also break (let cleanup happen)
                                break
                        elif should_continue:
                            # Continue loop for next iteration
                            continue
                        # If neither, fall through (shouldn't happen but safe)
                        
                    else:
                        # No plan exists - single-tool request
                        if self.reactive_planning and iteration > 0:
                            # Track how many single-tool iterations we've done
                            single_tool_iterations = self.execution_metrics.get("single_tool_iterations", 0) + 1
                            self.execution_metrics["single_tool_iterations"] = single_tool_iterations
                            
                            # After 3 single-tool iterations, force exit to get summary
                            # This prevents infinite loops where agent keeps outputting single tools
                            if single_tool_iterations >= 3:
                                print(f"\n⚠️ Completed {single_tool_iterations} tool executions - requesting final summary")
                                logger.info(f"Reactive planning: {single_tool_iterations} single-tool iterations, exiting")
                                break
                            else:
                                if verbose:
                                    print(f"⚠️ DEBUG: No plan after single-tool execution, continuing for agent response (iteration {single_tool_iterations}/3)")
                                logger.debug(f"Reactive planning: Single-tool complete, continuing for response ({single_tool_iterations}/3)")
                                continue
                        # else: Normal mode without plan - just continue (normal operation)

                # Increment tool call count for iteration limit
                tool_call_count += len(tool_calls)

            # Return the response (main loop will print it if not already displayed)
            # Note: _response_already_displayed flag is set by streaming methods
            
            # ═══════════════════════════════════════════════════════════════════
            # PLAN CLEANUP: If we exit the loop with an active plan, mark it appropriately
            # This handles max_iterations or System 2 halts that leave plans dangling
            # Also clears plan data that was preserved for loop exit check
            # ═══════════════════════════════════════════════════════════════════
            if "current_plan_id" in self.execution_metrics and "current_plan_steps" in self.execution_metrics:
                plan_id = self.execution_metrics["current_plan_id"]
                steps = self.execution_metrics["current_plan_steps"]
                completed_count = len([s for s in steps if s.get('status') == 'completed'])
                total_steps = len(steps)
                
                # Check if plan is fully complete
                all_done = all(s.get('status') in ['completed', 'failed'] for s in steps)
                
                if all_done:
                    # Plan completed successfully
                    # NOTE: May have already been marked completed at line ~4771
                    # This is a safety net for reactive mode where we preserved plan data
                    # for the loop exit check
                    if hasattr(self, 'memory_manager') and self.memory_manager:
                        updates = {
                            "status": "completed",
                            "completed_steps": completed_count,
                            "total_steps": total_steps,
                            "steps": steps,
                            "updated_at": datetime.now(UTC).isoformat()
                        }
                        if self.memory_manager.update_plan(plan_id, updates):
                            logger.info(f"📋 Plan {plan_id} marked as completed at end of execution loop")
                            if verbose:
                                print(f"\n✅ Plan {plan_id[:8]}... completed: {completed_count}/{total_steps} steps succeeded")
                    else:
                        # No memory manager - just log locally
                        logger.info(f"📋 Plan {plan_id} completed (local tracking only): {completed_count}/{total_steps} steps")
                        if verbose:
                            print(f"\n✅ Plan completed: {completed_count}/{total_steps} steps succeeded")
                else:
                    # Mark as abandoned since we're exiting with incomplete steps
                    reason = "Max iterations reached" if not self.stop_requested and not self.system2_halt_requested else \
                             "User requested stop" if self.stop_requested else \
                             self.system2_halt_reason
                    
                    self._mark_plan_abandoned(plan_id, completed_count, total_steps, reason)
                
                # ALWAYS clear cached plan data here - this is the final cleanup
                # In reactive mode, data was preserved for loop exit check
                # In batch mode, data may have been cleared already
                self.execution_metrics.pop("current_plan_id", None)
                self.execution_metrics.pop("current_plan_steps", None)
            
            # System 2 periodic maintenance happens in main loop after response display
            # (removed from here to avoid duplication)
            
            return last_response

        except Exception as e:
            error_msg = f"Error in SAM execution: {str(e)}"
            logger.error(error_msg)
            if verbose:
                traceback.print_exc()
            return error_msg

    def _calculate_recent_error_rate(self) -> float:
        """Calculate error rate for recent tool executions"""
        total_tools = self.execution_metrics["total_tool_count"]
        error_count = self.execution_metrics["tool_error_count"]

        if total_tools == 0:
            return 0.0

        return error_count / total_tools

    def _extract_and_store_plan(self, planning_text: str, tool_calls: List[Dict], iteration: int) -> Optional[str]:
        """
        Extract SAM's planning thoughts and store as a structured plan in Elasticsearch.
        
        Converts natural language planning into JSON format:
        {
            "description": "Overall goal from planning text",
            "steps": [
                {"description": "tool_name with args", "status": "pending", "order": 1},
                ...
            ]
        }
        
        Args:
            planning_text: SAM's natural language planning thoughts
            tool_calls: List of tools SAM plans to execute
            iteration: Current iteration number
            
        Returns:
            plan_id if successfully stored, None otherwise
        """
        if not hasattr(self, 'memory_manager') or not self.memory_manager:
            # No memory manager - store plan locally only (ephemeral agents)
            logger.debug("📋 No memory_manager - storing plan locally only")
            
            # Still create plan steps for reactive planning to track progress
            steps = []
            for idx, tool_call in enumerate(tool_calls, 1):
                tool_name = tool_call.get('name', 'unknown')
                tool_args = tool_call.get('arguments', {})
                
                # Format step description
                args_str = ", ".join(f"{k}={v}" for k, v in list(tool_args.items())[:2])
                if len(tool_args) > 2:
                    args_str += f", +{len(tool_args) - 2} more"
                    
                step_desc = f"{tool_name}({args_str})" if args_str else tool_name
                
                steps.append({
                    "description": step_desc,
                    "status": "pending",
                    "order": idx,
                    "tool_name": tool_name,
                    "tool_args": tool_args
                })
            
            # Store in execution_metrics for reactive planning
            self.execution_metrics["current_plan_steps"] = steps
            # Generate a local plan ID
            import uuid
            plan_id = f"local-{uuid.uuid4().hex[:8]}"
            return plan_id
            
        try:
            # Get the user's most recent request as the goal
            user_request = ""
            for msg in reversed(self.conversation_history):
                if msg.get('role') == 'user' and not msg.get('content', '').startswith('Tool execution'):
                    user_request = msg.get('content', '')[:200]
                    break
            
            # IMPORTANT: Filter out garbage thinking fragments
            # Don't store plans that are ONLY thinking with no substantial content
            # But allow plans that have thinking + actual tool calls (iteration 0)
            if planning_text:
                # Strip thinking tags to check actual content
                content_without_thinking = self._strip_thinking_tags(planning_text).strip()
                # Reject if there's no content after removing thinking tags, OR if it's just a short fragment
                if len(content_without_thinking) < 20 and len(tool_calls) == 0:
                    logger.warning(f"📋 Skipping plan storage - appears to be thinking fragment with no tools: {planning_text[:50]}...")
                    return None
            
            # Create plan description from planning text + user request
            # Prefer user request for clarity, use planning text as secondary
            plan_description = user_request if user_request else planning_text[:300]
            
            # Convert tool calls into plan steps (JSON format)
            steps = []
            for idx, tool_call in enumerate(tool_calls, 1):
                tool_name = tool_call.get('name', 'unknown')
                tool_args = tool_call.get('arguments', {})
                
                # Format step description
                args_str = ", ".join(f"{k}={v}" for k, v in list(tool_args.items())[:2])
                if len(tool_args) > 2:
                    args_str += f", +{len(tool_args) - 2} more"
                    
                step_desc = f"{tool_name}({args_str})" if args_str else tool_name
                
                steps.append({
                    "description": step_desc,
                    "status": "pending",
                    "order": idx,
                    "tool_name": tool_name,
                    "tool_args": tool_args
                })
            
            # Store plan in Elasticsearch
            plan_id = self.memory_manager.store_plan(
                description=plan_description,
                steps=steps,
                metadata={
                    "iteration": iteration,
                    "user_request": user_request,
                    "tool_count": len(tool_calls),
                    "planning_text_full": planning_text
                }
            )
            
            # Cache steps locally to avoid Elasticsearch refresh delays
            if plan_id:
                self.execution_metrics["current_plan_steps"] = steps
            
            return plan_id
            
        except Exception as e:
            logger.error(f"Error extracting and storing plan: {e}")
            return None
    
    def _get_plan_status_summary(self) -> Optional[str]:
        """
        Generate a readable plan status summary for injecting into conversation.
        Shows which steps are completed, in progress, or pending.
        
        Returns:
            Formatted plan status string or None if no active plan
        """
        if "current_plan_id" not in self.execution_metrics:
            return None
            
        steps = self.execution_metrics.get("current_plan_steps", [])
        if not steps:
            return None
            
        status_lines = ["\n📋 Plan Progress:"]
        for idx, step in enumerate(steps, 1):
            status = step.get('status', 'pending')
            desc = step.get('description', 'Unknown step')
            
            if status == 'completed':
                icon = "✅"
            elif status == 'failed':
                icon = "❌"
            elif status == 'in_progress' or status == 'in-progress':
                icon = "🔄"  # Currently executing
            elif idx == len([s for s in steps if s.get('status') in ['completed']]) + 1:
                # This is the next pending step after all completed ones
                icon = "🔄"
            else:
                icon = "⏳"  # Pending
                
            status_lines.append(f"   {icon} Step {idx}: {desc}")
            
        return "\n".join(status_lines)
    
    def _update_plan_progress(self, plan_id: str, current_step: int, total_steps: int, 
                            tool_name: str, result: str) -> None:
        """
        Update plan progress locally (cached). Only pushes to Elasticsearch when complete.
        
        Args:
            plan_id: The plan ID to update
            current_step: Current step number (1-indexed)
            total_steps: Total number of steps
            tool_name: Name of the tool that just executed
            result: Result from tool execution
        """
        if not hasattr(self, 'memory_manager') or not self.memory_manager:
            return
            
        try:
            # Use cached steps from execution_metrics
            steps = self.execution_metrics.get("current_plan_steps", [])
            if not steps:
                logger.debug(f"No cached steps found for plan {plan_id} - plan may be single-step")
                return
            
            # Determine if this step succeeded
            # Use the SAME logic as the main execution loop (line ~5000) to stay consistent
            success = not (
                result.startswith("❌") or 
                result.startswith("Error") or 
                "failed:" in result.lower() or 
                "Errors:" in result  # Catch "⚠️  Errors:" from PowerShell/tool failures
            )
            
            # Update the specific step in the cached steps array
            if current_step - 1 < len(steps):
                old_status = steps[current_step - 1].get('status', 'unknown')
                
                # CRITICAL FIX: Don't overwrite status if it was already assessed by step completion check
                # The step completion assessment (lines 5607-5665) is more thorough (uses LLM + pre-checks)
                # and should take precedence over this simple pattern matching
                if old_status not in ['completed', 'failed']:
                    # Status not yet assessed - use basic pattern matching as fallback
                    new_status = "completed" if success else "failed"
                    logger.debug(f"📊 _update_plan_progress: Step {current_step} status: {old_status} -> {new_status} (success={success}, has_Errors={'Errors:' in result})")
                    steps[current_step - 1]['status'] = new_status
                else:
                    # Status already assessed - keep it
                    logger.debug(f"📊 _update_plan_progress: Step {current_step} status already assessed as '{old_status}', keeping it")
                
                steps[current_step - 1]['result_preview'] = result[:200] if result else None
            
            # Calculate overall plan status based on ALL steps, not just current batch
            all_completed = all(s.get('status') in ['completed', 'failed'] for s in steps)
            is_complete = all_completed
            
            # Determine status: pending -> in_progress -> completed
            if is_complete:
                plan_status = "completed"
            elif current_step > 0:
                plan_status = "in_progress"
            else:
                plan_status = "pending"
            
            # ALWAYS update Elasticsearch, not just on completion
            # This ensures System 2 sees accurate progress and status
            updates = {
                "status": plan_status,
                "completed_steps": len([s for s in steps if s.get('status') == 'completed']),
                "steps": steps,
                "updated_at": datetime.now(UTC).isoformat()
            }
            
            # Use direct document update via sleep + retry
            import time
            update_success = False
            for attempt in range(3):
                if attempt > 0:
                    time.sleep(0.5)  # Wait 500ms between retries
                
                update_success = self.memory_manager.update_plan(plan_id, updates)
                if update_success:
                    break
            
            if update_success:
                if is_complete:
                    # BUGFIX: Don't clear plan data here!
                    # In reactive mode, the plan data is needed for the loop exit check (line ~5365)
                    # It will be cleared in the cleanup section after the loop exits (line ~5448)
                    # self.execution_metrics.pop("current_plan_id", None)  # ← REMOVED
                    # self.execution_metrics.pop("current_plan_steps", None)  # ← REMOVED
                    logger.info(f"📋 Plan {plan_id} marked as completed in Elasticsearch")
                else:
                    logger.debug(f"📋 Plan {plan_id} updated: {plan_status} - {len([s for s in steps if s.get('status') == 'completed'])}/{len(steps)} steps completed")
            else:
                logger.warning(f"Failed to update plan {plan_id} in Elasticsearch after 3 attempts")
            
        except Exception as e:
            logger.error(f"Error updating plan progress: {e}")

    def _mark_plan_abandoned(self, plan_id: str, completed_steps: int, total_steps: int, reason: str) -> None:
        """
        Mark a plan as abandoned when execution is stopped before completion.
        
        Args:
            plan_id: The plan ID to mark as abandoned
            completed_steps: Number of steps completed before abandonment
            total_steps: Total number of planned steps
            reason: Reason for abandonment (user stop, system halt, etc.)
        """
        if not hasattr(self, 'memory_manager') or not self.memory_manager:
            return
            
        try:
            steps = self.execution_metrics.get("current_plan_steps", [])
            
            # Update remaining steps as 'abandoned'
            for idx in range(completed_steps, len(steps)):
                if steps[idx].get('status') == 'pending':
                    steps[idx]['status'] = 'abandoned'
            
            updates = {
                "status": "abandoned",
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "steps": steps,
                "abandonment_reason": reason,
                "updated_at": datetime.now(UTC).isoformat()
            }
            
            import time
            for attempt in range(3):
                if attempt > 0:
                    time.sleep(0.5)
                    
                if self.memory_manager.update_plan(plan_id, updates):
                    logger.info(f"📋 Plan {plan_id} marked as abandoned: {reason}")
                    # Clear cached plan data
                    self.execution_metrics.pop("current_plan_id", None)
                    self.execution_metrics.pop("current_plan_steps", None)
                    break
                    
        except Exception as e:
            logger.error(f"Error marking plan as abandoned: {e}")
    
    def _cleanup_plans(self, mark_as: str = 'abandoned', delete: bool = False) -> str:
        """
        Clean up outstanding plans in ElasticSearch
        
        Args:
            mark_as: Status to mark plans as ('abandoned', 'cancelled', etc.)
            delete: If True, delete plans instead of marking them
            
        Returns:
            Status message
        """
        if not hasattr(self, 'memory_manager') or not self.memory_manager:
            return "❌ ElasticSearch memory not available"
        
        try:
            # Get all active (non-completed) plans
            active_plans = self.memory_manager.get_active_plans(max_results=100)
            
            if not active_plans:
                return "✅ No outstanding plans to clean up"
            
            if delete:
                # Delete plans from Elasticsearch
                deleted_count = 0
                for plan in active_plans:
                    if self.memory_manager.delete_plan(plan['plan_id']):
                        deleted_count += 1
                
                return f"🗑️  Deleted {deleted_count}/{len(active_plans)} outstanding plans"
            else:
                # Mark plans with the specified status
                updated_count = 0
                for plan in active_plans:
                    updates = {
                        "status": mark_as,
                        "updated_at": datetime.now(UTC).isoformat(),
                        "cleanup_reason": "Manual cleanup via CLI"
                    }
                    if self.memory_manager.update_plan(plan['plan_id'], updates):
                        updated_count += 1
                
                return f"🧹 Marked {updated_count}/{len(active_plans)} plans as '{mark_as}'"
                
        except Exception as e:
            logger.error(f"Error cleaning up plans: {e}")
            return f"❌ Error cleaning up plans: {str(e)}"
    
    def _build_tools_context(self) -> str:
        """Build optimized tools context - categorized and concise
        
        SECURITY: Only includes local_tools and mcp_tools.
        System 2 tools are NEVER exposed to System 1's context.
        """

        # Function calling doesn't work reliably with this model
        # Use smart text-based approach instead

        if not self.local_tools and not self.mcp_tools:
            return "\n\n<available_tools>\nNo tools available.\n</available_tools>"

        # Group tools by category for better organization
        tools_by_cat = {}

        for tool_name, tool_data in self.local_tools.items():
            tool_info = self.tool_info.get(tool_name)
            if tool_info:
                cat = tool_info.category.value
                if cat not in tools_by_cat:
                    tools_by_cat[cat] = []

                # Show ALL parameters with optional/required indicators and defaults
                param_strs = []
                for param_name, param_info in tool_info.parameters.items():
                    # Check if optional by looking for default value
                    has_default = 'default' in param_info
                    if has_default:
                        default_val = param_info['default']
                        # Show default value for key parameters
                        if default_val is not None and str(default_val) not in ['None', 'False', 'True', '']:
                            param_strs.append(f"[{param_name}={default_val}]")
                        else:
                            param_strs.append(f"[{param_name}]")
                    else:
                        param_strs.append(param_name)
                
                param_str = ", ".join(param_strs)

                # First sentence only
                desc = tool_info.description.split('.')[0].split('\n')[0][:80]

                tools_by_cat[cat].append(f"{tool_name}({param_str}): {desc}")

        # Build compact categorized list
        tools_text = []

        # Prioritize most-used categories
        priority_cats = ['utility', 'development', 'filesystem', 'system']

        for cat in priority_cats:
            if cat in tools_by_cat:
                tools_text.append(f"**{cat.upper()}**: {', '.join(tools_by_cat[cat])}")

        # Add remaining categories
        for cat, tools in tools_by_cat.items():
            if cat not in priority_cats:
                tools_text.append(f"**{cat.upper()}**: {', '.join(tools)}")

        # Add MCP tools with full descriptions
        if self.mcp_tools:
            mcp_tool_list = []
            for tool_name, (server_name, tool_info) in self.mcp_tools.items():
                desc = tool_info.get('description', '')
                # Get first sentence only
                desc = desc.split('.')[0].split('\n')[0][:80]
                
                # Extract parameters from input_schema
                schema = tool_info.get('input_schema', {})
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                param_strs = []
                for param_name, param_def in properties.items():
                    if param_name in required:
                        param_strs.append(param_name)
                    else:
                        param_strs.append(f"[{param_name}]")
                
                param_str = ", ".join(param_strs)
                mcp_tool_list.append(f"{tool_name}({param_str}): {desc}")
            
            tools_text.append(f"**BROWSER (MCP)**: {', '.join(mcp_tool_list)}")

        tools_context = f"""
    <available_tools>
    {chr(10).join(tools_text)}

    Total: {len(self.local_tools) + len(self.mcp_tools)} tools

    Usage: Respond with JSON in code block:
    ```json
    {{"name": "tool_name", "arguments": {{"arg": "value"}}}}
    ```
    </available_tools>"""

        return tools_context

    def _estimate_token_count(self, text: str) -> int:
        """More accurate token count estimation for Claude/LLM"""
        if not text:
            return 0

        # Claude typically uses ~3.5-4 characters per token for English text
        # Use conservative estimate to avoid hitting limits
        char_count = len(text)

        # Account for JSON structure, code blocks, special formatting
        if '```' in text or '{' in text or '"name":' in text:
            # Code/JSON tends to be more token-dense
            return int(char_count / 2.8)  # More conservative for small contexts
        else:
            # Regular text
            return int(char_count / 3.2)  # More conservative estimate

    def _get_context_status(self) -> str:
        """Get current context usage status"""
        total_tokens = 0
        message_breakdown = {"system": 0, "user": 0, "assistant": 0}

        for msg in self.conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tokens = self._estimate_token_count(content)
            total_tokens += tokens

            if role in message_breakdown:
                message_breakdown[role] += tokens

        percent_used = (total_tokens / self.context_limit) * 100

        warning = ""
        if percent_used > 90:
            warning = "🚨 CRITICAL: Context nearly full!"
        elif percent_used > 75:
            warning = "⚠️ WARNING: Context usage high"
        elif percent_used > 50:
            warning = "📊 INFO: Context at 50%"

        return (
            f"CONTEXT STATUS: ~{total_tokens:,} tokens used (~{percent_used:.1f}% of {self.context_limit:,}). "
            f"Messages: {len(self.conversation_history)} "
            f"Tools: {len(self.local_tools)} local, {len(self.mcp_tools)} MCP. "
            f"{warning}"
        )

    def _check_context_and_warn_user(self) -> None:
        current_tokens = sum(
            self._estimate_token_count(msg.get('content', ''))
            for msg in self.conversation_history
        )
        usage_percent = current_tokens / self.short_term_context_tokens  # Compare to short-term limit

        if usage_percent > 0.85:
            print(f"\n🚨 SHORT-TERM MEMORY: {usage_percent:.1%} full "
                  f"({current_tokens:,}/{self.short_term_context_tokens:,} tokens)")
            print(f"   📊 System 1 total capacity: {self.context_limit:,} tokens")
            print(f"   🧠 System 2 capacity: {self.system2_context_limit:,} tokens")
        elif usage_percent > 0.70:  # System2 intervention threshold
            print(f"\n⚠️ SHORT-TERM MEMORY: {usage_percent:.1%} full "
                  f"({current_tokens:,}/{self.short_term_context_tokens:,} tokens)")
            print(f"   📊 System 1 total capacity: {self.context_limit:,} tokens")
            print(f"   🧠 System 2 capacity: {self.system2_context_limit:,} tokens")
            print(f"   🧠 System 2 monitoring - may prune redundant messages")
        elif usage_percent > 0.55:  # Early warning
            print(f"\n📊 SHORT-TERM MEMORY: {usage_percent:.1%} full "
                  f"({current_tokens:,}/{self.short_term_context_tokens:,} tokens)")
            print(f"   📊 System 1 total capacity: {self.context_limit:,} tokens")
            print(f"   🧠 System 2 capacity: {self.system2_context_limit:,} tokens")


    def _fill_context_for_testing(self, target_percent: int = 58) -> str:
        """Fill context with sample messages for testing thresholds
        
        Args:
            target_percent: Target context usage percentage (10-95)
            
        Returns:
            Status message with results
        """
        try:
            # Calculate target tokens
            target_tokens = int(self.context_limit * (target_percent / 100))
            current_tokens = sum(self._estimate_token_count(msg.get('content', ''))
                                 for msg in self.conversation_history)
            
            if current_tokens >= target_tokens:
                return f"⚠️ Context already at {(current_tokens/self.context_limit)*100:.1f}% ({current_tokens:,} tokens)"
            
            tokens_needed = target_tokens - current_tokens
            
            # Sample messages of various types for realistic testing
            sample_messages = [
                "What files are in the current directory?",
                "Can you explain how the context management system works?",
                "Please read the configuration file and tell me what settings are enabled.",
                "I need to understand the System 2 intervention thresholds.",
                "Can you search the codebase for token estimation functions?",
                "What's the difference between System 1 and System 2 tools?",
                "Please analyze the memory management implementation.",
                "How does the ElasticSearch integration work?",
                "Can you show me examples of the tool execution flow?",
                "What happens when context reaches 60% usage?",
                "Explain the proactive rotation feature.",
                "How are MCP servers integrated into the system?",
                "What's the purpose of the System 3 moral authority?",
                "Can you describe the autonomous mode functionality?",
                "Tell me about the plugin architecture.",
            ]
            
            # Add messages until we reach target
            messages_added = 0
            estimated_tokens = current_tokens
            
            while estimated_tokens < target_tokens:
                # Cycle through sample messages
                user_msg = sample_messages[messages_added % len(sample_messages)]
                assistant_msg = f"Response {messages_added + 1}: " + " ".join([
                    "This is a sample response for context filling testing.",
                    "It contains enough content to simulate realistic token usage.",
                    "The actual content doesn't matter for threshold testing.",
                    f"Message index: {messages_added}",
                ])
                
                # Add user message
                self.conversation_history.append({
                    "role": "user",
                    "content": user_msg
                })
                estimated_tokens += self._estimate_token_count(user_msg)
                
                # Add assistant message
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_msg
                })
                estimated_tokens += self._estimate_token_count(assistant_msg)
                
                messages_added += 2
                
                # Safety check
                if messages_added > 1000:
                    break
            
            # Final count
            final_tokens = sum(self._estimate_token_count(msg.get('content', ''))
                               for msg in self.conversation_history)
            final_percent = (final_tokens / self.context_limit) * 100
            
            return (f"✅ Context filled: {messages_added} messages added\n"
                    f"📊 Current: {final_tokens:,}/{self.context_limit:,} tokens ({final_percent:.1f}%)\n"
                    f"🎯 Target: {target_percent}% ({target_tokens:,} tokens)\n"
                    f"💡 Ready to test threshold behaviors!")
            
        except Exception as e:
            return f"❌ Failed to fill context: {e}"

    def list_tools(self) -> Dict[str, Any]:
        """List all available tools (System 1 only - excludes System 2 tools)
        
        SECURITY: This only lists tools available to System 1.
        System 2 tools are kept in a separate registry and never exposed here.
        """
        local_tools = {}
        mcp_tools = {}

        # Local tools (System 1 only - tool_info does not include System 2 tools)
        for tool_name, tool_info in self.tool_info.items():
            local_tools[tool_name] = {
                "description": tool_info.description,
                "category": tool_info.category.value,
                "requires_approval": tool_info.requires_approval,
                "usage_count": tool_info.usage_count,
                "parameters": tool_info.parameters
            }

        # MCP tools
        for tool_name, (server_name, tool_info) in self.mcp_tools.items():
            mcp_tools[tool_name] = {
                "description": tool_info.get('description', ''),
                "server": server_name,
                "input_schema": tool_info.get('input_schema', {}),
                "category": "mcp"
            }

        return {
            "local_tools": local_tools,
            "mcp_tools": mcp_tools,
            "mcp_servers": self.list_mcp_servers(),
            "total_count": len(local_tools) + len(mcp_tools)
        }

    def list_system2_tools(self) -> str:
        """List all System 2 exclusive tools"""
        if not self.system2_tools:
            return "🧠 No System 2 tools available"

        result = "🧠 SYSTEM 2 EXCLUSIVE TOOLS (Metacognitive)\n"
        result += "=" * 60 + "\n"
        result += "🔒 These tools are ONLY accessible to System 2\n"
        result += "❌ System 1 cannot see or call these tools\n"
        result += "=" * 60
        
        # Group tools by category for display
        tools_by_cat = {}
        for tool_name, tool_data in self.system2_tools.items():
            tool_info = self.system2_tool_info.get(tool_name)
            if tool_info:
                cat = tool_info.category.value
                if cat not in tools_by_cat:
                    tools_by_cat[cat] = []
                
                desc = str(tool_info.description)
                desc = desc.split('.')[0].split('\n')[0]
                if len(desc) > 80:
                    desc = desc[:80] + "..."
                tools_by_cat[cat].append((tool_name, desc))
        
        # Display by category
        for cat in sorted(tools_by_cat.keys()):
            result += f"\n\n📁 {cat.upper()}:"
            for tool_name, desc in sorted(tools_by_cat[cat]):
                result += f"\n  🧠 {tool_name}: {desc}"

        result += f"\n\n{'=' * 60}"
        result += f"\n📊 Total: {len(self.system2_tools)} metacognitive tools"
        result += "\n💡 System 2 uses these to manage System 1's context & behavior"

        return result


# ===== CLI INTERFACE =====
def main():
    """CLI interface for SAM Agent"""
    # Add this line right at the start:
    autonomous_manager = None

    import argparse

    # Add argument parsing
    parser = argparse.ArgumentParser(description="SAM Agent - Secret Agent Man")
    parser.add_argument("--api", action="store_true", help="Run as HTTP API server")
    parser.add_argument("--api-host", type=str, default="0.0.0.0", help="API server host")
    parser.add_argument("--api-port", type=int, default=8888, help="API server port")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # If --api flag is provided, run API server instead of interactive mode
    if args.api:
        if FASTAPI_AVAILABLE:
            print("🌐 Starting SAM API Server...")

            # Initialize SAM with default values, it will load config internally
            sam = SAMAgent(safety_mode=True, auto_approve=True)

            # Auto-load all plugins from the plugins directory
            load_all_plugins(sam)

            # Enable System3 if configured (optional, not mandatory)
            system3_config = sam.raw_config.get('system3', {})
            if SYSTEM3_AVAILABLE and system3_config.get('enabled', False):
                use_claude = system3_config.get('use_claude', False)
                result = sam.enable_conscience(use_claude=use_claude)
                print(result)
            
            # Check approval mode
            agent_config = sam.raw_config.get('agent', {})
            auto_approve = agent_config.get('auto_approve', True)
            if not auto_approve:
                print("⚠️ Auto-approve disabled - tool calls will require approval")
                if system3_config.get('enabled', False):
                    print("   📋 System3 will evaluate tool calls")
                else:
                    print("   📋 WebSocket/UI approval will be required")
            
            # Start API server
            api_config = sam.raw_config.get('api', {})
            server = SAMAPIServer(sam, host=args.api_host, port=args.api_port)
            print(f"🚀 Starting server on http://{server.host}:{server.port}")
            print("📚 API docs available at http://localhost:8888/docs")
            server.run()
            return
        else:
            print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
            return

    # If no --api flag, continue with original interactive mode
    print("🕵️ Starting SAM initialization...")

    # Initialize SAM - will auto-load config.json
    sam = SAMAgent()

    # Auto-load all plugins from the plugins directory
    load_all_plugins(sam)

    # Check memory system status
    if hasattr(sam, 'memory_manager'):
        print("✅ ElasticSearch memory system initialized")
    else:
        print("📝 Using notes.txt for memory (ElasticSearch disabled)")

    # Test API connection and show model info (disable streaming for initialization)
    try:
        test_response = sam._single_completion_attempt([
            {"role": "user", "content": "Hello, are you working?"}
        ], stream=False)
        if "error" not in test_response.lower():
            print("✅ API connection test successful!")
            print(f"🤖 Using model: {sam.model_name}")
            print(f"🧠 Context limit: {sam.context_limit:,} tokens")
        else:
            print(f"❌ API test failed: {test_response}")
    except Exception as e:
        print(f"❌ API connection failed: {e}")

    # Display capabilities
    tools_info = sam.list_tools()
    print(f"\n=== 🕵️ SAM CAPABILITIES ===")
    print(f"🤖 Model: {sam.model_name}")
    print(f"🧠 Context: {sam.context_limit:,} tokens")
    print(f"🔧 Local tools: {len(sam.local_tools)}")
    print(f"🌐 MCP tools: {len(sam.mcp_tools)}")
    print(f"📡 MCP servers: {len(sam.mcp_sessions)}")
    print(f"🔌 Plugins: {len(sam.plugin_manager.plugins)}")
    print(f"🛡️ Safety mode: {'ON' if sam.safety_mode else 'OFF'}")
    print(f"⚠️ Auto-approve: {'ON' if sam.auto_approve else 'OFF'}")

    # Interactive loop
    print(f"\n=== 🖥️ SAM Agent Interactive Mode ===")
    print("Type 'exit' to quit, 'tools' to list available tools, 'tools2' for System 2 tools")
    print("Commands: 'debug', 'reset', 'context' (or 'status'), 'test_context [percent]'")
    print("Memory: 'memory status/migrate/test' (ElasticSearch memory)")
    print("Plans: 'plans list/cleanup/clear' (manage outstanding plans)")
    print("Commands: 'debug' (toggle debug), 'reset' (clear history), 'tools' (list tools)")
    print("Providers: 'provider claude/lmstudio', 'providers' (list available)")
    print("Safety: 'safety on/off', 'auto on/off', 'safety' (status)")
    print("Conscience: 'conscience on/test/live/stats', 'conscience' (status)")
    print("Autonomous: 'autonomous on/off', 'autonomous' (status), 'heartbeat' (manual prompt)")
    print("MCP Commands: 'mcp servers', 'mcp connect <server>', 'mcp disconnect <server>'")
    print("Testing: 'test_context [percent]' - Fill context to target % (default 58)")

    while True:
        try:
            # Add extra spacing for better readability
            print()  # Empty line before prompt
            user_input = input(f"💬 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                # Clean up MCP connections
                asyncio.run(sam.disconnect_mcp_servers())
                break

            # Handle safety commands
            elif (user_input.lower().startswith('safety') or
                  user_input.lower() in ['auto on', 'auto off'] or
                  user_input.lower() == 'auto'):

                safety_commands = {
                    'safety': sam.get_safety_status,
                    'safety on': lambda: sam.set_safety_mode(True),
                    'safety off': lambda: sam.set_safety_mode(False),
                    'auto on': lambda: sam.set_auto_approve(True),
                    'auto off': lambda: sam.set_auto_approve(False),
                    'auto': sam.get_safety_status,  # Add auto as alias for status
                }

                if user_input.lower() in safety_commands:
                    result = safety_commands[user_input.lower()]()
                    print(result)
                    continue

            # Handle autonomous commands
            elif user_input.lower().strip() == 'autonomous on':
                if autonomous_manager and autonomous_manager.is_running():
                    print("❤️ Autonomous mode already running")
                    continue

                result = sam.enable_autonomous_mode()
                print(result)

                # Import and start the autonomous manager
                try:
                    from autonomous_manager import SAMAutonomousManager
                    autonomous_manager = SAMAutonomousManager(sam)
                    autonomous_manager.start_autonomous_mode()

                    # Switch to collaborative mode
                    print("\n" + "=" * 60)
                    print("🤖 COLLABORATIVE MODE ACTIVE")
                    print("SAM is now exploring autonomously while accepting your input")
                    print("Type normally to interact, 'status' for info, 'autonomous off' to stop")
                    print("=" * 60 + "\n")

                    # Collaborative input loop
                    while autonomous_manager.is_running():
                        try:
                            collaborative_input = input("💬 You: ").strip()
                            if collaborative_input:
                                # This will be processed by the autonomous manager
                                autonomous_manager.inject_human_input(collaborative_input)

                                # Check if we should exit collaborative mode
                                if collaborative_input.lower().strip() == 'autonomous off':
                                    break

                        except KeyboardInterrupt:
                            print("\n🛑 Stopping autonomous mode...")
                            autonomous_manager.stop()
                            break

                    autonomous_manager = None
                    print("\n" + "=" * 60)
                    print("🤖 RETURNING TO STANDARD MODE")
                    print("=" * 60 + "\n")
                    continue

                except ImportError:
                    print("❌ Could not import autonomous manager")
                    continue

            elif user_input.lower().strip() == 'heartbeat':
                if sam.inject_heartbeat_prompt():
                    print("❤️ Heartbeat prompt injected")
                else:
                    time_since = time.time() - sam.execution_metrics["last_autonomous_prompt"]
                    wait_time = max(0, 180 - time_since)
                    print(f"❤️ Too soon for next heartbeat pulse (wait {wait_time / 60:.1f} more minutes)")
                continue

            # Handle conscience commands
            elif user_input.lower().startswith('conscience'):
                conscience_command = user_input.lower().strip()

                if conscience_command == 'conscience on':
                    result = sam.enable_conscience(use_claude=False, test_mode=False)  # Default to local LLM
                    print(result)
                    continue

                elif conscience_command == 'conscience test':
                    result = sam.test_conscience()
                    print(result)
                    continue

                elif conscience_command == 'conscience live':
                    result = sam.test_conscience_live()
                    print(result)
                    continue

                elif conscience_command == 'conscience stats':
                    result = sam.get_conscience_stats()
                    print(result)
                    continue

                elif conscience_command == 'conscience':
                    if hasattr(sam, 'system3'):
                        print("🛡️ System 3 (conscience) is ACTIVE")
                        print("📊 Use 'conscience stats' for statistics")
                        print("🧪 Use 'conscience test' for test scenarios")
                        print("🔬 Use 'conscience live' for live testing")
                    else:
                        print("❌ System 3 (conscience) is DISABLED")
                        print("💡 Use 'conscience on' to enable")
                    continue

            # Handle provider commands
            elif user_input.lower().startswith('provider '):
                provider_name = user_input.split(' ', 1)[1].strip()
                result = sam.switch_provider(provider_name)
                print(result)
                continue
            elif user_input.lower() == 'providers':
                result = sam.get_current_provider()
                print(result)
                continue

            # Handle utility commands
            elif user_input.lower() == 'debug':
                sam.debug_mode = not sam.debug_mode
                print(f"🐛 Debug mode: {'ON' if sam.debug_mode else 'OFF'}")
                continue

            elif user_input.lower() == 'reset':
                sam.conversation_history = []
                print("🔄 Conversation history cleared")
                continue

            elif user_input.lower() in ['context', 'status']:
                # Show current context usage
                current_tokens = sum(sam._estimate_token_count(msg.get('content', ''))
                                    for msg in sam.conversation_history)
                usage_percent = current_tokens / sam.context_limit
                messages_count = len(sam.conversation_history)
                
                print(f"\n📊 CONTEXT STATUS")
                print("=" * 60)
                print(f"🔢 Messages: {messages_count}")
                print(f"📏 Tokens: {current_tokens:,} / {sam.context_limit:,}")
                print(f"📊 Usage: {usage_percent:.1%}")
                
                # System2 intervention threshold from config (default 0.70)
                system2_threshold = sam.system2.token_threshold if hasattr(sam, 'system2') else 0.70
                
                # Show status indicators aligned with System2 intervention
                if usage_percent >= 0.85:
                    print(f"🚨 Status: CRITICAL - Context nearly full")
                elif usage_percent >= system2_threshold:
                    print(f"⚠️  Status: HIGH - System 2 intervention active ({system2_threshold:.0%})") 
                elif usage_percent >= (system2_threshold * 0.85):  # 85% of threshold (e.g., 59.5% if threshold is 70%)
                    print(f"⚡ Status: MODERATE - Approaching System 2 threshold ({system2_threshold:.0%})")
                else:
                    print(f"✅ Status: OK - Plenty of context available")
                
                print("=" * 60)
                continue

            # Handle tools commands
            elif user_input.lower() == 'tools':
                # List tools with current usage counts
                current_tools = sam.list_tools()
                system1_count = len(current_tools.get('local_tools', {})) + len(current_tools.get('mcp_tools', {}))
                
                print(f"\n🤖 SYSTEM 1 TOOLS ({system1_count} available):")
                print("=" * 60)

                # Local tools - organized by category
                if current_tools.get('local_tools'):
                    # Group tools by category
                    tools_by_category = {}
                    for name, info in current_tools.get('local_tools', {}).items():
                        category = info.get('category', 'other')
                        if category not in tools_by_category:
                            tools_by_category[category] = []
                        
                        approval = "🛡️" if info.get('requires_approval', False) else "✅"
                        usage = info.get('usage_count', 0)
                        usage_text = f" ({usage}x)" if usage > 0 else ""
                        
                        # Get first line of description only
                        desc = info.get('description', 'No description')
                        desc_short = desc.split('.')[0].split('\n')[0][:80]
                        
                        tools_by_category[category].append(f"  {approval} {name}: {desc_short}{usage_text}")
                    
                    # Display by category
                    priority_categories = ['utility', 'development', 'filesystem', 'system']
                    
                    for category in priority_categories:
                        if category in tools_by_category:
                            print(f"\n📁 {category.upper()}:")
                            for tool_line in sorted(tools_by_category[category]):
                                print(tool_line)
                    
                    # Display remaining categories
                    for category in sorted(tools_by_category.keys()):
                        if category not in priority_categories:
                            print(f"\n📁 {category.upper()}:")
                            for tool_line in sorted(tools_by_category[category]):
                                print(tool_line)

                # MCP tools
                if current_tools.get('mcp_tools'):
                    print(f"\n🌐 MCP TOOLS:")
                    for name, info in current_tools.get('mcp_tools', {}).items():
                        server = info.get('server', 'unknown')
                        desc = info.get('description', 'No description')
                        desc_short = desc.split('.')[0].split('\n')[0][:80]
                        print(f"  🌐 {name}: {desc_short} (Server: {server})")
                
                # Add note about System 2 tools
                system2_count = len(sam.system2_tools)
                print(f"\n{'=' * 60}")
                print(f"🧠 System 2 has {system2_count} exclusive metacognitive tools")
                print(f"💡 Use 'tools2' to view System 2 tools (not accessible to System 1)")
                continue

            elif user_input.lower() == 'tools2':
                # List System 2 exclusive tools
                system2_info = sam.list_system2_tools()
                print(system2_info)
                continue

            # Test command to fill context
            elif user_input.lower().startswith('test_context'):
                # Extract target percentage (default 58%)
                parts = user_input.split()
                target_percent = 58  # Default: just before 60% rotation
                if len(parts) > 1:
                    try:
                        target_percent = int(parts[1])
                        if target_percent < 10 or target_percent > 95:
                            print("❌ Target must be between 10-95%")
                            continue
                    except ValueError:
                        print("❌ Invalid percentage. Usage: test_context [percent]")
                        continue
                
                print(f"🧪 Filling context to ~{target_percent}%...")
                result = sam._fill_context_for_testing(target_percent)
                print(result)
                continue

            # Plan management commands
            elif user_input.lower().startswith('plans '):
                plans_command = user_input[6:].strip().lower()

                if plans_command == 'cleanup':
                    if hasattr(sam, 'memory_manager'):
                        print("🧹 Cleaning up outstanding plans (marking as abandoned)...")
                        result = sam._cleanup_plans(mark_as='abandoned')
                        print(result)
                    else:
                        print("❌ ElasticSearch memory not available")
                    continue

                elif plans_command == 'clear':
                    print("⚠️  This will permanently delete all non-completed plans.")
                    confirm = input("Type 'yes' to confirm: ")
                    if confirm.lower() == 'yes':
                        if hasattr(sam, 'memory_manager'):
                            print("🗑️  Deleting outstanding plans...")
                            result = sam._cleanup_plans(delete=True)
                            print(result)
                        else:
                            print("❌ ElasticSearch memory not available")
                    else:
                        print("❌ Cancelled")
                    continue

                elif plans_command == 'list':
                    if hasattr(sam, 'memory_manager'):
                        plans = sam.memory_manager.get_active_plans(max_results=50)
                        if not plans:
                            print("✅ No active plans")
                        else:
                            print(f"\n📋 Active Plans ({len(plans)}):")
                            for i, plan in enumerate(plans, 1):
                                status_icon = "🔄" if plan['status'] == 'in_progress' else "⏸️"
                                print(f"{i}. {status_icon} {plan['description'][:80]}")
                                print(f"   Status: {plan['status']} | Progress: {plan['completed_steps']}/{plan['total_steps']}")
                            print()
                    else:
                        print("❌ ElasticSearch memory not available")
                    continue

            # Memory management commands
            elif user_input.lower().startswith('memory '):
                memory_command = user_input[7:].strip().lower()

                if memory_command == 'status':
                    if hasattr(sam, 'memory_manager'):
                        print("✅ ElasticSearch memory system active")
                        print(f"📊 Host: {sam.config.elasticsearch.host}")
                        print(f"📂 Datastream: {sam.config.elasticsearch.datastream_name}")
                    else:
                        print("📝 Using notes.txt for memory storage")
                    continue

                elif memory_command == 'migrate':
                    if hasattr(sam, 'memory_manager'):
                        print("🔄 Migrating notes.txt to ElasticSearch...")
                        success = sam.memory_manager.migrate_from_notes_txt()
                        print(f"{'✅' if success else '❌'} Migration {'complete' if success else 'failed'}")
                    else:
                        print("❌ ElasticSearch memory not available")
                    continue

                elif memory_command == 'test':
                    if hasattr(sam, 'memory_manager'):
                        print("🧪 Testing ElasticSearch connection...")
                        try:
                            # Store a test memory
                            success = sam.memory_manager.store_memory(
                                content="Test memory from SAM initialization",
                                memory_type="test",
                                metadata={"test": True}
                            )
                            print(f"{'✅' if success else '❌'} Store test {'passed' if success else 'failed'}")

                            # Retrieve recent memories
                            memories = sam.memory_manager.get_recent_memories(max_results=1)
                            print(f"{'✅' if memories else '❌'} Retrieve test {'passed' if memories else 'failed'}")
                        except Exception as e:
                            print(f"❌ Test failed: {e}")
                    else:
                        print("❌ ElasticSearch memory not available")
                    continue

            # Handle MCP-specific commands
            elif user_input.lower().startswith('mcp '):
                mcp_command = user_input[4:].strip()

                if mcp_command == 'servers':
                    servers = sam.list_mcp_servers()
                    if servers:
                        print(f"\n🌐 Connected MCP Servers ({len(servers)}):")
                        for name, info in servers.items():
                            print(f"  📡 {name}: {info['tool_count']} tools")
                            for tool in info['tools']:
                                print(f"    - {tool}")
                    else:
                        print("🌐 No MCP servers connected")
                    continue

                elif mcp_command.startswith('connect '):
                    server_name = mcp_command[8:].strip()
                    if hasattr(sam.config, 'mcp') and sam.config.mcp.servers and server_name in sam.config.mcp.servers:
                        server_config = sam.config.mcp.servers[server_name]
                        result = asyncio.run(sam.connect_to_mcp_server(
                            server_name=server_name,
                            server_type=server_config.get('type', 'stdio'),
                            server_path_or_url=server_config.get('path', ''),
                            headers=server_config.get('headers', {})
                        ))
                        if result:
                            print(f"✅ Connected to MCP server: {server_name}")
                        else:
                            print(f"❌ Failed to connect to MCP server: {server_name}")
                    else:
                        print(f"❌ Server '{server_name}' not found in configuration")
                    continue

                elif mcp_command.startswith('disconnect '):
                    server_name = mcp_command[11:].strip()

                    if server_name in sam.mcp_sessions:
                        session_data = sam.mcp_sessions[server_name]

                        # Handle the dictionary format properly
                        if isinstance(session_data, dict):
                            # Terminate the process if it exists
                            if 'process' in session_data:
                                process = session_data['process']
                                if process and process.returncode is None:
                                    process.terminate()
                                    try:
                                        process.wait(timeout=2)  # Wait up to 2 seconds for clean shutdown
                                    except:
                                        process.kill()  # Force kill if it doesn't terminate

                        elif hasattr(session_data, 'process'):
                            # Handle direct session objects
                            if session_data.process and session_data.process.returncode is None:
                                session_data.process.terminate()
                                try:
                                    session_data.process.wait(timeout=2)
                                except:
                                    session_data.process.kill()

                        # Remove from sessions and tools
                        del sam.mcp_sessions[server_name]
                        tools_to_remove = [tool for tool, (srv, _) in sam.mcp_tools.items() if srv == server_name]
                        for tool in tools_to_remove:
                            del sam.mcp_tools[tool]

                        print(f"✅ Disconnected from MCP server: {server_name}")
                    else:
                        print(f"❌ Server '{server_name}' is not connected")
                    continue

            # If we get here, it's a regular query for the LLM
            print("\n🤖 SAM is thinking...")

            # Run SAM with the user input (async)
            response = asyncio.run(sam.run(user_input, max_iterations=sam.max_iterations, verbose=sam.debug_mode))

            # Only print if response wasn't already displayed with flow separator
            if not hasattr(sam, '_response_already_displayed') or not sam._response_already_displayed:
                print(f"\n🤖 SAM: {response}")

            # Check for System 2 periodic wakeup AFTER response is displayed
            # Use actual_user_prompts (not user_prompt_count) to wake every N USER prompts
            wakeup_interval = sam.execution_metrics.get("system2_wakeup_interval", 5)
            actual_user_prompts = sam.execution_metrics["actual_user_prompts"]
            
            if actual_user_prompts > 0 and actual_user_prompts % wakeup_interval == 0:
                # System 2 wakeup triggered after every N user prompts
                sam._periodic_system2_wakeup()
            elif sam.debug_mode:
                # Only show diagnostic info in debug mode
                next_wakeup = ((actual_user_prompts // wakeup_interval) + 1) * wakeup_interval
                print(f"📊 System 2 check: user prompt #{actual_user_prompts}, next wakeup at #{next_wakeup}")


        except KeyboardInterrupt:
            if autonomous_manager and autonomous_manager.is_running():
                autonomous_manager.stop()
            print("\n\n👋 Goodbye!")
            asyncio.run(sam.disconnect_mcp_servers())
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            if sam.debug_mode:
                traceback.print_exc()


# ===== API SERVER (if FastAPI available) =====
if FASTAPI_AVAILABLE:
    class SAMAPIServer:
        """FastAPI server wrapper for SAM Agent"""

        def __init__(self, sam_agent: SAMAgent, host: str = "0.0.0.0", port: int = 8000):
            self.sam_agent = sam_agent
            self.host = host
            self.port = port
            self.app = FastAPI(
                title="SAM Agent API",
                description="Secret Agent Man API with Safety Controls and MCP Support",
                version="1.0.0"
            )
            self.start_time = time.time()
            self._setup_routes()

        def _setup_routes(self):
            """Setup FastAPI routes"""

            @self.app.post("/query", response_model=QueryResponse)
            async def process_query(request: QueryRequest):
                try:
                    # Handle auto-approve override
                    original_auto_approve = self.sam_agent.auto_approve
                    if request.auto_approve is not None:
                        self.sam_agent.auto_approve = request.auto_approve

                    # Process the query
                    response = await self.sam_agent.run(
                        request.message,
                        max_iterations=request.max_iterations,
                        verbose=request.verbose
                    )

                    # Restore original setting
                    self.sam_agent.auto_approve = original_auto_approve

                    return QueryResponse(
                        response=response,
                        session_id=request.session_id or "default",
                        status="success",
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )

                except Exception as e:
                    logger.error(f"API Error: {str(e)}")
                    return QueryResponse(
                        response="",
                        session_id=request.session_id or "default",
                        status="error",
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        error=str(e)
                    )

            @self.app.get("/health", response_model=HealthResponse)
            async def health_check():
                return HealthResponse(
                    status="healthy",
                    model=self.sam_agent.model_name,
                    tools_count=len(self.sam_agent.local_tools) + len(self.sam_agent.mcp_tools),
                    plugins_count=len(self.sam_agent.plugin_manager.plugins),
                    uptime_seconds=time.time() - self.start_time
                )

            @self.app.get("/tools")
            async def list_tools():
                return self.sam_agent.list_tools()

            @self.app.get("/mcp/servers")
            async def list_mcp_servers():
                return self.sam_agent.list_mcp_servers()

            @self.app.post("/mcp/connect")
            async def connect_mcp_server(server_name: str, server_type: str, server_url: str):
                success = await self.sam_agent.connect_to_mcp_server(server_name, server_type, server_url)
                return {"success": success, "server": server_name}

            @self.app.post("/execute-tool", response_model=ToolExecutionResponse)
            async def execute_tool_direct(request: ToolExecutionRequest):
                start_time = time.time()
                try:
                    result = await self.sam_agent._execute_tool(request.tool_name, request.arguments)
                    execution_time = time.time() - start_time

                    return ToolExecutionResponse(
                        success=True,
                        result=result,
                        execution_time=execution_time,
                        tool_name=request.tool_name
                    )

                except Exception as e:
                    execution_time = time.time() - start_time
                    return ToolExecutionResponse(
                        success=False,
                        error=str(e),
                        execution_time=execution_time,
                        tool_name=request.tool_name
                    )

            @self.app.get("/safety")
            async def get_safety_status():
                return self.sam_agent.get_detailed_safety_status()

            @self.app.post("/safety/mode")
            async def set_safety_mode(enabled: bool):
                result = self.sam_agent.set_safety_mode(enabled)
                return {"message": result, "safety_mode": self.sam_agent.safety_mode}

            @self.app.post("/safety/auto-approve")
            async def set_auto_approve(enabled: bool):
                result = self.sam_agent.set_auto_approve(enabled)
                return {"message": result, "auto_approve": self.sam_agent.auto_approve}

        def run(self):
            """Run the API server"""
            import uvicorn
            uvicorn.run(self.app, host=self.host, port=self.port)


    def run_api_server():
        """Start SAM as an API server"""
        print("🌐 Starting SAM API Server...")

        # Initialize SAM
        sam = SAMAgent(safety_mode=True, auto_approve=False)

        # Load core tools
        try:
            from plugins.core_tools import CoreToolsPlugin
            core_plugin = CoreToolsPlugin()
            core_plugin.register_tools(sam)
            print("✅ Core tools plugin loaded!")
        except ImportError as e:
            print(f"⚠️  Could not load core tools: {e}")

        # Start API server
        server = SAMAPIServer(sam)
        print(f"🚀 Starting server on http://{server.host}:{server.port}")
        print("📚 API docs available at http://localhost:8000/docs")
        server.run()

else:
    def run_api_server():
        print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")

if __name__ == "__main__":
    main()