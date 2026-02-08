#!/usr/bin/env python3
"""
System 2 Tools Plugin for SAM Agent
Metacognitive tools for context management, loop detection, and strategic intervention
These tools are exclusively available to the System 2 metacognitive agent
"""

import json
import logging
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from sam_agent import SAMPlugin, ToolCategory

logger = logging.getLogger("SAM.System2Tools")


def format_tool_call_safely(tool_name: str, arguments: dict, result: str = None) -> str:
    """
    Format a tool call for safe display without triggering re-execution.
    Use this when showing historical tool calls in responses or memories.
    """
    formatted = f"HISTORICAL TOOL EXECUTION (archived - not for execution):\n"
    formatted += f"Tool: {tool_name}\n"
    formatted += f"Arguments: {json.dumps(arguments, indent=2)}\n"
    if result:
        result_preview = result[:500] + "..." if len(result) > 500 else result
        formatted += f"Result: {result_preview}\n"
    return formatted


class System2ToolsPlugin(SAMPlugin):
    """Metacognitive tools exclusively for System 2 agent"""

    def __init__(self):
        super().__init__(
            name="System 2 Metacognitive Tools",
            version="1.0.0",
            description="Advanced context management and loop detection tools for System 2"
        )
        self.restricted = True  # Flag to prevent System 1 access
        self.intervention_log = []
        self._system1_agent = None  # Reference to System 1 agent for message manipulation

    def register_tools(self, agent):
        """Register System 2 exclusive tools"""
        # Store reference to System 1 agent for message manipulation
        self._system1_agent = agent
        
        # NOTE: These tools are NOT registered with System 1's normal tool registry
        # Instead, they're registered with System 2's exclusive registry
        if hasattr(agent, 'register_system2_tool'):
            # Core context analysis tools
            agent.register_system2_tool(
                self.analyze_conversation_patterns,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.identify_critical_information,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            # Behavioral intervention tools
            agent.register_system2_tool(
                self.inject_metacognitive_guidance,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.detect_tool_usage_patterns,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.assess_progress_metrics,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            # Message management tools (PRUNING ONLY - no summarization)
            # ElasticSearch preserves full history, so we only need to DELETE redundant messages
            agent.register_system2_tool(
                self.delete_messages,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.extract_current_plan,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.read_message_content,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.get_active_plans,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.check_plan_progress,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )
            
            agent.register_system2_tool(
                self.auto_cleanup_old_tool_results,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )
            
            agent.register_system2_tool(
                self.prune_old_conversation_messages,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )
            
            agent.register_system2_tool(
                self.detect_and_break_tool_loop,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            logger.info("System 2 tools registered successfully")
        else:
            logger.warning("System 2 tool registration not available - falling back to regular registration")
            # Fallback to regular registration if System 2 isn't fully implemented yet
            self._register_fallback_tools(agent)

    def _register_fallback_tools(self, agent):
        """Fallback registration for testing purposes"""
        agent.register_local_tool(
            self.analyze_conversation_patterns,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

    # ===== CONTEXT ANALYSIS TOOLS =====

    def analyze_conversation_patterns(self, conversation_history: List[Dict]) -> str:
        """
        Analyze conversation for redundancy, loops, and efficiency patterns

        Args:
            conversation_history: List of conversation messages

        Returns:
            Analysis report of conversation patterns
        """
        try:
            analysis = {
                "total_messages": len(conversation_history),
                "message_breakdown": {"user": 0, "assistant": 0, "system": 0},
                "tool_calls_detected": 0,
                "repeated_phrases": [],
                "conversation_topics": [],
                "efficiency_score": 0.0
            }

            # Basic message analysis
            repeated_phrases = {}
            tool_pattern = re.compile(r'```json\s*\{.*?"name".*?\}', re.DOTALL)

            for msg in conversation_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role in analysis["message_breakdown"]:
                    analysis["message_breakdown"][role] += 1

                # Detect tool calls
                if tool_pattern.search(content):
                    analysis["tool_calls_detected"] += 1

                # Find repeated phrases (simple approach)
                words = content.lower().split()
                for i in range(len(words) - 2):
                    phrase = " ".join(words[i:i + 3])
                    if len(phrase) > 10:  # Only meaningful phrases
                        repeated_phrases[phrase] = repeated_phrases.get(phrase, 0) + 1

            # Identify highly repeated phrases
            analysis["repeated_phrases"] = [
                                               {"phrase": phrase, "count": count}
                                               for phrase, count in repeated_phrases.items()
                                               if count >= 3
                                           ][:5]  # Top 5

            # Calculate efficiency score
            if analysis["total_messages"] > 0:
                tool_ratio = analysis["tool_calls_detected"] / analysis["total_messages"]
                repetition_penalty = len(analysis["repeated_phrases"]) * 0.1
                analysis["efficiency_score"] = max(0, tool_ratio - repetition_penalty)

            # Generate report
            report = f"🧠 **CONVERSATION PATTERN ANALYSIS**\n\n"
            report += f"📊 **Message Statistics:**\n"
            report += f"• Total messages: {analysis['total_messages']}\n"
            report += f"• User: {analysis['message_breakdown']['user']}\n"
            report += f"• Assistant: {analysis['message_breakdown']['assistant']}\n"
            report += f"• System: {analysis['message_breakdown']['system']}\n"
            report += f"• Tool calls detected: {analysis['tool_calls_detected']}\n\n"

            if analysis["repeated_phrases"]:
                report += f"🔄 **Repeated Patterns:**\n"
                for item in analysis["repeated_phrases"]:
                    report += f"• \"{item['phrase'][:50]}...\" ({item['count']} times)\n"
                report += "\n"

            report += f"⚡ **Efficiency Score:** {analysis['efficiency_score']:.2f}\n"

            if analysis['efficiency_score'] < 0.3:
                report += "⚠️ **Recommendation:** Low efficiency detected - consider context compression\n"
            elif len(analysis["repeated_phrases"]) > 3:
                report += "⚠️ **Recommendation:** High repetition detected - consider loop breaking\n"
            else:
                report += "✅ **Status:** Conversation efficiency appears normal\n"

            return report

        except Exception as e:
            return f"❌ Error analyzing conversation patterns: {str(e)}"

    def compress_conversation_segment(self, messages: List[Dict], target_reduction: float = 0.7) -> str:
        """
        [DEPRECATED - NO LONGER USED WITH ELASTICSEARCH]
        Intelligently compress a segment of conversation while preserving key information
        
        With ElasticSearch offloading, we PRUNE messages instead of compressing them.
        Full conversation history is preserved in ES, so lossy compression is unnecessary.

        Args:
            messages: List of messages to compress
            target_reduction: Target reduction ratio (0.7 = reduce to 30% of original)

        Returns:
            Compressed summary of the conversation segment
        """
        try:
            if not messages:
                return "No messages to compress"

            # Extract key information
            user_requests = []
            assistant_responses = []
            tool_executions = []

            for msg in messages:
                content = msg.get("content", "")
                role = msg.get("role", "")

                if role == "user":
                    if not content.startswith("Here are the results"):
                        user_requests.append(content[:200])  # First 200 chars
                elif role == "assistant":
                    if "```json" in content:
                        # Extract tool calls
                        tool_matches = re.findall(r'"name":\s*"([^"]+)"', content)
                        if tool_matches:
                            tool_executions.extend(tool_matches)
                    else:
                        assistant_responses.append(content[:150])  # First 150 chars

            # Create intelligent summary
            summary_parts = []

            if user_requests:
                summary_parts.append(f"User requested: {'; '.join(user_requests[:3])}")

            if tool_executions:
                tool_counts = {}
                for tool in tool_executions:
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1
                tool_summary = ", ".join([f"{tool}({count})" for tool, count in tool_counts.items()])
                summary_parts.append(f"Tools used: {tool_summary}")

            if assistant_responses:
                summary_parts.append(f"Assistant provided: {assistant_responses[0]}")

            if not summary_parts:
                summary_parts.append("Various exchanges and tool executions")

            compressed_summary = f"[COMPRESSED SEGMENT: {'. '.join(summary_parts)}]"

            original_length = sum(len(msg.get("content", "")) for msg in messages)
            compressed_length = len(compressed_summary)
            reduction_achieved = 1 - (compressed_length / original_length) if original_length > 0 else 0

            result = f"✅ Compressed {len(messages)} messages\n"
            result += f"📊 Reduction: {reduction_achieved:.1%} (target: {target_reduction:.1%})\n"
            result += f"📝 Summary: {compressed_summary}"

            return result

        except Exception as e:
            return f"❌ Error compressing conversation segment: {str(e)}"

    def identify_critical_information(self, conversation_history: List[Dict]) -> str:
        """
        Identify critical information that must be preserved during compression

        Args:
            conversation_history: Full conversation history

        Returns:
            List of critical information to preserve
        """
        try:
            critical_info = {
                "user_goals": [],
                "successful_tools": [],
                "important_results": [],
                "error_patterns": [],
                "context_references": []
            }

            for i, msg in enumerate(conversation_history):
                content = msg.get("content", "")
                role = msg.get("role", "")

                # Identify user goals (questions, requests)
                if role == "user" and not content.startswith("Here are the results"):
                    if any(word in content.lower() for word in ["please", "can you", "i need", "help me"]):
                        critical_info["user_goals"].append({
                            "message_index": i,
                            "goal": content[:100],
                            "priority": "high"
                        })

                # Track successful tool executions
                if "successfully" in content.lower() or "✅" in content:
                    if "Tool" in content:
                        tool_match = re.search(r"Tool '([^']+)'", content)
                        if tool_match:
                            critical_info["successful_tools"].append({
                                "tool": tool_match.group(1),
                                "message_index": i,
                                "result_preview": content[:100]
                            })

                # Identify important results
                if any(indicator in content.lower() for indicator in ["found", "discovered", "result:", "output:"]):
                    critical_info["important_results"].append({
                        "message_index": i,
                        "content_preview": content[:100],
                        "importance": "medium"
                    })

                # Track error patterns
                if any(error_word in content.lower() for error_word in ["error", "failed", "❌"]):
                    critical_info["error_patterns"].append({
                        "message_index": i,
                        "error_preview": content[:100]
                    })

            # Generate critical information report
            report = "🎯 **CRITICAL INFORMATION IDENTIFICATION**\n\n"

            if critical_info["user_goals"]:
                report += f"🎯 **User Goals ({len(critical_info['user_goals'])}):**\n"
                for goal in critical_info["user_goals"][:3]:
                    report += f"• [{goal['message_index']}] {goal['goal']}\n"
                report += "\n"

            if critical_info["successful_tools"]:
                report += f"✅ **Successful Tools ({len(critical_info['successful_tools'])}):**\n"
                for tool in critical_info["successful_tools"][:5]:
                    report += f"• {tool['tool']} (msg {tool['message_index']})\n"
                report += "\n"

            if critical_info["error_patterns"]:
                report += f"❌ **Error Patterns ({len(critical_info['error_patterns'])}):**\n"
                for error in critical_info["error_patterns"][:3]:
                    report += f"• [{error['message_index']}] {error['error_preview']}\n"
                report += "\n"

            preservation_count = (len(critical_info["user_goals"]) +
                                  len(critical_info["successful_tools"]) +
                                  len(critical_info["important_results"]))

            report += f"📋 **Summary:** {preservation_count} critical items identified for preservation"

            return report

        except Exception as e:
            return f"❌ Error identifying critical information: {str(e)}"

    # ===== LOOP DETECTION AND BREAKING TOOLS =====

    def detect_tool_usage_patterns(self, recent_tools: List[str], threshold: int = 3) -> str:
        """
        Detect problematic tool usage patterns

        Args:
            recent_tools: List of recently used tool names
            threshold: Number of repetitions to consider problematic

        Returns:
            Pattern analysis and recommendations
        """
        try:
            if not recent_tools:
                return "No recent tool usage to analyze"

            # Count tool frequencies
            tool_counts = {}
            for tool in recent_tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

            # Detect sequential repetitions
            sequential_patterns = []
            current_tool = None
            current_count = 0

            for tool in recent_tools:
                if tool == current_tool:
                    current_count += 1
                else:
                    if current_count >= threshold:
                        sequential_patterns.append({
                            "tool": current_tool,
                            "consecutive_count": current_count
                        })
                    current_tool = tool
                    current_count = 1

            # Check final sequence
            if current_count >= threshold:
                sequential_patterns.append({
                    "tool": current_tool,
                    "consecutive_count": current_count
                })

            # Detect alternating patterns
            alternating_patterns = []
            if len(recent_tools) >= 6:
                for i in range(len(recent_tools) - 3):
                    pattern = recent_tools[i:i + 4]
                    if pattern[0] == pattern[2] and pattern[1] == pattern[3]:
                        alternating_patterns.append({
                            "pattern": f"{pattern[0]} ↔ {pattern[1]}",
                            "position": i
                        })

            # Generate report
            report = "🔍 **TOOL USAGE PATTERN ANALYSIS**\n\n"
            report += f"📊 **Recent Tools:** {', '.join(recent_tools[-10:])}\n\n"

            # Overall frequency analysis
            frequent_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            report += f"🔢 **Tool Frequency:**\n"
            for tool, count in frequent_tools:
                report += f"• {tool}: {count} times\n"
            report += "\n"

            # Sequential repetition analysis
            if sequential_patterns:
                report += f"🔄 **Sequential Repetitions Detected:**\n"
                for pattern in sequential_patterns:
                    report += f"• {pattern['tool']}: {pattern['consecutive_count']} consecutive uses\n"
                report += "⚠️ **Recommendation:** Break tool loop with alternative approach\n\n"

            # Alternating pattern analysis
            if alternating_patterns:
                report += f"↔️ **Alternating Patterns Detected:**\n"
                for pattern in alternating_patterns[:3]:
                    report += f"• {pattern['pattern']} (position {pattern['position']})\n"
                report += "⚠️ **Recommendation:** Try different tool combination\n\n"

            # Overall assessment
            if sequential_patterns or alternating_patterns:
                report += "🚨 **Status:** Problematic patterns detected - intervention recommended"
            elif max(tool_counts.values()) > len(recent_tools) * 0.6:
                report += "⚠️ **Status:** High tool repetition - consider diversification"
            else:
                report += "✅ **Status:** Tool usage patterns appear healthy"

            return report

        except Exception as e:
            return f"❌ Error detecting tool patterns: {str(e)}"

    def inject_metacognitive_guidance(self, intervention_type: str, context: Dict[str, Any]) -> str:
        """
        Generate metacognitive guidance message for System 1

        Args:
            intervention_type: Type of intervention needed
            context: Additional context for guidance generation

        Returns:
            Formatted guidance message for injection into conversation
        """
        try:
            guidance_templates = {
                "loop_breaking": [
                    "You've been using the same approach repeatedly. Try a different strategy or ask the user for clarification.",
                    "The current tool pattern isn't making progress. Consider breaking down the task differently.",
                    "Multiple attempts with the same tool suggest you need a new approach. Step back and reassess."
                ],
                "stagnation": [
                    "You've executed many tools but progress is unclear. Summarize what you've learned so far.",
                    "Consider asking the user for clarification or feedback on your current approach.",
                    "You may be overcomplicating the task. Try a simpler, more direct approach."
                ],
                "token_management": [
                    "Context is getting full. Focus on the most important information going forward.",
                    "Summarize previous progress before continuing with new tool executions.",
                    "Prioritize essential tools only - avoid exploratory or redundant tool calls."
                ],
                "error_recovery": [
                    "Multiple tool errors detected. Verify your approach and parameters before continuing.",
                    "Consider using simpler, more reliable tools instead of complex ones.",
                    "Break down the current task into smaller, more manageable steps."
                ]
            }

            # Select appropriate guidance
            guidance_options = guidance_templates.get(intervention_type, ["Consider a different approach."])

            # Add context-specific elements
            context_info = ""
            if context.get("consecutive_tools"):
                context_info += f" (Tool '{context['consecutive_tools']}' used {context.get('count', 0)} times)"
            if context.get("error_rate"):
                context_info += f" (Error rate: {context['error_rate']:.1%})"

            # Format the guidance message
            guidance_text = guidance_options[0] + context_info

            guidance_message = f"""<metacognitive_guidance>
🧠 System 2 Intervention ({intervention_type}):

{guidance_text}

This guidance is being provided because your current execution pattern suggests intervention is needed. Please adjust your approach accordingly.
</metacognitive_guidance>"""

            # Log the intervention
            self.intervention_log.append({
                "timestamp": time.time(),
                "type": intervention_type,
                "guidance": guidance_text,
                "context": context
            })

            return guidance_message

        except Exception as e:
            return f"❌ Error generating metacognitive guidance: {str(e)}"

    # ===== STRATEGIC ASSESSMENT TOOLS =====

    def assess_progress_metrics(self, system1_state: Dict[str, Any]) -> str:
        """
        Assess System 1's progress toward the user's goals

        Args:
            system1_state: Current state metrics from System 1

        Returns:
            Progress assessment and recommendations
        """
        try:
            assessment = {
                "progress_score": 0.0,
                "efficiency_rating": "unknown",
                "bottlenecks": [],
                "recommendations": []
            }

            # Calculate progress score based on multiple factors
            factors = []

            # Tool success rate
            if system1_state.get("total_tool_calls", 0) > 0:
                success_rate = 1 - system1_state.get("recent_error_rate", 0)
                factors.append(success_rate * 0.3)  # 30% weight

                if success_rate < 0.7:
                    assessment["bottlenecks"].append("High tool error rate")

            # Stagnation penalty
            stagnation_factor = max(0, 1 - (system1_state.get("tools_since_progress", 0) / 10))
            factors.append(stagnation_factor * 0.4)  # 40% weight

            if system1_state.get("tools_since_progress", 0) > 5:
                assessment["bottlenecks"].append("Tools not producing measurable progress")

            # Loop penalty
            consecutive_tools = system1_state.get("consecutive_identical_tools", 0)
            loop_factor = max(0, 1 - (consecutive_tools / 8))
            factors.append(loop_factor * 0.3)  # 30% weight

            if consecutive_tools > 3:
                assessment["bottlenecks"].append("Repetitive tool usage pattern")

            # Calculate overall progress score
            if factors:
                assessment["progress_score"] = sum(factors) / len(factors)

            # Determine efficiency rating
            if assessment["progress_score"] > 0.8:
                assessment["efficiency_rating"] = "excellent"
            elif assessment["progress_score"] > 0.6:
                assessment["efficiency_rating"] = "good"
            elif assessment["progress_score"] > 0.4:
                assessment["efficiency_rating"] = "fair"
            else:
                assessment["efficiency_rating"] = "poor"

            # Generate recommendations
            if assessment["progress_score"] < 0.5:
                assessment["recommendations"].append("Consider asking user for clarification")
                assessment["recommendations"].append("Simplify the current approach")

            if consecutive_tools > 2:
                assessment["recommendations"].append("Try alternative tools or methods")

            if system1_state.get("recent_error_rate", 0) > 0.3:
                assessment["recommendations"].append("Focus on more reliable tools")

            # Generate report
            report = "📈 **PROGRESS ASSESSMENT REPORT**\n\n"
            report += f"🎯 **Progress Score:** {assessment['progress_score']:.2f}/1.0\n"
            report += f"⚡ **Efficiency Rating:** {assessment['efficiency_rating'].upper()}\n\n"

            if assessment["bottlenecks"]:
                report += f"🚧 **Identified Bottlenecks:**\n"
                for bottleneck in assessment["bottlenecks"]:
                    report += f"• {bottleneck}\n"
                report += "\n"

            if assessment["recommendations"]:
                report += f"💡 **Recommendations:**\n"
                for rec in assessment["recommendations"]:
                    report += f"• {rec}\n"
                report += "\n"

            # System 1 state details
            report += f"📊 **Current Metrics:**\n"
            report += f"• Total tool calls: {system1_state.get('total_tool_calls', 0)}\n"
            report += f"• Consecutive identical tools: {consecutive_tools}\n"
            report += f"• Tools since progress: {system1_state.get('tools_since_progress', 0)}\n"
            report += f"• Recent error rate: {system1_state.get('recent_error_rate', 0):.1%}\n"
            report += f"• Token usage: {system1_state.get('token_usage_percent', 0):.1%}"

            return report

        except Exception as e:
            return f"❌ Error assessing progress metrics: {str(e)}"

    def generate_strategic_summary(self, conversation_history: List[Dict], focus: str = "comprehensive") -> str:
        """
        [DEPRECATED - NO LONGER USED WITH ELASTICSEARCH]
        Generate a high-level strategic summary of the session
        
        With ElasticSearch, semantic search retrieves relevant context on-demand.
        Manual summarization is unnecessary since full history is preserved.

        Args:
            conversation_history: Full conversation history
            focus: Summary focus ("comprehensive", "goals", "tools", "outcomes")

        Returns:
            Strategic summary of the session
        """
        try:
            summary_data = {
                "session_start": datetime.now().isoformat(),
                "total_exchanges": len(conversation_history),
                "user_goals": [],
                "tools_used": [],
                "outcomes_achieved": [],
                "unresolved_issues": []
            }

            # Extract information based on focus
            for i, msg in enumerate(conversation_history):
                content = msg.get("content", "")
                role = msg.get("role", "")

                if focus in ["comprehensive", "goals"] and role == "user":
                    if not content.startswith("Here are the results"):
                        # This is likely a user request/goal
                        summary_data["user_goals"].append({
                            "index": i,
                            "request": content[:100] + "..." if len(content) > 100 else content
                        })

                if focus in ["comprehensive", "tools"]:
                    # Extract tool usage
                    tool_matches = re.findall(r'"name":\s*"([^"]+)"', content)
                    for tool in tool_matches:
                        if tool not in summary_data["tools_used"]:
                            summary_data["tools_used"].append(tool)

                if focus in ["comprehensive", "outcomes"]:
                    # Look for successful outcomes
                    if "successfully" in content.lower() or "completed" in content.lower():
                        summary_data["outcomes_achieved"].append({
                            "index": i,
                            "outcome": content[:80] + "..." if len(content) > 80 else content
                        })

                    # Look for unresolved issues
                    if "error" in content.lower() or "failed" in content.lower():
                        summary_data["unresolved_issues"].append({
                            "index": i,
                            "issue": content[:80] + "..." if len(content) > 80 else content
                        })

            # Generate summary based on focus
            report = f"📋 **STRATEGIC SESSION SUMMARY** ({focus.upper()})\n\n"

            if focus in ["comprehensive", "goals"]:
                report += f"🎯 **User Goals ({len(summary_data['user_goals'])}):**\n"
                for goal in summary_data["user_goals"][:5]:
                    report += f"• [{goal['index']}] {goal['request']}\n"
                report += "\n"

            if focus in ["comprehensive", "tools"]:
                report += f"🔧 **Tools Utilized ({len(summary_data['tools_used'])}):**\n"
                for tool in summary_data["tools_used"][:10]:
                    report += f"• {tool}\n"
                report += "\n"

            if focus in ["comprehensive", "outcomes"]:
                report += f"✅ **Outcomes Achieved ({len(summary_data['outcomes_achieved'])}):**\n"
                for outcome in summary_data["outcomes_achieved"][:5]:
                    report += f"• [{outcome['index']}] {outcome['outcome']}\n"
                report += "\n"

                if summary_data["unresolved_issues"]:
                    report += f"❌ **Unresolved Issues ({len(summary_data['unresolved_issues'])}):**\n"
                    for issue in summary_data["unresolved_issues"][:3]:
                        report += f"• [{issue['index']}] {issue['issue']}\n"
                    report += "\n"

            # Overall session metrics
            report += f"📊 **Session Metrics:**\n"
            report += f"• Total exchanges: {summary_data['total_exchanges']}\n"
            report += f"• Unique tools used: {len(summary_data['tools_used'])}\n"
            report += f"• Successful outcomes: {len(summary_data['outcomes_achieved'])}\n"
            report += f"• Issues encountered: {len(summary_data['unresolved_issues'])}\n"

            # Calculate success rate
            if summary_data['user_goals']:
                success_rate = len(summary_data['outcomes_achieved']) / len(summary_data['user_goals'])
                report += f"• Estimated success rate: {success_rate:.1%}"

            return report

        except Exception as e:
            return f"❌ Error generating strategic summary: {str(e)}"

    # ===== MESSAGE MANIPULATION TOOLS =====

    def delete_messages(self, message_indices: List[int]) -> str:
        """
        Delete specific messages from System 1's conversation history
        
        Automatically archives tool results to ElasticSearch before deletion
        so System 1 can retrieve them later if needed.
        
        Args:
            message_indices: List of message indices to delete (0-based)
            
        Returns:
            Status of deletion operation
        """
        try:
            if not hasattr(self, '_system1_agent'):
                return "❌ Error: System 1 agent reference not available"
            
            # Check if we have memory manager for archiving
            has_memory = (hasattr(self._system1_agent, 'memory_manager') and 
                         self._system1_agent.memory_manager is not None)
            
            # Sort indices in reverse order to delete from end to beginning
            sorted_indices = sorted(message_indices, reverse=True)
            deleted_count = 0
            archived_count = 0
            errors = []
            
            for idx in sorted_indices:
                try:
                    if 0 <= idx < len(self._system1_agent.conversation_history):
                        # Don't delete the system message (index 0)
                        if idx == 0:
                            errors.append(f"Cannot delete system message at index {idx}")
                            continue
                        
                        msg = self._system1_agent.conversation_history[idx]
                        
                        # Archive if this looks like a tool result message
                        if has_memory and self._is_tool_result_message(msg):
                            archived = self._archive_tool_result_message(idx, msg)
                            if archived:
                                archived_count += 1
                        
                        # Delete the message
                        deleted_msg = self._system1_agent.conversation_history.pop(idx)
                        deleted_count += 1
                        logger.info(f"System 2: Deleted message at index {idx} (role: {deleted_msg.get('role', 'unknown')})")
                    else:
                        errors.append(f"Index {idx} out of range")
                except Exception as e:
                    errors.append(f"Failed to delete index {idx}: {str(e)}")
            
            result = f"✅ Successfully deleted {deleted_count} message(s)"
            if archived_count > 0:
                result += f"\n📦 Archived {archived_count} tool result(s) to ElasticSearch"
            if errors:
                result += f"\n⚠️ Errors: {'; '.join(errors)}"
            
            return result
            
        except Exception as e:
            return f"❌ Error deleting messages: {str(e)}"
    
    def _is_tool_result_message(self, message: Dict[str, Any]) -> bool:
        """Check if a message contains tool result data"""
        content = message.get('content', '')
        role = message.get('role', '')
        
        # Check for common tool result patterns
        # NOTE: The actual conversation_history stores tool results as "Tool execution results:"
        # The "📊 RAW RESULTS:" is just console output, not stored in history!
        tool_indicators = [
            'Tool execution results:',  # ACTUAL format stored in conversation_history
            'Tool result:',             # System 2's own tool results
            '{"tool_name"',             # JSON tool results
            '"tool_result"',
            'Here are the results',     # Legacy format
            'RAW RESULTS:',             # Sometimes in display (but check anyway)
        ]
        
        return role in ['user', 'assistant'] and any(indicator in content for indicator in tool_indicators)
    
    def _archive_tool_result_message(self, index: int, message: Dict[str, Any]) -> bool:
        """Extract and archive tool result from a message"""
        try:
            content = message.get('content', '')
            metadata = message.get('metadata', {})
            
            # Try to get tool names from metadata (most accurate)
            tools_used = metadata.get('tools_used', [])
            if tools_used:
                # If multiple tools, join them
                tool_name = ", ".join(tools_used) if len(tools_used) > 1 else tools_used[0]
            else:
                # Fall back to pattern matching
                tool_name = "unknown_tool"
                
                # Parse different tool result formats
                if 'Tool execution results:' in content:
                    tool_result = content.split('Tool execution results:', 1)[1].strip()
                    
                    # Try to extract tool name from the result content
                    if '🔍 Search Results for:' in tool_result:
                        tool_name = "web_search"
                    elif 'File content:' in tool_result:
                        tool_name = "read_file"
                    elif 'Files and folders:' in tool_result:
                        tool_name = "list_directory"
                    elif '📝 Memory stored' in tool_result:
                        tool_name = "store_memory"
                    elif 'Screenshot saved' in tool_result:
                        tool_name = "take_screenshot"
                else:
                    tool_result = content
            
            tool_args = {}
            tool_result = content.split('Tool execution results:', 1)[1].strip() if 'Tool execution results:' in content else content
            
            # Archive to memory manager
            memory_mgr = self._system1_agent.memory_manager
            success = memory_mgr.store_tool_result_archive(
                message_index=index,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                metadata={
                    'role': message.get('role', 'unknown'),
                    'archived_at': time.time()  # Use Unix timestamp (float)
                }
            )
            
            if success:
                logger.info(f"System 2: Archived tool result from message {index}: {tool_name}")
            
            return success
            
        except Exception as e:
            logger.warning(f"Failed to archive tool result from message {index}: {e}")
            return False

    def summarize_and_replace_messages(self, start_index: int, end_index: int, summary: str = None) -> str:
        """
        [DEPRECATED - NO LONGER USED WITH ELASTICSEARCH]
        Summarize a range of messages and replace them with a single summary message
        
        With ElasticSearch offloading, use delete_messages() instead of summarization.
        Full conversation history is preserved in ES, so lossy summarization is unnecessary.
        
        Args:
            start_index: Starting message index (inclusive, 0-based)
            end_index: Ending message index (inclusive, 0-based)
            summary: Optional custom summary. If None, auto-generate from messages
            
        Returns:
            Status of summarization operation
        """
        try:
            if not hasattr(self, '_system1_agent'):
                return "❌ Error: System 1 agent reference not available"
            
            history = self._system1_agent.conversation_history
            
            # Validate indices
            if start_index < 1:  # Don't allow summarizing system message
                return "❌ Error: Cannot summarize system message (index 0)"
            
            if start_index >= end_index:
                return "❌ Error: start_index must be less than end_index"
            
            if end_index >= len(history):
                return f"❌ Error: end_index {end_index} out of range (max: {len(history)-1})"
            
            # Extract messages to summarize
            messages_to_summarize = history[start_index:end_index+1]
            
            # Generate summary if not provided
            if summary is None:
                summary = self._auto_generate_summary(messages_to_summarize)
            
            # Create summary message
            summary_message = {
                "role": "system",
                "content": f"[CONTEXT SUMMARY - Messages {start_index}-{end_index}]: {summary}"
            }
            
            # Calculate token savings
            original_tokens = sum(len(msg.get('content', '')) // 4 for msg in messages_to_summarize)
            summary_tokens = len(summary) // 4
            tokens_saved = original_tokens - summary_tokens
            
            # Replace messages with summary
            self._system1_agent.conversation_history = (
                history[:start_index] + 
                [summary_message] + 
                history[end_index+1:]
            )
            
            logger.info(f"System 2: Summarized messages {start_index}-{end_index} (saved ~{tokens_saved} tokens)")
            
            return (f"✅ Successfully summarized {end_index - start_index + 1} messages\n"
                   f"📊 Token savings: ~{tokens_saved} tokens\n"
                   f"📝 Summary: {summary[:100]}...")
            
        except Exception as e:
            return f"❌ Error summarizing messages: {str(e)}"

    def extract_current_plan(self) -> str:
        """
        Extract the current System 1 plan or active task from conversation history
        
        Returns:
            Current plan/task description or indication that no plan exists
        """
        try:
            if not hasattr(self, '_system1_agent'):
                return "❌ Error: System 1 agent reference not available"
            
            history = self._system1_agent.conversation_history
            
            # Look for recent user requests and assistant acknowledgments
            plan_indicators = [
                r"(?:plan|task|goal|objective|working on|need to|trying to|attempting to)[\s:]+(.{20,200})",
                r"(?:I will|I'll|Let me|I'm going to)[\s]+(.{20,150})",
                r"(?:user (?:wants|needs|requests|asks))[\s]+(.{20,150})"
            ]
            
            # Search from most recent messages backwards
            for i in range(len(history) - 1, max(0, len(history) - 20), -1):
                msg = history[i]
                content = msg.get('content', '').lower()
                role = msg.get('role', '')
                
                for pattern in plan_indicators:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        plan_text = matches[0].strip()
                        return (f"🎯 Current Plan (from message {i}, {role}):\n"
                               f"{plan_text[:300]}")
            
            # Look for the most recent user request
            for i in range(len(history) - 1, -1, -1):
                msg = history[i]
                if msg.get('role') == 'user' and not msg.get('content', '').startswith('Here are the results'):
                    user_request = msg.get('content', '')[:200]
                    return (f"🎯 Current Task (most recent user request):\n"
                           f"{user_request}")
            
            return "ℹ️ No clear plan or task identified in recent conversation"
            
        except Exception as e:
            return f"❌ Error extracting plan: {str(e)}"

    def _auto_generate_summary(self, messages: List[Dict]) -> str:
        """
        Auto-generate a summary from a list of messages
        
        Args:
            messages: List of message dictionaries to summarize
            
        Returns:
            Generated summary string
        """
        user_requests = []
        tool_executions = {}
        results_found = []
        errors = []
        
        for msg in messages:
            content = msg.get('content', '')
            role = msg.get('role', '')
            
            if role == 'user':
                if not content.startswith('Here are the results'):
                    user_requests.append(content[:80])
            elif role == 'assistant':
                # Extract tool calls
                tool_matches = re.findall(r'"name":\s*"([^"]+)"', content)
                for tool in tool_matches:
                    tool_executions[tool] = tool_executions.get(tool, 0) + 1
            
            # Look for results
            if 'found' in content.lower() or 'result' in content.lower():
                results_found.append(content[:60])
            
            # Look for errors
            if 'error' in content.lower() or '❌' in content:
                errors.append(content[:60])
        
        summary_parts = []
        
        if user_requests:
            summary_parts.append(f"User: {user_requests[0]}")
        
        if tool_executions:
            tools_str = ', '.join([f"{tool}({count}x)" for tool, count in list(tool_executions.items())[:5]])
            summary_parts.append(f"Tools: {tools_str}")
        
        if results_found:
            summary_parts.append(f"Results: {results_found[0]}")
        
        if errors:
            summary_parts.append(f"Errors: {len(errors)} encountered")
        
        return '. '.join(summary_parts) if summary_parts else "Various tool executions and exchanges"

    def read_message_content(self, message_index: int) -> str:
        """
        Read the full content of a specific message (for when System 2 needs details)
        
        Args:
            message_index: Index of message to read (0-based)
            
        Returns:
            Full message content with metadata
        """
        try:
            if not hasattr(self, '_system1_agent'):
                return "❌ Error: System 1 agent reference not available"
            
            history = self._system1_agent.conversation_history
            
            if message_index < 0 or message_index >= len(history):
                return f"❌ Error: Index {message_index} out of range (0-{len(history)-1})"
            
            msg = history[message_index]
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            token_estimate = len(content) // 4
            
            return (
                f"📄 Message [{message_index}] - Role: {role} (~{token_estimate} tokens)\n"
                f"{'='*60}\n"
                f"{content}\n"
                f"{'='*60}"
            )
            
        except Exception as e:
            return f"❌ Error reading message content: {str(e)}"

    # ===== PLAN TRACKING TOOLS =====

    def get_active_plans(self, max_results: int = None) -> str:
        """
        Get all active (incomplete) plans from ElasticSearch.
        System 2 tool for tracking System 1's progress on multi-step tasks.
        
        Args:
            max_results: Max plans to return (default 20 for System 2 overview)
        
        Returns:
            Summary of active plans with progress
        """
        try:
            if not hasattr(self, '_system1_agent'):
                return "❌ Error: System 1 agent reference not available"
            
            if not hasattr(self._system1_agent, 'memory_manager'):
                return "❌ Error: Memory manager not available"
            
            # System 2 gets more plans for better oversight
            if max_results is None:
                max_results = 20
            
            plans = self._system1_agent.memory_manager.get_active_plans(max_results=max_results)
            
            if not plans:
                return "✅ No active plans found - System 1 has no pending multi-step tasks"
            
            result = []
            result.append("🧠 **SYSTEM 2: ACTIVE PLAN TRACKING**")
            result.append("=" * 60)
            result.append(f"Found {len(plans)} active plans that need attention")
            result.append("")
            
            for i, plan in enumerate(plans, 1):
                completed = plan['completed_steps']
                total = plan['total_steps']
                progress = (completed / total * 100) if total > 0 else 0
                
                result.append(f"{i}. **{plan['description']}**")
                result.append(f"   ID: {plan['plan_id']}")
                result.append(f"   Status: {plan['status']} | Progress: {completed}/{total} ({progress:.0f}%)")
                
                # Show pending steps
                pending_steps = [s for s in plan['steps'] if s['status'] != 'completed']
                if pending_steps:
                    result.append(f"   Next steps:")
                    for step in pending_steps[:2]:
                        result.append(f"   - [{step['status']}] {step['description']}")
                    if len(pending_steps) > 2:
                        result.append(f"   ... and {len(pending_steps) - 2} more pending")
                
                result.append("")
            
            result.append("=" * 60)
            result.append("💡 **RECOMMENDATION:** Consider reminding System 1 about incomplete plans")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"❌ Error getting active plans: {str(e)}"
    
    def check_plan_progress(self, plan_id: str) -> str:
        """
        Check detailed progress of a specific plan
        
        Args:
            plan_id: The plan ID to check
            
        Returns:
            Detailed plan status with all steps
        """
        try:
            if not hasattr(self, '_system1_agent'):
                return "❌ Error: System 1 agent reference not available"
            
            if not hasattr(self._system1_agent, 'memory_manager'):
                return "❌ Error: Memory manager not available"
            
            plan = self._system1_agent.memory_manager.get_plan(plan_id)
            
            if not plan:
                return f"❌ Plan not found: {plan_id}"
            
            result = []
            result.append("🧠 **SYSTEM 2: PLAN PROGRESS CHECK**")
            result.append("=" * 60)
            result.append(f"Plan: {plan['description']}")
            result.append(f"ID: {plan['plan_id']}")
            result.append(f"Status: {plan['status']}")
            result.append(f"Progress: {plan['completed_steps']}/{plan['total_steps']} steps")
            result.append(f"Created: {plan['created_at']}")
            result.append(f"Updated: {plan['updated_at']}")
            result.append("")
            result.append("**All Steps:**")
            
            for step in sorted(plan['steps'], key=lambda x: x.get('order', 0)):
                status_icon = "✅" if step['status'] == 'completed' else "🔄" if step['status'] in ['in_progress', 'in-progress'] else "⏸️"
                result.append(f"{step['order']}. {status_icon} [{step['status']}] {step['description']}")
            
            result.append("")
            result.append("=" * 60)
            
            # Add recommendations
            pending = sum(1 for s in plan['steps'] if s['status'] not in ['completed', 'in_progress', 'in-progress'])
            if pending > 0:
                result.append(f"💡 **RECOMMENDATION:** {pending} steps still pending - may need reminder to System 1")
            elif all(s['status'] == 'completed' for s in plan['steps']):
                result.append(f"💡 **RECOMMENDATION:** All steps complete - consider marking plan as completed")
            else:
                result.append(f"💡 **STATUS:** Plan is actively in progress")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"❌ Error checking plan progress: {str(e)}"
    
    def auto_cleanup_old_tool_results(self, keep_recent: int = 2, aggressive: bool = False) -> str:
        """
        Automatically identify and remove old tool result messages
        
        This is a direct, non-LLM tool that scans conversation history,
        identifies tool result patterns, and removes old ones while keeping
        recent results for context. Also deduplicates identical results.
        
        Args:
            keep_recent: Number of most recent tool results to keep (default: 2)
            aggressive: If True, deduplicate identical results and be more ruthless (default: False)
            
        Returns:
            Status report of cleanup operation
        """
        try:
            if not hasattr(self, '_system1_agent'):
                return "❌ Error: System 1 agent reference not available"
            
            history = self._system1_agent.conversation_history
            total_messages = len(history)
            
            logger.info(f"🔍 DEBUG auto_cleanup: history len={len(history)}, aggressive={aggressive}")
            
            # AGGRESSIVE MODE: If we have 50+ messages, be more ruthless
            if aggressive or total_messages >= 50:
                logger.info(f"🚨 Aggressive cleanup mode activated ({total_messages} messages)")
                keep_recent = min(keep_recent, 2)  # Force smaller keep count
            
            # Find all tool result messages with their indices
            tool_result_indices = []
            result_signatures = {}  # Track duplicate results
            
            for idx, msg in enumerate(history):
                if idx == 0:  # Skip system message
                    continue
                
                is_tool_result = self._is_tool_result_message(msg)
                if is_tool_result:
                    content = msg.get('content', '')
                    token_count = len(content) // 4
                    
                    # Create signature for deduplication (first 200 chars)
                    signature = content[:200].strip()
                    
                    tool_result_indices.append({
                        'index': idx,
                        'tokens': token_count,
                        'role': msg.get('role', 'unknown'),
                        'signature': signature,
                        'is_duplicate': signature in result_signatures
                    })
                    
                    # Track first occurrence of this signature
                    if signature not in result_signatures:
                        result_signatures[signature] = idx
                    
                    logger.info(f"  ✅ Found tool result at index {idx} ({token_count} tokens) duplicate={signature in result_signatures}")
            
            if not tool_result_indices:
                logger.info(f"⚠️ No tool result messages detected in {len(history)} messages")
                return "✅ No tool result messages found to clean up"
            
            # DEDUPLICATION: Remove exact duplicate results (keep first occurrence)
            duplicates_to_remove = []
            if aggressive:
                duplicates_to_remove = [item for item in tool_result_indices if item['is_duplicate']]
                logger.info(f"📊 Found {len(duplicates_to_remove)} duplicate tool results to remove")
            
            # Determine what to delete
            to_delete = []
            
            # Add duplicates first
            to_delete.extend(duplicates_to_remove)
            
            # Add old results (keep only recent N, excluding duplicates we're already deleting)
            non_duplicates = [item for item in tool_result_indices if not item['is_duplicate']]
            if len(non_duplicates) > keep_recent:
                old_results = non_duplicates[:-keep_recent]
                to_delete.extend(old_results)
            
            # Remove duplicates from to_delete list
            to_delete_unique = []
            seen_indices = set()
            for item in to_delete:
                if item['index'] not in seen_indices:
                    to_delete_unique.append(item)
                    seen_indices.add(item['index'])
            to_delete = to_delete_unique
            
            if not to_delete:
                return f"✅ Only {len(tool_result_indices)} tool result(s) found - all kept for context"
            
            # Archive and delete
            indices_to_delete = [item['index'] for item in to_delete]
            tokens_to_free = sum(item['tokens'] for item in to_delete)
            duplicate_count = len(duplicates_to_remove)
            
            # Archive each before deletion (if memory manager available)
            archived_count = 0
            if hasattr(self._system1_agent, 'memory_manager') and self._system1_agent.memory_manager:
                for item in to_delete:
                    idx = item['index']
                    msg = history[idx]
                    if self._archive_tool_result_message(idx, msg):
                        archived_count += 1
            
            # Delete (in reverse order to preserve indices)
            deleted_count = 0
            errors = []
            for idx in sorted(indices_to_delete, reverse=True):
                try:
                    deleted_msg = history.pop(idx)
                    deleted_count += 1
                    logger.info(f"System 2: Auto-deleted tool result at index {idx}")
                except Exception as e:
                    logger.error(f"❌ Failed to delete index {idx}: {str(e)}")
                    errors.append(f"Failed to delete index {idx}: {str(e)}")
            
            # Build report
            result = f"✅ Auto-cleanup completed:\n"
            result += f"   🗑️  Removed: {deleted_count} old tool result message(s)\n"
            if duplicate_count > 0:
                result += f"   🔁 Deduped: {duplicate_count} duplicate result(s)\n"
            result += f"   💾 Freed: ~{tokens_to_free:,} tokens\n"
            if archived_count > 0:
                result += f"   📦 Archived: {archived_count} result(s) to ElasticSearch\n"
            result += f"   ✅ Kept: {keep_recent} most recent tool result(s) for context"
            result += f"\n   📊 Final history length: {len(history)} messages"
            
            if errors:
                result += f"\n   ⚠️  Errors: {'; '.join(errors)}"
            
            logger.info(f"System 2: Auto-cleanup freed ~{tokens_to_free:,} tokens by removing {deleted_count} results")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in auto_cleanup_old_tool_results: {e}")
            return f"❌ Error during auto-cleanup: {str(e)}"
    
    def prune_old_conversation_messages(self, keep_recent: int = 8, aggressive: bool = False) -> str:
        """
        Aggressively prune old conversation messages (user/assistant exchanges)
        
        Removes old conversation messages that are no longer relevant to the current context.
        Unlike auto_cleanup_old_tool_results which only removes tool results, this removes
        actual user requests and assistant responses, keeping only recent exchanges.
        
        Args:
            keep_recent: Number of recent user/assistant message pairs to keep (default: 8)
            aggressive: If True, be even more ruthless - keep only 6 recent pairs (default: False)
            
        Returns:
            Status report of pruning operation
        """
        try:
            if not hasattr(self, '_system1_agent'):
                return "❌ Error: System 1 agent reference not available"
            
            history = self._system1_agent.conversation_history
            total_messages = len(history)
            
            # Adjust for aggressive mode
            if aggressive:
                keep_recent = min(keep_recent, 6)
                logger.info(f"🚨 Aggressive conversation pruning mode activated")
            
            # Identify conversation messages (user/assistant) to prune
            # We keep:
            # 1. System message (index 0)
            # 2. Recent N user/assistant messages
            # We delete:
            # 3. Old user/assistant messages in the middle
            
            # Find indices to keep
            keep_indices = {0}  # Always keep system message
            
            # Keep recent messages (last keep_recent*2 to account for user/assistant pairs)
            recent_message_count = min(keep_recent * 2, len(history) - 1)
            for i in range(len(history) - recent_message_count, len(history)):
                keep_indices.add(i)
            
            # Build list of messages to delete
            to_delete = []
            tokens_to_free = 0
            
            for idx in range(1, len(history)):  # Skip system message at 0
                if idx not in keep_indices:
                    msg = history[idx]
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    
                    # Only prune user/assistant conversation messages
                    # Don't delete system intervention messages (they're important)
                    if role in ['user', 'assistant']:
                        # Skip tool result messages - those are handled by auto_cleanup_old_tool_results
                        if not self._is_tool_result_message(msg):
                            # Skip System 2 reports - they should stay
                            if not content.startswith('🧠 SYSTEM 2:'):
                                tokens = len(content) // 4
                                to_delete.append({
                                    'index': idx,
                                    'role': role,
                                    'tokens': tokens,
                                    'preview': content[:60].replace('\n', ' ')
                                })
                                tokens_to_free += tokens
            
            if not to_delete:
                return f"✅ Only {len(history)} messages in history - no old conversations to prune"
            
            # Archive to ElasticSearch before deletion
            archived_count = 0
            if hasattr(self._system1_agent, 'memory_manager') and self._system1_agent.memory_manager:
                for item in to_delete:
                    idx = item['index']
                    msg = history[idx]
                    
                    # Store in ElasticSearch as archived conversation
                    try:
                        self._system1_agent.memory_manager.store_memory(
                            content=msg.get('content', ''),
                            memory_type="conversation_archive",
                            metadata={
                                "role": msg.get('role', 'unknown'),
                                "original_index": idx,
                                "archived_by": "system2_prune",
                                "archived_at": time.time()
                            }
                        )
                        archived_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to archive message {idx}: {e}")
            
            # Delete messages (in reverse order to preserve indices)
            deleted_count = 0
            indices_to_delete = [item['index'] for item in to_delete]
            
            for idx in sorted(indices_to_delete, reverse=True):
                try:
                    history.pop(idx)
                    deleted_count += 1
                    logger.info(f"System 2: Pruned conversation message at index {idx}")
                except Exception as e:
                    logger.error(f"Failed to delete message at {idx}: {e}")
            
            # Build report
            result = f"✅ Conversation pruning completed:\n"
            result += f"   🗑️  Removed: {deleted_count} old conversation message(s)\n"
            result += f"   💾 Freed: ~{tokens_to_free:,} tokens\n"
            if archived_count > 0:
                result += f"   📦 Archived: {archived_count} message(s) to ElasticSearch\n"
            result += f"   ✅ Kept: System message + {recent_message_count} recent message(s)\n"
            result += f"   📊 Final history length: {len(history)} messages"
            
            logger.info(f"System 2: Conversation pruning freed ~{tokens_to_free:,} tokens by removing {deleted_count} messages")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prune_old_conversation_messages: {e}")
            return f"❌ Error during conversation pruning: {str(e)}"
    
    def detect_and_break_tool_loop(self, lookback_count: int = 10) -> str:
        """
        Detect if System 1 is stuck in a tool loop and force a break
        
        Examines recent conversation history for repeated identical tool calls
        with identical or nearly identical arguments. More aggressive than
        the built-in loop detection.
        
        Args:
            lookback_count: Number of recent messages to examine (default: 10)
            
        Returns:
            Report on loop detection and actions taken
        """
        try:
            if not hasattr(self, '_system1_agent'):
                return "❌ Error: System 1 agent reference not available"
            
            history = self._system1_agent.conversation_history
            
            # Look at recent assistant messages for tool calls
            recent_tool_calls = []
            for msg in history[-lookback_count:]:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    metadata = msg.get('metadata', {})
                    
                    # Extract tool names from metadata
                    tools_used = metadata.get('tools', [])
                    if tools_used:
                        recent_tool_calls.append({
                            'tools': tools_used,
                            'content_sample': content[:100]
                        })
            
            if len(recent_tool_calls) < 3:
                return "✅ No tool loop detected (insufficient tool call history)"
            
            # Check for repeated identical tool usage
            tool_sequence = [tuple(sorted(call['tools'])) for call in recent_tool_calls]
            
            # Count consecutive repeats
            max_consecutive = 1
            current_consecutive = 1
            repeated_tools = None
            
            for i in range(1, len(tool_sequence)):
                if tool_sequence[i] == tool_sequence[i-1]:
                    current_consecutive += 1
                    if current_consecutive > max_consecutive:
                        max_consecutive = current_consecutive
                        repeated_tools = tool_sequence[i]
                else:
                    current_consecutive = 1
            
            # If we see 3+ consecutive identical tool calls, that's a loop
            if max_consecutive >= 3:
                logger.warning(f"🚨 System 2: Tool loop detected! '{repeated_tools}' called {max_consecutive} times consecutively")
                
                # Force inject a metacognitive message to break the loop
                intervention_msg = {
                    'role': 'system',
                    'content': f"<system2_intervention>LOOP DETECTED: You have called the tool(s) {repeated_tools} {max_consecutive} times consecutively with similar results. This suggests the current approach is not making progress. STOP attempting this tool and instead: 1) Acknowledge the loop, 2) Summarize what you learned from the repeated attempts, 3) Propose a DIFFERENT approach or admit the task cannot be completed with available tools.</system2_intervention>"
                }
                
                history.append(intervention_msg)
                
                return f"🚨 LOOP DETECTED AND BROKEN:\n" \
                       f"   🔁 Tool(s) '{repeated_tools}' repeated {max_consecutive} times\n" \
                       f"   ✋ Injected intervention message to force approach change\n" \
                       f"   📊 Examined last {lookback_count} messages"
            
            # Check for oscillation (A -> B -> A -> B pattern)
            if len(tool_sequence) >= 4:
                alternating_count = 0
                for i in range(2, len(tool_sequence)):
                    if tool_sequence[i] == tool_sequence[i-2] and tool_sequence[i] != tool_sequence[i-1]:
                        alternating_count += 1
                
                if alternating_count >= 2:
                    logger.warning(f"🚨 System 2: Tool oscillation detected")
                    return f"⚠️ OSCILLATION DETECTED:\n" \
                           f"   🔄 System 1 is alternating between different tools\n" \
                           f"   💡 Suggestion: This may indicate confusion or insufficient information\n" \
                           f"   📊 Examined last {lookback_count} messages"
            
            return f"✅ No problematic tool loops detected\n" \
                   f"   📊 Examined {len(recent_tool_calls)} recent tool calls\n" \
                   f"   🔁 Max consecutive repeats: {max_consecutive}"
            
        except Exception as e:
            logger.error(f"Error in detect_and_break_tool_loop: {e}")
            return f"❌ Error detecting tool loop: {str(e)}"


# Plugin factory
def create_plugin():
    """Factory function to create the plugin instance"""
    return System2ToolsPlugin()


# For testing
if __name__ == "__main__":
    plugin = System2ToolsPlugin()
    print(f"System 2 Tools Plugin: {plugin.name} v{plugin.version}")
    print(f"Description: {plugin.description}")
    print(f"Restricted access: {plugin.restricted}")