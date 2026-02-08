#!/usr/bin/env python3
"""
Context-Aware Dynamic Limits System
Automatically adjusts tool response sizes based on available token budget
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("SAM.ContextLimits")


@dataclass
class ContextBudget:
    """Current context budget information"""
    total_capacity: int  # Total context window
    currently_used: int  # Tokens already used
    remaining: int  # Tokens available
    usage_percent: float  # Percentage used (0.0-1.0)
    
    @property
    def is_constrained(self) -> bool:
        """Returns True if context is getting tight"""
        return self.usage_percent > 0.6
    
    @property
    def pressure_level(self) -> str:
        """Get pressure level: low, medium, high, critical"""
        if self.usage_percent < 0.5:
            return "low"
        elif self.usage_percent < 0.7:
            return "medium"
        elif self.usage_percent < 0.85:
            return "high"
        else:
            return "critical"


class ContextAwareLimitCalculator:
    """Calculates dynamic limits based on available context"""
    
    def __init__(self, agent):
        """
        Initialize with SAM agent reference
        
        Args:
            agent: SAMAgent instance
        """
        self.agent = agent
        
        # Default baseline limits (used when context is abundant)
        self.baseline_limits = {
            'max_results': 10,
            'max_chars': 15000,
            'max_bytes': 10000,
            'max_file_size': 10 * 1024 * 1024,  # 10MB
        }
        
        # Minimum safe limits (used when context is critical)
        self.minimum_limits = {
            'max_results': 2,
            'max_chars': 1000,
            'max_bytes': 1000,
            'max_file_size': 5000,
        }
    
    def get_current_budget(self) -> ContextBudget:
        """Get current context budget"""
        # Calculate tokens currently in use
        used_tokens = sum(
            self.agent._estimate_token_count(msg.get('content', ''))
            for msg in self.agent.conversation_history
        )
        
        total_capacity = self.agent.context_limit
        remaining = total_capacity - used_tokens
        usage_percent = used_tokens / total_capacity if total_capacity > 0 else 0
        
        return ContextBudget(
            total_capacity=total_capacity,
            currently_used=used_tokens,
            remaining=remaining,
            usage_percent=usage_percent
        )
    
    def calculate_dynamic_limit(self, limit_type: str, tool_name: str = None) -> int:
        """
        Calculate dynamic limit based on current context pressure
        
        Args:
            limit_type: Type of limit ('max_results', 'max_chars', 'max_bytes', 'max_file_size')
            tool_name: Optional tool name for logging
            
        Returns:
            Dynamically calculated limit value
        """
        budget = self.get_current_budget()
        
        baseline = self.baseline_limits.get(limit_type, 10000)
        minimum = self.minimum_limits.get(limit_type, 1000)
        
        # Calculate scaling factor based on context pressure
        # At 0% usage: factor = 1.0 (use baseline)
        # At 50% usage: factor = 0.75
        # At 70% usage: factor = 0.5
        # At 85% usage: factor = 0.25
        # At 95%+ usage: factor = 0.0 (use minimum)
        
        if budget.usage_percent < 0.5:
            # Plenty of room - use full baseline
            factor = 1.0
        elif budget.usage_percent < 0.7:
            # Getting tighter - scale down gradually
            factor = 1.0 - ((budget.usage_percent - 0.5) / 0.2) * 0.25
        elif budget.usage_percent < 0.85:
            # High pressure - scale down more aggressively
            factor = 0.75 - ((budget.usage_percent - 0.7) / 0.15) * 0.5
        elif budget.usage_percent < 0.95:
            # Critical - use minimal
            factor = 0.25 - ((budget.usage_percent - 0.85) / 0.1) * 0.25
        else:
            # Emergency - absolute minimum
            factor = 0.0
        
        # Calculate final limit
        dynamic_limit = int(minimum + (baseline - minimum) * factor)
        
        # Log the adjustment if significant
        if factor < 0.75 and tool_name:
            logger.info(
                f"📊 Context-aware limit for {tool_name}.{limit_type}: "
                f"{dynamic_limit} (budget: {budget.usage_percent:.1%}, "
                f"pressure: {budget.pressure_level})"
            )
        
        return dynamic_limit
    
    def get_tool_limits(self, tool_name: str) -> Dict[str, int]:
        """
        Get all relevant dynamic limits for a tool
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary with dynamic limit values
        """
        budget = self.get_current_budget()
        
        limits = {
            '_context_budget': {
                'total': budget.total_capacity,
                'used': budget.currently_used,
                'remaining': budget.remaining,
                'usage_percent': budget.usage_percent,
                'pressure': budget.pressure_level
            }
        }
        
        # Add dynamic limits for each type
        for limit_type in self.baseline_limits.keys():
            limits[limit_type] = self.calculate_dynamic_limit(limit_type, tool_name)
        
        return limits
    
    def inject_limits_into_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject dynamic limits into tool arguments if not already specified
        
        Args:
            tool_name: Name of the tool being called
            args: Original tool arguments
            
        Returns:
            Arguments with dynamic limits injected
        """
        # Get the actual tool function to check its signature
        tool_func = None
        
        # Check local_tools first
        if hasattr(self.agent, 'local_tools') and tool_name in self.agent.local_tools:
            tool_func = self.agent.local_tools[tool_name].get('function')
        # MCP tools don't have local functions to inspect (they're remote)
        # self.agent.mcp_tools[tool_name] is a tuple (server_name, tool_info), not a dict
        
        # Get dynamic limits
        limits = self.get_tool_limits(tool_name)
        
        # Create new args dict with injected limits
        enhanced_args = args.copy()
        
        # Get function parameters to check what it actually accepts
        # Use None to indicate "couldn't inspect", empty set to indicate "no parameters"
        accepted_params = None  # Default: inspection failed
        if tool_func:
            try:
                import inspect
                sig = inspect.signature(tool_func)
                accepted_params = set(sig.parameters.keys())  # Could be empty set for no-param functions
            except Exception as e:
                logger.debug(f"Could not inspect {tool_name} signature: {e}")
                accepted_params = None  # Explicitly mark as inspection failure
        
        # Check if tool function accepts _context_budget parameter
        if accepted_params is not None and '_context_budget' in accepted_params:
            enhanced_args['_context_budget'] = limits['_context_budget']
        
        # Map of common argument names to limit types
        arg_to_limit = {
            'max_results': 'max_results',
            'max_chars': 'max_chars',
            'max_bytes': 'max_bytes',
            'max_file_size': 'max_file_size',
            'max_size': 'max_bytes',
            'limit': 'max_results',
        }
        
        # Inject limits only if:
        # 1. Not explicitly specified by the agent
        # 2. The function actually accepts that parameter
        #    - If we couldn't inspect (None), inject it (better to try than fail)
        #    - If we did inspect, only inject if parameter exists
        for arg_name, limit_type in arg_to_limit.items():
            if arg_name not in enhanced_args and limit_type in limits:
                # Only inject if:
                # - We couldn't inspect the signature (accepted_params is None), OR
                # - The function accepts this parameter (arg_name in accepted_params)
                # DO NOT inject if we successfully inspected and parameter is not accepted
                if accepted_params is None or arg_name in accepted_params:
                    enhanced_args[arg_name] = limits[limit_type]
        
        return enhanced_args
    
    def should_truncate_result(self, result: str, tool_name: str) -> Tuple[bool, int]:
        """
        Determine if a result should be truncated based on context budget
        
        Args:
            result: The result string
            tool_name: Name of the tool
            
        Returns:
            Tuple of (should_truncate, max_length)
        """
        budget = self.get_current_budget()
        result_tokens = len(result) // 4  # Rough token estimate
        
        # If result would consume more than 20% of remaining budget, truncate
        max_allowed_tokens = int(budget.remaining * 0.2)
        
        if result_tokens > max_allowed_tokens:
            # Calculate max chars (4 chars ≈ 1 token)
            max_chars = max_allowed_tokens * 4
            logger.warning(
                f"⚠️ Result from {tool_name} too large for current context "
                f"(~{result_tokens} tokens, max: ~{max_allowed_tokens}). "
                f"Truncating to {max_chars} chars"
            )
            return True, max_chars
        
        return False, len(result)


# Example of how to update a tool to use context-aware limits
def example_context_aware_tool(query: str, max_results: int = None, 
                              _context_budget: Dict = None) -> str:
    """
    Example of a context-aware tool
    
    Args:
        query: Search query
        max_results: Maximum results (auto-limited if not specified)
        _context_budget: Automatically injected context info (internal)
        
    Returns:
        Results string
    """
    # Use the dynamically calculated limit if not explicitly specified
    if max_results is None:
        max_results = 10  # Fallback default
    
    # Tool can check context budget to adjust behavior
    if _context_budget:
        pressure = _context_budget.get('pressure', 'unknown')
        if pressure in ['high', 'critical']:
            # Return more concise results under pressure
            logger.info(f"📊 Context pressure {pressure} - returning concise results")
    
    # Tool implementation here
    results = f"Query: {query}\nReturning top {max_results} results..."
    
    return results
