#!/usr/bin/env python3
"""
SearXNG Search Tool Plugin for SAM Agent
Web search powered by SearXNG metasearch engine
"""

import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("SAM.SearXNG")

# Try to import required libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available - install with: pip install requests")

from sam_agent import SAMPlugin, ToolCategory


class SearXNGPlugin(SAMPlugin):
    """Web search using SearXNG metasearch engine"""

    def __init__(self):
        super().__init__(
            name="SearXNG Search",
            version="1.0.0",
            description="Search the web using SearXNG metasearch engine"
        )
        self.config = None

    def on_load(self, agent):
        """Initialize SearXNG configuration from agent config"""
        if hasattr(agent.config, 'searxng'):
            self.config = agent.config.searxng
            logger.info(f"SearXNG tools connected to {self.config.base_url}")
        else:
            logger.warning("SearXNG config not found - using defaults")
            # Create a simple object with defaults
            class DefaultConfig:
                enabled = True
                base_url = "http://localhost:8888"
                language = "en"
                min_score = 1.0
                max_results = 10
                timeout = 15
                verify_ssl = False
            self.config = DefaultConfig()

    def register_tools(self, agent):
        """Register search tools with the agent"""
        
        agent.register_local_tool(
            self.web_search,
            category=ToolCategory.WEB,
            requires_approval=False  # Safe read-only operation
        )

        agent.register_local_tool(
            self.check_dependencies,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

    def web_search(self, query: str, 
                   max_results: Optional[int] = None,
                   min_score: Optional[float] = None,
                   language: Optional[str] = None,
                   _context_budget: dict = None) -> str:
        """
        Search the web using SearXNG and return formatted results.
        **Context-aware**: Automatically adjusts result count based on available tokens.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (automatically limited based on context)
            min_score: Minimum relevance score threshold (default: from config or 1.0)
            language: Language code for results (default: from config or 'en')

        Returns:
            Formatted search results with titles, URLs, snippets, and scores

        Examples:
            web_search("Python async programming")
            web_search("climate change")
            web_search("machine learning", min_score=2.0, language="en")
        """
        # Dependency check
        if not REQUESTS_AVAILABLE:
            return "❌ requests library not available. Install with: pip install requests"

        # Check if SearXNG is enabled
        if not getattr(self.config, 'enabled', True):
            return "❌ SearXNG search is disabled in config.json"

        # Get configuration values
        base_url = getattr(self.config, 'base_url', 'http://localhost:8888')
        language = language or getattr(self.config, 'language', 'en')
        min_score = min_score if min_score is not None else getattr(self.config, 'min_score', 1.0)
        
        # Use auto-injected limit (fallback to config or 10)
        if max_results is None:
            max_results = getattr(self.config, 'max_results', 10)
        timeout = getattr(self.config, 'timeout', 15)
        verify_ssl = getattr(self.config, 'verify_ssl', False)

        # Validate inputs
        if not query or not query.strip():
            return "❌ Search query cannot be empty"

        try:
            # Build search URL
            search_url = f"{base_url}/search"
            
            # Build parameters
            params = {
                'q': query.strip(),
                'format': 'json',
                'language': language
            }

            print(f"🔍 Searching SearXNG: '{query}'")
            print(f"   Language: {language}, Min Score: {min_score}, Max Results: {max_results}")

            # Make the request
            response = requests.get(
                search_url,
                params=params,
                timeout=timeout,
                verify=verify_ssl
            )
            response.raise_for_status()

            # Parse JSON response
            data = response.json()
            
            # Get results
            results = data.get('results', [])
            
            if not results:
                return f"📭 No results found for: '{query}'"

            # Filter by score and limit results
            filtered_results = [
                r for r in results 
                if r.get('score', 0) >= min_score
            ][:max_results]

            if not filtered_results:
                return f"📭 No results met the minimum score threshold ({min_score}) for: '{query}'"

            # Format results
            output = f"🔍 Search Results for: '{query}'\n"
            output += f"{'=' * 80}\n"
            output += f"Found {len(filtered_results)} results (filtered from {len(results)} total)\n\n"

            for idx, result in enumerate(filtered_results, 1):
                title = result.get('title', 'No title')
                url = result.get('url', 'No URL')
                content = result.get('content', '')
                score = result.get('score', 0)
                
                # Create snippet from content (limit to 200 chars)
                snippet = content[:200] + '...' if len(content) > 200 else content
                snippet = snippet.replace('\n', ' ').strip()

                output += f"{idx}. {title}\n"
                output += f"   URL: {url}\n"
                output += f"   Score: {score:.2f}\n"
                if snippet:
                    output += f"   {snippet}\n"
                output += "\n"

            # Add metadata
            query_time = data.get('number_of_results', 'unknown')
            output += f"{'=' * 80}\n"
            output += f"Total results available: {query_time}\n"
            output += f"Showing top {len(filtered_results)} results with score >= {min_score}\n"

            return output

        except requests.exceptions.Timeout:
            return f"⏱️ Search request timed out after {timeout} seconds. SearXNG may be slow or unreachable."
        
        except requests.exceptions.ConnectionError:
            return f"❌ Cannot connect to SearXNG at {base_url}. Check that the service is running and the URL is correct."
        
        except requests.exceptions.HTTPError as e:
            return f"❌ HTTP error from SearXNG: {e.response.status_code} - {e.response.reason}"
        
        except ValueError as e:
            return f"❌ Invalid JSON response from SearXNG: {str(e)}"
        
        except Exception as e:
            logger.error(f"Unexpected error in web_search: {str(e)}", exc_info=True)
            return f"❌ Unexpected error during search: {str(e)}"

    def check_dependencies(self) -> str:
        """
        Check if required dependencies are installed.

        Returns:
            Status of required dependencies
        """
        status = "SearXNG Search Tool - Dependency Status\n"
        status += "=" * 50 + "\n"
        
        status += f"requests library: {'✅ Available' if REQUESTS_AVAILABLE else '❌ Not installed'}\n"
        
        if not REQUESTS_AVAILABLE:
            status += "\nTo install missing dependencies:\n"
            status += "  pip install requests\n"
        
        # Check configuration
        if self.config:
            status += f"\nConfiguration:\n"
            status += f"  Enabled: {getattr(self.config, 'enabled', True)}\n"
            status += f"  Base URL: {getattr(self.config, 'base_url', 'Not configured')}\n"
            status += f"  Language: {getattr(self.config, 'language', 'en')}\n"
            status += f"  Min Score: {getattr(self.config, 'min_score', 1.0)}\n"
            status += f"  Max Results: {getattr(self.config, 'max_results', 10)}\n"
        else:
            status += "\n⚠️ Configuration not loaded yet\n"
        
        return status


def create_plugin():
    """Factory function to create the plugin instance"""
    return SearXNGPlugin()
