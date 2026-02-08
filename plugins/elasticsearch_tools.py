#!/usr/bin/env python3
"""
ElasticSearch Tools Plugin for SAM Agent
Task-focused ElasticSearch operations with high-level workflows
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("SAM.ESTools")

try:
    import requests
    from requests.auth import HTTPBasicAuth
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available - install with: pip install requests")

from sam_agent import SAMPlugin, ToolCategory


class ElasticSearchToolsPlugin(SAMPlugin):
    """Task-focused ElasticSearch operations with high-level workflows"""

    def __init__(self):
        super().__init__(
            name="ElasticSearch Tools",
            version="2.0.0",
            description="High-level ElasticSearch management: cluster config, index management, queries, maintenance"
        )
        self.es_config = None
        self.host = None
        self.auth = None
        self.verify_ssl = False

    def on_load(self, agent):
        """Initialize ES connection and store agent reference"""
        self._agent = agent  # Store reference for accessing memory manager
        
        if hasattr(agent.config, 'elasticsearch'):
            self.es_config = agent.config.elasticsearch
            self.host = self.es_config.host
            self.auth = HTTPBasicAuth(
                self.es_config.username,
                self.es_config.password
            )
            self.verify_ssl = self.es_config.verify_ssl
            logger.info(f"ElasticSearch tools connected to {self.host}")
        else:
            logger.warning("ElasticSearch config not found - tools will not function")

    def register_tools(self, agent):
        """Register ElasticSearch tools"""

        # Simple cluster overview
        agent.register_local_tool(
            self.cluster_overview,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

        # Raw Elasticsearch API access - Kibana DevTools style
        agent.register_local_tool(
            self.es_api,
            category=ToolCategory.DATA,
            requires_approval=False
        )

        # Plan management tools
        agent.register_local_tool(
            self.store_plan,
            category=ToolCategory.DATA,
            requires_approval=False
        )

        agent.register_local_tool(
            self.update_plan_step,
            category=ToolCategory.DATA,
            requires_approval=False
        )

        agent.register_local_tool(
            self.get_active_plans,
            category=ToolCategory.DATA,
            requires_approval=False
        )

        agent.register_local_tool(
            self.get_plan_details,
            category=ToolCategory.DATA,
            requires_approval=False
        )

    def _check_connection(self) -> Optional[str]:
        """Check if ES connection is configured"""
        if not REQUESTS_AVAILABLE:
            return "❌ requests library not available. Install with: pip install requests"
        if not self.host or not self.auth:
            return "❌ ElasticSearch not configured. Check config.json elasticsearch section."
        return None

    # ===== CLUSTER OVERVIEW =====

    def cluster_overview(self) -> str:
        """
        Get comprehensive cluster overview: health, indices, datastreams, settings

        Returns:
            Complete cluster status report

        Examples:
            cluster_overview()  # See everything in one call
        """
        error = self._check_connection()
        if error:
            return error

        try:
            result = []
            result.append("=" * 60)
            result.append("🗄️  ELASTICSEARCH CLUSTER OVERVIEW")
            result.append("=" * 60)

            # Cluster health
            health_resp = requests.get(
                f"{self.host}/_cluster/health",
                auth=self.auth,
                verify=self.verify_ssl,
                timeout=10
            )

            if health_resp.status_code == 200:
                health = health_resp.json()
                status = health.get('status', 'unknown')
                status_icon = "🟢" if status == 'green' else "🟡" if status == 'yellow' else "🔴"

                result.append(f"\n{status_icon} CLUSTER HEALTH: {status.upper()}")
                result.append(f"   Cluster: {health.get('cluster_name', 'N/A')}")
                result.append(
                    f"   Nodes: {health.get('number_of_nodes', 0)} ({health.get('number_of_data_nodes', 0)} data)")
                result.append(
                    f"   Shards: {health.get('active_shards', 0)} active, {health.get('unassigned_shards', 0)} unassigned")

            # Indices summary
            indices_resp = requests.get(
                f"{self.host}/_cat/indices?format=json",
                auth=self.auth,
                verify=self.verify_ssl,
                timeout=10
            )

            if indices_resp.status_code == 200:
                indices = indices_resp.json()
                total_docs = sum(int(idx.get('docs.count', 0) or 0) for idx in indices)
                total_size = sum(self._parse_size(idx.get('store.size', '0b')) for idx in indices)

                result.append(f"\n📊 INDICES: {len(indices)} total")
                result.append(f"   Documents: {total_docs:,}")
                result.append(f"   Total size: {self._format_bytes(total_size)}")

                # Show top 5 largest indices
                sorted_indices = sorted(indices,
                                        key=lambda x: self._parse_size(x.get('store.size', '0b')),
                                        reverse=True)[:5]

                if sorted_indices:
                    result.append(f"\n   Top 5 by size:")
                    for idx in sorted_indices:
                        health_icon = "🟢" if idx.get('health') == 'green' else "🟡" if idx.get(
                            'health') == 'yellow' else "🔴"
                        result.append(
                            f"   {health_icon} {idx['index']}: {idx.get('store.size', 'N/A')} ({idx.get('docs.count', 'N/A')} docs)")

            # Datastreams summary
            ds_resp = requests.get(
                f"{self.host}/_data_stream",
                auth=self.auth,
                verify=self.verify_ssl,
                timeout=10
            )

            if ds_resp.status_code == 200:
                datastreams = ds_resp.json().get('data_streams', [])
                result.append(f"\n🗂️  DATASTREAMS: {len(datastreams)}")

                if datastreams:
                    for ds in datastreams[:5]:  # Show first 5
                        result.append(f"   • {ds['name']} (generation: {ds['generation']})")

            result.append("=" * 60)
            return "\n".join(result)

        except Exception as e:
            return f"❌ Error getting cluster overview: {e}"

    def _parse_size(self, size_str: str) -> int:
        """Parse ES size string to bytes"""
        try:
            if not size_str:
                return 0
            size_str = size_str.strip().lower()
            if size_str.endswith('kb'):
                return int(float(size_str[:-2]) * 1024)
            elif size_str.endswith('mb'):
                return int(float(size_str[:-2]) * 1024 * 1024)
            elif size_str.endswith('gb'):
                return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
            elif size_str.endswith('b'):
                return int(float(size_str[:-1]))
            return int(float(size_str))
        except:
            return 0

    def _format_bytes(self, bytes_val: int) -> str:
        """Format bytes to human readable"""
        if bytes_val < 1024:
            return f"{bytes_val}B"
        elif bytes_val < 1024 * 1024:
            return f"{bytes_val / 1024:.1f}KB"
        elif bytes_val < 1024 * 1024 * 1024:
            return f"{bytes_val / (1024 * 1024):.1f}MB"
        else:
            return f"{bytes_val / (1024 * 1024 * 1024):.1f}GB"

    def _strip_embeddings(self, data: Any) -> Any:
        """
        Recursively strip embedding fields from response data to save context space.
        Embeddings are huge arrays that bloat the context window unnecessarily.
        """
        if isinstance(data, dict):
            return {
                k: self._strip_embeddings(v) 
                for k, v in data.items() 
                if k not in ('embeddings', 'embedding', 'vector', 'vectors')
            }
        elif isinstance(data, list):
            return [self._strip_embeddings(item) for item in data]
        else:
            return data

    # ===== RAW API ACCESS =====

    def es_api(self, method: str, endpoint: str, body: Optional[str] = None) -> str:
        """
        Execute raw Elasticsearch API requests (Kibana DevTools style)
        
        Provides direct access to any Elasticsearch API endpoint without parameter limitations.
        Use this for maximum flexibility when specialized tools are too restrictive.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, HEAD)
            endpoint: ES API endpoint path (e.g., "/_search", "/my-index/_doc/1", "/_cat/indices")
            body: Optional JSON body as string (for POST/PUT requests)
        
        Returns:
            API response with formatted JSON or error details
        
        Examples:
            # Search across all indices
            es_api("GET", "/_search", '{"query": {"match_all": {}}, "size": 5}')
            
            # Get cluster health
            es_api("GET", "/_cluster/health")
            
            # Create an index with mappings
            es_api("PUT", "/my-index", '{"mappings": {"properties": {"field1": {"type": "text"}}}}')
            
            # Search specific index with aggregations
            es_api("POST", "/logs-*/_search", '{"size": 0, "aggs": {"by_status": {"terms": {"field": "status"}}}}')
            
            # Delete by query
            es_api("POST", "/my-index/_delete_by_query", '{"query": {"match": {"status": "old"}}}')
            
            # Update index settings
            es_api("PUT", "/my-index/_settings", '{"index": {"number_of_replicas": 0}}')
            
            # Get document by ID
            es_api("GET", "/my-index/_doc/abc123")
            
            # Bulk operations
            es_api("POST", "/_bulk", '{"index": {"_index": "test"}}\\n{"field": "value"}\\n')
            
            # Cat APIs for quick views
            es_api("GET", "/_cat/indices?v")
            es_api("GET", "/_cat/shards?v")
        """
        error = self._check_connection()
        if error:
            return error
        
        method = method.upper()
        if method not in ['GET', 'POST', 'PUT', 'DELETE', 'HEAD']:
            return f"❌ Invalid HTTP method: {method}. Use GET, POST, PUT, DELETE, or HEAD"
        
        # Ensure endpoint starts with /
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        
        try:
            # Parse body if provided
            json_body = None
            if body:
                try:
                    json_body = json.loads(body)
                except json.JSONDecodeError as e:
                    # For bulk API and other NDJSON formats, pass as-is
                    if '/_bulk' in endpoint or '\n' in body:
                        # Use data parameter for raw text
                        pass
                    else:
                        return f"❌ Invalid JSON body: {e}\n\nBody:\n{body}"
            
            # Build request URL
            url = f"{self.host}{endpoint}"
            
            # Execute request
            kwargs = {
                'auth': self.auth,
                'verify': self.verify_ssl,
                'timeout': 60
            }
            
            if json_body is not None:
                kwargs['json'] = json_body
            elif body:
                # Raw body for NDJSON formats
                kwargs['data'] = body
                kwargs['headers'] = {'Content-Type': 'application/x-ndjson'}
            
            response = requests.request(method, url, **kwargs)
            
            # Format response
            result = []
            result.append(f"{'='*60}")
            result.append(f"🔧 {method} {endpoint}")
            result.append(f"{'='*60}")
            result.append(f"Status: {response.status_code}")
            result.append("")
            
            # Try to parse JSON response
            try:
                json_response = response.json()
                # Strip embeddings to save context space
                json_response = self._strip_embeddings(json_response)
                result.append(json.dumps(json_response, indent=2))
            except json.JSONDecodeError:
                # Plain text response (e.g., from _cat APIs)
                result.append(response.text)
            
            result.append(f"{'='*60}")
            
            return "\n".join(result)
            
        except requests.exceptions.Timeout:
            return f"❌ Request timeout after 60 seconds for {method} {endpoint}"
        except requests.exceptions.RequestException as e:
            return f"❌ Request failed: {e}"
        except Exception as e:
            return f"❌ Error executing API call: {e}"

    # ===== PLAN MANAGEMENT =====

    def store_plan(self, description: str, steps: str) -> str:
        """
        Store a plan with steps in ElasticSearch for tracking and future reference

        Args:
            description: Overall plan description/goal
            steps: JSON string of steps array, each with 'description', 'status', 'order'
                   Example: '[{"description": "Step 1", "status": "pending", "order": 1}]'

        Returns:
            Success message with plan ID

        Examples:
            store_plan("Build authentication system", '[{"description": "Design schema", "status": "pending", "order": 1}]')
        """
        if not hasattr(self, 'es_config') or not self.es_config:
            return "❌ ElasticSearch not configured"

        # Get memory manager from agent
        if not hasattr(self, '_agent') or not hasattr(self._agent, 'memory_manager'):
            return "❌ Memory manager not available"

        try:
            # Parse steps JSON
            import json
            steps_list = json.loads(steps)

            # Validate steps format
            for step in steps_list:
                if 'description' not in step or 'status' not in step or 'order' not in step:
                    return "❌ Each step must have 'description', 'status', and 'order' fields"

            # Store plan
            plan_id = self._agent.memory_manager.store_plan(
                description=description,
                steps=steps_list,
                metadata={"stored_via": "elasticsearch_tools"}
            )

            if plan_id:
                return f"✅ Plan stored successfully!\n\nPlan ID: {plan_id}\nDescription: {description}\nSteps: {len(steps_list)}"
            else:
                return "❌ Failed to store plan"

        except json.JSONDecodeError as e:
            return f"❌ Invalid JSON for steps: {e}"
        except Exception as e:
            return f"❌ Error storing plan: {e}"

    def update_plan_step(self, plan_id: str, step_order: int, new_status: str) -> str:
        """
        Update the status of a specific step in a plan

        Args:
            plan_id: The plan ID to update
            step_order: The step order number (1-based)
            new_status: New status (pending, in_progress, completed)

        Returns:
            Success message

        Examples:
            update_plan_step("plan-abc-123", 2, "completed")
        """
        if not hasattr(self, '_agent') or not hasattr(self._agent, 'memory_manager'):
            return "❌ Memory manager not available"

        try:
            # Get current plan
            plan = self._agent.memory_manager.get_plan(plan_id)
            if not plan:
                return f"❌ Plan not found: {plan_id}"

            # Update the step
            steps = plan['steps']
            step_found = False
            for step in steps:
                if step.get('order') == step_order:
                    step['status'] = new_status
                    step_found = True
                    break

            if not step_found:
                return f"❌ Step {step_order} not found in plan"

            # Update plan
            success = self._agent.memory_manager.update_plan(plan_id, {'steps': steps})

            if success:
                return f"✅ Step {step_order} updated to '{new_status}' in plan: {plan['description']}"
            else:
                return "❌ Failed to update plan"

        except Exception as e:
            return f"❌ Error updating plan step: {e}"

    def get_active_plans(self, max_results: int = None, _context_budget: dict = None) -> str:
        """
        Get all active (non-completed) plans with their progress.
        **Context-aware**: Automatically adjusts result count based on available tokens.

        Args:
            max_results: Maximum number of plans to return (automatically limited based on context)

        Returns:
            Formatted list of active plans with status

        Examples:
            get_active_plans()  # Get active plans (automatically sized)
            get_active_plans(5)  # Override with specific limit
        """
        # Use auto-injected limit
        if max_results is None:
            max_results = 10
        if not hasattr(self, '_agent') or not hasattr(self._agent, 'memory_manager'):
            return "❌ Memory manager not available"

        try:
            plans = self._agent.memory_manager.get_active_plans(max_results)

            if not plans:
                return "📋 No active plans found"

            result = []
            result.append("=" * 60)
            result.append("📋 ACTIVE PLANS")
            result.append("=" * 60)
            result.append("")

            for i, plan in enumerate(plans, 1):
                completed = plan['completed_steps']
                total = plan['total_steps']
                progress = (completed / total * 100) if total > 0 else 0

                result.append(f"{i}. {plan['description']}")
                result.append(f"   Plan ID: {plan['plan_id']}")
                result.append(f"   Status: {plan['status']}")
                result.append(f"   Progress: {completed}/{total} steps ({progress:.0f}%)")
                result.append(f"   Created: {plan['created_at']}")
                result.append("")

                # Show incomplete steps
                incomplete_steps = [s for s in plan['steps'] if s['status'] != 'completed']
                if incomplete_steps:
                    result.append("   Pending steps:")
                    for step in incomplete_steps[:3]:  # Show first 3 pending
                        result.append(f"   - [{step['status']}] {step['description']}")
                    if len(incomplete_steps) > 3:
                        result.append(f"   ... and {len(incomplete_steps) - 3} more")
                result.append("")

            result.append("=" * 60)
            return "\n".join(result)

        except Exception as e:
            return f"❌ Error getting active plans: {e}"

    def get_plan_details(self, plan_id: str) -> str:
        """
        Get detailed information about a specific plan

        Args:
            plan_id: The plan ID

        Returns:
            Detailed plan information with all steps

        Examples:
            get_plan_details("plan-abc-123")
        """
        if not hasattr(self, '_agent') or not hasattr(self._agent, 'memory_manager'):
            return "❌ Memory manager not available"

        try:
            plan = self._agent.memory_manager.get_plan(plan_id)

            if not plan:
                return f"❌ Plan not found: {plan_id}"

            result = []
            result.append("=" * 60)
            result.append("📋 PLAN DETAILS")
            result.append("=" * 60)
            result.append("")
            result.append(f"Description: {plan['description']}")
            result.append(f"Plan ID: {plan['plan_id']}")
            result.append(f"Status: {plan['status']}")
            result.append(f"Progress: {plan['completed_steps']}/{plan['total_steps']} steps")
            result.append(f"Created: {plan['created_at']}")
            result.append(f"Updated: {plan['updated_at']}")
            result.append("")
            result.append("Steps:")
            result.append("")

            for step in sorted(plan['steps'], key=lambda x: x.get('order', 0)):
                status_icon = "✅" if step['status'] == 'completed' else "🔄" if step['status'] == 'in_progress' else "⏸️"
                result.append(f"{step['order']}. {status_icon} [{step['status']}] {step['description']}")

            result.append("")
            result.append("=" * 60)
            return "\n".join(result)

        except Exception as e:
            return f"❌ Error getting plan details: {e}"


# Plugin factory
def create_plugin():
    """Factory function to create the plugin instance"""
    return ElasticSearchToolsPlugin()


# For testing
if __name__ == "__main__":
    plugin = ElasticSearchToolsPlugin()
    print(f"ElasticSearch Tools Plugin: {plugin.name} v{plugin.version}")
    print(f"Description: {plugin.description}")