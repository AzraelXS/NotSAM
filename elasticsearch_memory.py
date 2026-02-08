#!/usr/bin/env python3
"""
Enhanced ElasticSearch Memory System for SAM Agent
Comprehensive conversation logging with semantic search and context restoration
"""

import logging
import time
import json
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional
import requests
from requests.auth import HTTPBasicAuth
import urllib3

# Disable SSL warnings for self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("SAM.ElasticMemory")

# ⭐ EXTENDED VALID MEMORY TYPES
VALID_MEMORY_TYPES = [
    "conversation",  # Conversation messages (user/assistant/system)
    "tool_execution",  # Tool call records
    "tool_result_archive",  # Archived tool results removed from active context
    "core",  # Critical facts that define identity - ALWAYS loaded
    "personal",  # Personal facts about user/creator
    "experience",  # Events and interactions
    "observation",  # Things learned or noticed
    "reflection",  # Thoughts and insights
    "system_event",  # System 2/3 interventions
    "migrated",  # Imported from notes.txt
    "plan",  # Plans with steps and completion tracking
    "test"  # Test/debug entries
]

# ⭐ HIGH-VALUE MEMORY TYPES (prioritized in search)
HIGH_VALUE_MEMORY_TYPES = [
    "core",          # Core identity facts - HIGHEST priority, always loaded
    "personal",      # Personal facts - high priority
    "experience",    # Life events and interactions
    "observation",   # Learned patterns and insights
    "reflection",    # Thoughtful analysis
    "system_event",  # Important system actions
    "migrated"       # Imported knowledge
]

# ⭐ LOW-VALUE MEMORY TYPES (deprioritized in search unless explicitly requested)
LOW_VALUE_MEMORY_TYPES = [
    "conversation",      # Too noisy - contains tool requests
    "tool_execution",    # Technical execution logs
    "tool_result_archive",  # Archived results
    "plan"               # Plan metadata - not useful for knowledge retrieval
]

# ⭐ NOISE PATTERNS (content patterns to filter out)
NOISE_PATTERNS = [
    "search_memories",
    "search memories",
    "find memories",
    "retrieve memories",
    "look up",
    "store_memory",
    "get_recent_memories",
    "HISTORICAL TOOL EXECUTION",
    "Tool:",
    "Arguments:"
]


class ElasticSearchMemoryManager:
    """Manage SAM's memories in ElasticSearch with comprehensive conversation logging"""

    def __init__(self, config):
        self.config = config
        self.es_config = config.elasticsearch if hasattr(config, 'elasticsearch') else None
        self.embeddings_config = config.embeddings if hasattr(config, 'embeddings') else None

        if not self.es_config or not self.es_config.enabled:
            raise ValueError("ElasticSearch not enabled in configuration")

        self.host = self.es_config.host
        self.auth = HTTPBasicAuth(self.es_config.username, self.es_config.password)
        self.verify_ssl = self.es_config.verify_ssl
        self.datastream_name = self.es_config.datastream_name

        # Initialize datastream
        self._initialize_datastream()

        logger.info(f"ElasticSearch Memory Manager initialized: {self.host}")

    def register_tools(self, agent):
        """Register memory tools with SAM agent"""
        from sam_agent import ToolCategory

        agent.register_local_tool(
            self.store_memory,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

        agent.register_local_tool(
            self.search_memories,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

        agent.register_local_tool(
            self.get_recent_memories,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )
        
        agent.register_local_tool(
            self.retrieve_archived_tool_results,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )
    
    def format_tool_call_for_memory(self, tool_name: str, arguments: dict, result: str = None) -> str:
        """Format a tool call for safe storage in memory (prevents re-execution)"""
        formatted = f"HISTORICAL TOOL EXECUTION (archived):\n"
        formatted += f"Tool: {tool_name}\n"
        formatted += f"Arguments: {json.dumps(arguments, indent=2)}\n"
        if result:
            formatted += f"Result: {result[:500]}..." if len(result) > 500 else f"Result: {result}\n"
        return formatted

    def _initialize_datastream(self):
        """Create datastream and index template if they don't exist"""
        template_name = f"{self.datastream_name}-template"
        template = {
            "index_patterns": [f"{self.datastream_name}*"],
            "data_stream": {},
            "template": {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "memory_type": {"type": "keyword"},
                        "role": {"type": "keyword"},  # For conversation messages
                        "tool_name": {"type": "keyword"},  # For tool executions
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.embeddings_config.dimension if self.embeddings_config else 768,
                            "index": True,
                            "similarity": "cosine"
                        },
                        # Tool execution fields
                        "arguments": {"type": "object", "enabled": False},  # Store as-is, no dynamic mapping
                        "result": {"type": "text"},
                        "success": {"type": "boolean"},
                        # Tool result archive fields
                        "message_index": {"type": "integer"},
                        "tool_args": {"type": "object", "enabled": False},  # Store as-is, no dynamic mapping
                        "tool_result": {"type": "text"},
                        # Plan-specific fields
                        "plan_id": {"type": "keyword"},
                        "description": {"type": "text"},
                        "status": {"type": "keyword"},
                        "steps": {"type": "object", "enabled": False},  # Store as-is, no dynamic mapping
                        "total_steps": {"type": "integer"},
                        "completed_steps": {"type": "integer"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"},
                        "metadata": {
                            "properties": {
                                "session_id": {"type": "keyword"},
                                "tool_used": {"type": "keyword"},
                                "category": {"type": "keyword"},
                                "importance": {"type": "float"},
                                "iteration": {"type": "integer"},
                                "archived_timestamp": {"type": "float"}
                            }
                        }
                    }
                }
            }
        }

        try:
            # Create template
            response = requests.put(
                f"{self.host}/_index_template/{template_name}",
                json=template,
                auth=self.auth,
                verify=self.verify_ssl
            )

            if response.status_code in [200, 201]:
                logger.info(f"Index template created: {template_name}")

            # Create datastream
            response = requests.put(
                f"{self.host}/_data_stream/{self.datastream_name}",
                auth=self.auth,
                verify=self.verify_ssl
            )

            if response.status_code in [200, 201]:
                logger.info(f"Datastream created: {self.datastream_name}")

        except Exception as e:
            logger.error(f"Error initializing datastream: {e}")

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector for text using LMStudio embeddings API"""
        if not self.embeddings_config or not self.embeddings_config.enabled:
            logger.warning("Embeddings not enabled")
            return None

        try:
            response = requests.post(
                f"{self.embeddings_config.base_url}/embeddings",
                json={
                    "input": text,
                    "model": self.embeddings_config.model
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'data' in result and len(result['data']) > 0:
                    return result['data'][0]['embedding']

            logger.error(f"Failed to get embedding: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    # ===== CONVERSATION LOGGING =====

    def store_conversation_message(self, role: str, content: str,
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a conversation message in ElasticSearch

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata (tool_name, iteration, etc.)

        Returns:
            Success boolean
        """
        try:
            # Get embedding for semantic search
            embedding = self._get_embedding(content)

            # Prepare document
            document = {
                "@timestamp": datetime.now(UTC).isoformat(),
                "memory_type": "conversation",
                "role": role,
                "content": content,
                "metadata": metadata or {}
            }

            if embedding:
                document["embedding"] = embedding

            # Store in datastream
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_doc",
                json=document,
                auth=self.auth,
                verify=self.verify_ssl
            )

            if response.status_code in [200, 201]:
                logger.debug(f"Stored conversation message: {role}")
                return True
            else:
                logger.error(f"Failed to store message: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error storing conversation message: {e}")
            return False

    def store_tool_execution(self, tool_name: str, arguments: Dict[str, Any],
                             result: str, success: bool = True,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a tool execution record

        Args:
            tool_name: Name of the tool executed
            arguments: Tool arguments
            result: Tool execution result
            success: Whether execution was successful
            metadata: Optional metadata

        Returns:
            Success boolean
        """
        try:
            # Create searchable content
            content = f"Tool: {tool_name}\nArguments: {json.dumps(arguments)}\nResult: {result[:500]}"

            # Get embedding
            embedding = self._get_embedding(content)

            # Prepare document
            document = {
                "@timestamp": datetime.now(UTC).isoformat(),
                "memory_type": "tool_execution",
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "success": success,
                "content": content,
                "metadata": metadata or {}
            }

            if embedding:
                document["embedding"] = embedding

            # Store in datastream
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_doc",
                json=document,
                auth=self.auth,
                verify=self.verify_ssl
            )

            if response.status_code in [200, 201]:
                logger.debug(f"Stored tool execution: {tool_name}")
                return True
            else:
                logger.error(f"Failed to store tool execution: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error storing tool execution: {e}")
            return False

    # ===== CONVERSATION RETRIEVAL =====

    def get_recent_conversation(self, max_messages: int = None,
                                max_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve recent conversation messages for context restoration

        Args:
            max_messages: Maximum number of messages to retrieve
            max_tokens: Maximum tokens to retrieve (approximate)

        Returns:
            List of conversation messages in chronological order
        """
        try:
            # Default to reasonable limits if not specified
            if max_messages is None and max_tokens is None:
                max_messages = 50  # Default to last 50 messages

            # If max_tokens specified, retrieve more messages than needed
            # and then trim by tokens
            query_size = max_messages if max_messages else 200

            # Build query for conversation messages only
            search_query = {
                "query": {
                    "term": {"memory_type": "conversation"}
                },
                "size": query_size,
                "sort": [
                    {"@timestamp": {"order": "desc"}}  # Most recent first
                ],
                "_source": ["role", "content", "@timestamp", "metadata"]
            }

            response = requests.post(
                f"{self.host}/{self.datastream_name}/_search",
                json=search_query,
                auth=self.auth,
                verify=self.verify_ssl
            )

            if response.status_code == 200:
                results = response.json()
                hits = results.get('hits', {}).get('hits', [])

                # Convert to message format and reverse (chronological order)
                messages = []
                for hit in reversed(hits):  # Reverse to get chronological order
                    source = hit['_source']
                    role = source.get('role', 'user')  # Default to 'user' instead of 'unknown'

                    # Validate role - only allow user, assistant, system
                    if role not in ['user', 'assistant', 'system']:
                        logger.warning(f"Invalid role '{role}' in restored message, defaulting to 'user'")
                        role = 'user'

                    messages.append({
                        'role': role,
                        'content': source.get('content', ''),
                        'timestamp': source.get('@timestamp', ''),
                        'metadata': source.get('metadata', {})
                    })

                # If max_tokens specified, trim by token count
                if max_tokens:
                    messages = self._trim_messages_by_tokens(messages, max_tokens)

                logger.info(f"Retrieved {len(messages)} conversation messages")
                return messages
            else:
                logger.error(f"Failed to retrieve conversation: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error retrieving conversation: {e}")
            return []

    def _trim_messages_by_tokens(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """Trim messages to fit within token limit (keep most recent)"""
        trimmed = []
        total_tokens = 0

        # Process from most recent backwards
        for msg in reversed(messages):
            content = msg.get('content', '')
            # Rough token estimate: 4 chars per token
            msg_tokens = len(content) // 4

            if total_tokens + msg_tokens <= max_tokens:
                trimmed.insert(0, msg)  # Insert at beginning to maintain order
                total_tokens += msg_tokens
            else:
                break

        logger.info(f"Trimmed to {len(trimmed)} messages (~{total_tokens} tokens)")
        return trimmed

    # ===== LEGACY MEMORY METHODS (kept for compatibility) =====

    def store_memory(self, content: str, memory_type: str = "experience",
                     metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> str:
        """
        Store a memory in ElasticSearch with embedding
        
        MEMORY TYPE GUIDANCE:
        - **core**: Critical identity-defining facts that should ALWAYS be loaded into context.
                   Use for: fundamental facts about people (names, birthdays, relationships),
                   SAM's core purpose/identity, essential knowledge that defines who you are.
                   These get HIGHEST priority and are loaded at startup.
        
        - **personal**: Important personal facts that should be readily available.
                       Use for: preferences, characteristics, important details about people.
                       Loaded as part of core memories at startup.
        
        - **experience**: Significant events, interactions, and experiences.
                         Use for: things that happened, conversations, accomplishments.
                         Loaded as part of core memories at startup.
        
        - **observation**: Patterns and insights you've learned.
        - **reflection**: Deep thoughts and analysis.
        - **system_event**: Important system/AI oversight events.
        
        EXAMPLES:
        - "User's name is John" → memory_type="core" (fundamental fact)
        - "User prefers dark mode" → memory_type="personal" (preference)
        - "Had a great conversation about AI ethics" → memory_type="experience" (event)
        - "Users seem to prefer concise responses" → memory_type="observation" (pattern)
        
        Note: SAM was created by Azrael, an autistic adult with ADHD, time blindness, and dyscalculia,
        who was nevertheless an autodidactic pattern recognition savant.

        Args:
            content: The memory content to store
            memory_type: Type of memory (default: "experience")
            metadata: Optional metadata dictionary
            tags: Optional list of tags for categorization

        Returns:
            Success message or error
        """
        if memory_type not in VALID_MEMORY_TYPES:
            return f"❌ Invalid memory type '{memory_type}'. Valid types: {', '.join(VALID_MEMORY_TYPES)}"

        try:
            embedding = self._get_embedding(content)

            # Merge tags into metadata if provided
            if metadata is None:
                metadata = {}
            if tags:
                metadata['tags'] = tags

            document = {
                "@timestamp": datetime.utcnow().isoformat(),
                "memory_type": memory_type,
                "content": content,
                "metadata": metadata
            }

            if embedding:
                document["embedding"] = embedding

            response = requests.post(
                f"{self.host}/{self.datastream_name}/_doc",
                json=document,
                auth=self.auth,
                verify=self.verify_ssl
            )

            if response.status_code in [200, 201]:
                logger.info(f"Memory stored: {memory_type}")
                return f"✅ Memory stored successfully (type: {memory_type})"
            else:
                logger.error(f"Failed to store memory: {response.status_code}")
                return f"❌ Failed to store memory: HTTP {response.status_code}"

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return f"❌ Error storing memory: {str(e)}"

    def search_memories(self, query: str, max_results: int = 5,
                        memory_type: Optional[str] = None,
                        include_low_value: bool = False,
                        _context_budget: dict = None) -> str:
        """Search memories semantically (default: 5 results). Vector similarity with quality filtering.
        **Context-aware**: Automatically adjusts result count based on available tokens.
        
        IMPORTANT: By default, searches across ALL high-value memory types (personal, 
        experience, observation, reflection, etc.) and automatically filters out noisy
        conversation logs and tool executions.
        
        Only specify memory_type if the user explicitly requests a specific category
        (e.g., "search my personal memories" or "find experiences about...").
        
        For general searches like "What is X's birthday?" or "Do you remember...?",
        leave memory_type as None to search across all relevant types.
        
        Args:
            query: Search query (what to look for)
            max_results: Max results to return (default: 5, automatically limited based on context)
            memory_type: ONLY specify if user explicitly requests a specific category.
                        Leave as None for broad searches. Valid types: personal, 
                        experience, observation, reflection, system_event, migrated, etc.
            include_low_value: Include conversation/tool_execution types (default: False)
        
        Examples:
            - "What is the user's preference?" → memory_type=None (searches all)
            - "Search my personal facts" → memory_type="personal"
            - "Find experiences about..." → memory_type="experience"
        """
        # max_results already has a default of 5, but can be overridden
        if memory_type and memory_type not in VALID_MEMORY_TYPES:
            return f"❌ Invalid memory type '{memory_type}'. Valid types: {', '.join(VALID_MEMORY_TYPES)}"

        memories = self.semantic_search(
            query, 
            max_results, 
            memory_type,
            include_low_value=include_low_value
        )

        if not memories:
            search_scope = f" in '{memory_type}' memories" if memory_type else " across all memories"
            return f"🔍 No memories found matching: '{query}'{search_scope}"

        result = f"🔍 Found {len(memories)} memories matching '{query}':\n\n"
        for i, mem in enumerate(memories, 1):
            result += f"{i}. [{mem['timestamp']}] ({mem['memory_type']})\n"
            # Show full content without truncation
            result += f"   {mem['content']}\n"
            result += f"   Score: {mem['score']:.3f}\n\n"

        return result

    def semantic_search(self, query: str, max_results: int = None,
                        memory_type: Optional[str] = None,
                        include_low_value: bool = False) -> List[Dict[str, Any]]:
        """Search memories semantically using vector similarity with quality filtering
        
        Args:
            query: Search query
            max_results: Max results to return
            memory_type: Specific memory type filter (overrides include_low_value)
            include_low_value: If False, excludes conversation/tool_execution types
        """
        if max_results is None:
            max_results = self.es_config.max_results

        query_embedding = self._get_embedding(query)
        if not query_embedding:
            logger.warning("Could not generate query embedding, falling back to text search")
            return self.text_search(query, max_results, memory_type)

        # Fetch 3x more results to account for filtering
        fetch_size = max_results * 3 if not include_low_value else max_results

        search_body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": fetch_size,
                "num_candidates": fetch_size * 2
            },
            "_source": ["content", "memory_type", "role", "tool_name", "@timestamp", "metadata"]
        }

        # Apply filters
        filter_conditions = []
        
        if memory_type:
            # Explicit memory type request - honor it
            filter_conditions.append({"term": {"memory_type": memory_type}})
        elif not include_low_value:
            # Default: exclude low-value memory types
            filter_conditions.append({
                "bool": {
                    "must_not": [
                        {"terms": {"memory_type": LOW_VALUE_MEMORY_TYPES}}
                    ]
                }
            })
        
        if filter_conditions:
            if len(filter_conditions) == 1:
                search_body["knn"]["filter"] = filter_conditions[0]
            else:
                search_body["knn"]["filter"] = {
                    "bool": {"must": filter_conditions}
                }

        try:
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_search",
                json=search_body,
                auth=self.auth,
                verify=self.verify_ssl
            )

            if response.status_code == 200:
                results = response.json()
                hits = results.get('hits', {}).get('hits', [])

                memories = []
                for hit in hits:
                    source = hit['_source']
                    content = source.get('content', '')
                    mem_type = source.get('memory_type', '')
                    score = hit.get('_score', 0)
                    
                    # Quality filtering: check for noise patterns
                    if not include_low_value and self._is_noisy_content(content):
                        logger.debug(f"Filtered noisy content: {content[:50]}...")
                        continue
                    
                    # Score boosting for high-value memories
                    if mem_type in HIGH_VALUE_MEMORY_TYPES:
                        score *= 1.5  # 50% boost for high-value types
                    
                    memories.append({
                        'content': content,
                        'memory_type': mem_type,
                        'role': source.get('role', ''),
                        'tool_name': source.get('tool_name', ''),
                        'timestamp': source.get('@timestamp', ''),
                        'metadata': source.get('metadata', {}),
                        'score': score
                    })

                # Re-sort by adjusted scores and limit to max_results
                memories.sort(key=lambda x: x['score'], reverse=True)
                memories = memories[:max_results]

                logger.info(f"Semantic search returned {len(memories)} quality results (filtered from {len(hits)} raw hits)")
                return memories
            else:
                error_text = response.text
                logger.error(f"Search failed: {response.status_code} - {error_text}")
                return []

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _is_noisy_content(self, content: str) -> bool:
        """Check if content matches noise patterns that should be filtered"""
        content_lower = content.lower()
        for pattern in NOISE_PATTERNS:
            if pattern.lower() in content_lower:
                return True
        return False

    def get_recent_memories(self, max_results: int = 10,
                            memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get most recent memories chronologically"""
        if memory_type and memory_type not in VALID_MEMORY_TYPES:
            return f"❌ Invalid memory type '{memory_type}'. Valid types: {', '.join(VALID_MEMORY_TYPES)}"

        if max_results is None:
            max_results = self.es_config.max_results

        search_query = {
            "query": {
                "match_all": {}
            },
            "size": max_results,
            "sort": [
                {"@timestamp": {"order": "desc"}}
            ],
            "_source": ["content", "memory_type", "role", "tool_name", "@timestamp", "metadata"]
        }

        if memory_type:
            search_query["query"] = {
                "term": {"memory_type": memory_type}
            }

        try:
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_search",
                json=search_query,
                auth=self.auth,
                verify=self.verify_ssl
            )

            if response.status_code == 200:
                results = response.json()
                hits = results.get('hits', {}).get('hits', [])

                memories = []
                for hit in hits:
                    source = hit['_source']
                    memories.append({
                        'content': source.get('content', ''),
                        'memory_type': source.get('memory_type', ''),
                        'role': source.get('role', ''),
                        'tool_name': source.get('tool_name', ''),
                        'timestamp': source.get('@timestamp', ''),
                        'metadata': source.get('metadata', {})
                    })

                logger.info(f"Retrieved {len(memories)} recent memories")
                return memories
            else:
                logger.error(f"Search failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            return []

    def load_core_memories(self, max_core: int = 20, max_personal: int = 15, max_experience: int = 10,
                           max_tokens: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load core memories that define SAM's persistent identity and sense of self
        
        Returns the most important identity-defining facts in priority order:
        1. Explicit 'core' memories (highest priority - always loaded)
        2. 'personal' memories (important facts)
        3. 'experience' memories (key events)
        
        Args:
            max_core: Max explicit 'core' memories to load (HIGHEST priority)
            max_personal: Max personal memories to load (facts about users, SAM, etc.)
            max_experience: Max experience memories to load (important events, interactions)
            max_tokens: Optional token budget to limit total memory size
        
        Returns:
            Dict with 'core', 'personal', and 'experience' keys, each containing a list of memories
        """
        try:
            core_memories = {
                'core': [],
                'personal': [],
                'experience': []
            }
            
            # Load CORE memories first (HIGHEST PRIORITY - explicit identity facts)
            core_query = {
                "query": {
                    "term": {"memory_type": "core"}
                },
                "size": max_core,
                "sort": [
                    {"@timestamp": {"order": "desc"}}  # Most recent first
                ],
                "_source": ["content", "memory_type", "@timestamp", "metadata", "tags"]
            }
            
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_search",
                json=core_query,
                auth=self.auth,
                verify=self.verify_ssl
            )
            
            if response.status_code == 200:
                results = response.json()
                hits = results.get('hits', {}).get('hits', [])
                
                for hit in hits:
                    source = hit['_source']
                    content = source.get('content', '')
                    
                    # Filter out noisy content
                    if not self._is_noisy_content(content):
                        core_memories['core'].append({
                            'content': content,
                            'memory_type': 'core',
                            'timestamp': source.get('@timestamp', ''),
                            'metadata': source.get('metadata', {}),
                            'tags': source.get('tags', [])
                        })
                
                logger.info(f"Loaded {len(core_memories['core'])} CORE identity memories")
            
            # Load personal memories (high priority - facts about users, SAM's identity)
            personal_query = {
                "query": {
                    "term": {"memory_type": "personal"}
                },
                "size": max_personal,
                "sort": [
                    {"@timestamp": {"order": "desc"}}  # Most recent first
                ],
                "_source": ["content", "memory_type", "@timestamp", "metadata", "tags"]
            }
            
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_search",
                json=personal_query,
                auth=self.auth,
                verify=self.verify_ssl
            )
            
            if response.status_code == 200:
                results = response.json()
                hits = results.get('hits', {}).get('hits', [])
                
                for hit in hits:
                    source = hit['_source']
                    content = source.get('content', '')
                    
                    # Filter out noisy content
                    if not self._is_noisy_content(content):
                        core_memories['personal'].append({
                            'content': content,
                            'memory_type': 'personal',
                            'timestamp': source.get('@timestamp', ''),
                            'metadata': source.get('metadata', {}),
                            'tags': source.get('tags', [])
                        })
                
                logger.info(f"Loaded {len(core_memories['personal'])} personal core memories")
            
            # Load experience memories (important interactions and events)
            experience_query = {
                "query": {
                    "term": {"memory_type": "experience"}
                },
                "size": max_experience,
                "sort": [
                    {"@timestamp": {"order": "desc"}}
                ],
                "_source": ["content", "memory_type", "@timestamp", "metadata", "tags"]
            }
            
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_search",
                json=experience_query,
                auth=self.auth,
                verify=self.verify_ssl
            )
            
            if response.status_code == 200:
                results = response.json()
                hits = results.get('hits', {}).get('hits', [])
                
                for hit in hits:
                    source = hit['_source']
                    content = source.get('content', '')
                    
                    # Filter out noisy content
                    if not self._is_noisy_content(content):
                        core_memories['experience'].append({
                            'content': content,
                            'memory_type': 'experience',
                            'timestamp': source.get('@timestamp', ''),
                            'metadata': source.get('metadata', {}),
                            'tags': source.get('tags', [])
                        })
                
                logger.info(f"Loaded {len(core_memories['experience'])} experience core memories")
            
            # If token budget specified, trim memories to fit
            if max_tokens:
                core_memories = self._trim_core_memories_by_tokens(core_memories, max_tokens)
            
            total_loaded = len(core_memories['core']) + len(core_memories['personal']) + len(core_memories['experience'])
            logger.info(f"✅ Loaded {total_loaded} total core memories for persistent identity")
            
            return core_memories
            
        except Exception as e:
            logger.error(f"Error loading core memories: {e}")
            return {'core': [], 'personal': [], 'experience': []}
    
    def _trim_core_memories_by_tokens(self, core_memories: Dict[str, List], max_tokens: int) -> Dict[str, List]:
        """Trim core memories to fit within token budget, prioritizing: core > personal > experience"""
        total_tokens = 0
        trimmed = {'core': [], 'personal': [], 'experience': []}
        
        # First, add CORE memories (HIGHEST priority - always included if possible)
        for mem in core_memories.get('core', []):
            content = mem.get('content', '')
            mem_tokens = len(content) // 4  # Rough estimate
            
            if total_tokens + mem_tokens <= max_tokens:
                trimmed['core'].append(mem)
                total_tokens += mem_tokens
            else:
                break
        
        # Second, add personal memories (high priority)
        for mem in core_memories.get('personal', []):
            content = mem.get('content', '')
            mem_tokens = len(content) // 4  # Rough estimate
            
            if total_tokens + mem_tokens <= max_tokens:
                trimmed['personal'].append(mem)
                total_tokens += mem_tokens
            else:
                break
        
        # Finally, add experience memories with remaining budget
        for mem in core_memories.get('experience', []):
            content = mem.get('content', '')
            mem_tokens = len(content) // 4
            
            if total_tokens + mem_tokens <= max_tokens:
                trimmed['experience'].append(mem)
                total_tokens += mem_tokens
            else:
                break
        
        logger.info(f"Trimmed core memories to ~{total_tokens} tokens")
        return trimmed

    def text_search(self, query: str, max_results: int = None,
                    memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback text-based search"""
        if max_results is None:
            max_results = self.es_config.max_results

        search_query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"content": query}}
                    ]
                }
            },
            "size": max_results,
            "_source": ["content", "memory_type", "role", "tool_name", "@timestamp", "metadata"]
        }

        if memory_type:
            search_query["query"]["bool"]["filter"] = [
                {"term": {"memory_type": memory_type}}
            ]

        try:
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_search",
                json=search_query,
                auth=self.auth,
                verify=self.verify_ssl
            )

            if response.status_code == 200:
                results = response.json()
                hits = results.get('hits', {}).get('hits', [])

                memories = []
                for hit in hits:
                    source = hit['_source']
                    memories.append({
                        'content': source.get('content', ''),
                        'memory_type': source.get('memory_type', ''),
                        'role': source.get('role', ''),
                        'tool_name': source.get('tool_name', ''),
                        'timestamp': source.get('@timestamp', ''),
                        'metadata': source.get('metadata', {}),
                        'score': hit.get('_score', 0)
                    })

                logger.info(f"Text search returned {len(memories)} results")
                return memories
            else:
                logger.error(f"Search failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []

    # ===== PLAN MANAGEMENT =====

    def store_plan(self, description: str, steps: List[Dict[str, Any]], 
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Store a plan with steps in ElasticSearch
        
        Args:
            description: Overall plan description/goal
            steps: List of step dicts with 'description', 'status', 'order' keys
            metadata: Optional metadata (session_id, conversation_ref, etc.)
            
        Returns:
            Plan ID if successful, None otherwise
            
        Example:
            plan_id = store_plan(
                "Implement authentication system",
                [
                    {"description": "Design database schema", "status": "completed", "order": 1},
                    {"description": "Create user model", "status": "in_progress", "order": 2},
                    {"description": "Add login endpoint", "status": "pending", "order": 3}
                ],
                metadata={"session_id": "abc123", "priority": "high"}
            )
        """
        import uuid
        
        try:
            plan_id = str(uuid.uuid4())
            timestamp = datetime.now(UTC).isoformat()
            
            # Calculate overall plan status
            statuses = [step.get('status', 'pending') for step in steps]
            if all(s == 'completed' for s in statuses):
                plan_status = 'completed'
            elif any(s in ['in_progress', 'in-progress'] for s in statuses):
                plan_status = 'in_progress'
            else:
                plan_status = 'pending'
            
            # Get embedding for semantic search of plans
            embedding = self._get_embedding(description)
            
            # Prepare document
            document = {
                "@timestamp": timestamp,
                "memory_type": "plan",
                "plan_id": plan_id,
                "description": description,
                "status": plan_status,
                "steps": steps,
                "total_steps": len(steps),
                "completed_steps": sum(1 for s in statuses if s == 'completed'),
                "created_at": timestamp,
                "updated_at": timestamp,
                "metadata": metadata or {}
            }
            
            if embedding:
                document["embedding"] = embedding
            
            # Store in datastream with refresh=true to make immediately searchable
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_doc?refresh=true",
                json=document,
                auth=self.auth,
                verify=self.verify_ssl
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Plan stored with ID: {plan_id}")
                return plan_id
            else:
                logger.error(f"Failed to store plan: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing plan: {e}")
            return None
    
    def update_plan(self, plan_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing plan
        
        Args:
            plan_id: The plan ID to update
            updates: Dict with fields to update (steps, status, metadata, etc.)
            
        Returns:
            Success boolean
            
        Example:
            update_plan("plan-123", {
                "steps": updated_steps_list,
                "status": "in_progress"
            })
        """
        try:
            # First find the document - try with .keyword suffix first, then without
            search_queries = [
                # Try with .keyword mapping first (standard for text fields)
                {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"memory_type": "plan"}},
                                {"term": {"plan_id.keyword": plan_id}}
                            ]
                        }
                    },
                    "size": 1
                },
                # Fallback to match query without .keyword
                {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"memory_type": "plan"}},
                                {"match": {"plan_id": plan_id}}
                            ]
                        }
                    },
                    "size": 1
                }
            ]
            
            hits = []
            for search_query in search_queries:
                response = requests.post(
                    f"{self.host}/{self.datastream_name}/_search",
                    json=search_query,
                    auth=self.auth,
                    verify=self.verify_ssl
                )
                
                if response.status_code == 200:
                    results = response.json()
                    hits = results.get('hits', {}).get('hits', [])
                    if hits:
                        break  # Found it!
            
            if not hits:
                logger.error(f"Plan not found: {plan_id}")
                return False
            
            current_doc = hits[0]['_source']
            
            # Build update document
            update_doc = dict(current_doc)
            update_doc.update(updates)
            update_doc['updated_at'] = datetime.now(UTC).isoformat()
            update_doc['@timestamp'] = datetime.now(UTC).isoformat()
            
            # Recalculate status if steps were updated
            if 'steps' in updates:
                steps = updates['steps']
                statuses = [step.get('status', 'pending') for step in steps]
                update_doc['total_steps'] = len(steps)
                update_doc['completed_steps'] = sum(1 for s in statuses if s == 'completed')
                
                if all(s == 'completed' for s in statuses):
                    update_doc['status'] = 'completed'
                elif any(s in ['in_progress', 'in-progress'] for s in statuses):
                    update_doc['status'] = 'in_progress'
            
            # Datastreams are append-only - create new document with same plan_id
            # Queries will get the latest by @timestamp
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_doc?refresh=true",
                json=update_doc,
                auth=self.auth,
                verify=self.verify_ssl
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Plan updated (new version): {plan_id}")
                return True
            else:
                logger.error(f"Failed to update plan: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating plan: {e}")
            return False
    
    def delete_plan(self, plan_id: str) -> bool:
        """
        Delete all versions of a plan from the datastream
        
        Args:
            plan_id: The plan ID to delete
            
        Returns:
            Success boolean
        """
        try:
            # Use delete_by_query to remove all documents with this plan_id
            delete_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"memory_type": "plan"}},
                            {"match": {"plan_id": plan_id}}
                        ]
                    }
                }
            }
            
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_delete_by_query?refresh=true",
                json=delete_query,
                auth=self.auth,
                verify=self.verify_ssl
            )
            
            if response.status_code == 200:
                result = response.json()
                deleted_count = result.get('deleted', 0)
                logger.info(f"Deleted {deleted_count} document(s) for plan: {plan_id}")
                return deleted_count > 0
            else:
                logger.error(f"Failed to delete plan: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting plan: {e}")
            return False
    
    def get_active_plans(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get all active (non-completed) plans
        
        Args:
            max_results: Maximum number of plans to return
            
        Returns:
            List of plan dicts with metadata
        """
        try:
            # NOTE: Don't filter by status in the query, because plans can have multiple versions
            # and we need to deduplicate FIRST, then filter out completed ones
            # Otherwise we might return old "pending" versions when the latest is "completed"
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"memory_type": "plan"}}
                        ]
                    }
                },
                "sort": [
                    {"@timestamp": {"order": "desc"}}
                ],
                "size": max_results * 10  # Get more to account for multiple versions per plan
            }
            
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_search",
                json=search_query,
                auth=self.auth,
                verify=self.verify_ssl
            )
            
            if response.status_code == 200:
                results = response.json()
                hits = results.get('hits', {}).get('hits', [])
                
                # Deduplicate plans - keep only latest version of each plan_id
                seen_plans = {}
                for hit in hits:
                    source = hit['_source']
                    plan_id = source.get('plan_id', '')
                    
                    # Since results are sorted by @timestamp desc, first occurrence is latest
                    if plan_id not in seen_plans:
                        seen_plans[plan_id] = {
                            'plan_id': plan_id,
                            'description': source.get('description', ''),
                            'status': source.get('status', ''),
                            'steps': source.get('steps', []),
                            'total_steps': source.get('total_steps', 0),
                            'completed_steps': source.get('completed_steps', 0),
                            'created_at': source.get('created_at', ''),
                            'updated_at': source.get('updated_at', ''),
                            'metadata': source.get('metadata', {})
                        }
                
                # Filter out completed plans AFTER deduplication
                # This ensures we check the LATEST version of each plan
                active_plans = [p for p in seen_plans.values() if p['status'] != 'completed']
                
                # Limit to max_results
                plans = active_plans[:max_results]
                
                logger.info(f"Retrieved {len(plans)} active plans (deduplicated from {len(hits)} documents)")
                return plans
            else:
                logger.error(f"Failed to get active plans: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting active plans: {e}")
            return []
    
    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific plan by ID
        
        Args:
            plan_id: The plan ID
            
        Returns:
            Plan dict or None
        """
        try:
            # Try with .keyword mapping first, then without
            search_queries = [
                {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"memory_type": "plan"}},
                                {"term": {"plan_id.keyword": plan_id}}
                            ]
                        }
                    },
                    "sort": [{"@timestamp": {"order": "desc"}}],
                    "size": 1
                },
                {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"memory_type": "plan"}},
                                {"match": {"plan_id": plan_id}}
                            ]
                        }
                    },
                    "sort": [{"@timestamp": {"order": "desc"}}],
                    "size": 1
                }
            ]
            
            hits = []
            for search_query in search_queries:
                response = requests.post(
                    f"{self.host}/{self.datastream_name}/_search",
                    json=search_query,
                    auth=self.auth,
                    verify=self.verify_ssl
                )
                
                if response.status_code == 200:
                    results = response.json()
                    hits = results.get('hits', {}).get('hits', [])
                    if hits:
                        break
            
            if hits:
                source = hits[0]['_source']
                return {
                    'plan_id': source.get('plan_id', ''),
                    'description': source.get('description', ''),
                    'status': source.get('status', ''),
                    'steps': source.get('steps', []),
                    'total_steps': source.get('total_steps', 0),
                    'completed_steps': source.get('completed_steps', 0),
                    'created_at': source.get('created_at', ''),
                    'updated_at': source.get('updated_at', ''),
                    'metadata': source.get('metadata', {})
                }
            
            logger.warning(f"Plan not found: {plan_id}")
            return None
                
        except Exception as e:
            logger.error(f"Error getting plan: {e}")
            return None
    
    # ===== TOOL RESULT ARCHIVING =====
    
    def store_tool_result_archive(self, message_index: int, tool_name: str, 
                                   tool_args: Dict[str, Any], tool_result: str,
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Archive a tool result before removing it from conversation history
        
        This allows System 2 to aggressively clean up large tool results while
        preserving them for later retrieval if System 1 needs to reference them.
        
        Args:
            message_index: The conversation history index of the message
            tool_name: Name of the tool that was executed
            tool_args: Arguments passed to the tool
            tool_result: The raw result from the tool
            metadata: Optional metadata (user_message, timestamp, etc.)
            
        Returns:
            Success boolean
        """
        try:
            # Create searchable content
            args_str = json.dumps(tool_args, indent=2) if tool_args else "{}"
            content = f"Tool: {tool_name}\nArguments: {args_str}\n\nResult:\n{tool_result}"
            
            # Get embedding for semantic search
            embedding = self._get_embedding(f"{tool_name} {args_str}")
            
            # Prepare document
            document = {
                "@timestamp": datetime.now(UTC).isoformat(),
                "memory_type": "tool_result_archive",
                "message_index": message_index,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": tool_result,
                "content": content,
                "metadata": metadata or {}
            }
            
            if embedding:
                document["embedding"] = embedding
            
            # Store in datastream
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_doc",
                json=document,
                auth=self.auth,
                verify=self.verify_ssl
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Archived tool result: {tool_name} (index {message_index})")
                return True
            else:
                error_msg = response.text if hasattr(response, 'text') else 'Unknown error'
                logger.error(f"Failed to archive tool result: {response.status_code} - {error_msg}")
                logger.debug(f"Document that failed: {document}")
                return False
                
        except Exception as e:
            logger.error(f"Error archiving tool result: {e}")
            return False
    
    def retrieve_archived_tool_results(self, tool_name: Optional[str] = None,
                                        since_timestamp: Optional[str] = None,
                                        max_results: int = None,
                                        _context_budget: dict = None) -> str:
        """Retrieve archived tool results that were removed from conversation history.
        **Context-aware**: Automatically adjusts result count based on available tokens.
        
        System 1 can use this to look up results from past tool executions
        even after System 2 has cleaned them from the active context.
        
        Args:
            tool_name: Optional filter by specific tool name
            since_timestamp: Optional ISO timestamp to retrieve results after
            max_results: Maximum number of results to return (automatically limited)
            
        Returns:
            Formatted string with archived tool results
        """
        # Use auto-injected limit
        if max_results is None:
            max_results = 10
        
        try:
            # Build query
            must_clauses = [
                {"term": {"memory_type": "tool_result_archive"}}
            ]
            
            if tool_name:
                must_clauses.append({"term": {"tool_name.keyword": tool_name}})
            
            search_query = {
                "query": {
                    "bool": {
                        "must": must_clauses
                    }
                },
                "size": max_results,
                "sort": [
                    {"@timestamp": {"order": "desc"}}
                ],
                "_source": ["tool_name", "tool_args", "tool_result", "@timestamp", "message_index", "metadata"]
            }
            
            if since_timestamp:
                search_query["query"]["bool"]["filter"] = [
                    {"range": {"@timestamp": {"gte": since_timestamp}}}
                ]
            
            response = requests.post(
                f"{self.host}/{self.datastream_name}/_search",
                json=search_query,
                auth=self.auth,
                verify=self.verify_ssl
            )
            
            if response.status_code == 200:
                results = response.json()
                hits = results.get('hits', {}).get('hits', [])
                
                if not hits:
                    filter_str = f" for tool '{tool_name}'" if tool_name else ""
                    return f"🔍 No archived tool results found{filter_str}"
                
                output = f"📦 Found {len(hits)} archived tool result(s):\n\n"
                
                for i, hit in enumerate(hits, 1):
                    source = hit['_source']
                    timestamp = source.get('@timestamp', 'Unknown')
                    tool = source.get('tool_name', 'Unknown')
                    msg_idx = source.get('message_index', 'N/A')
                    tool_args = source.get('tool_args', {})
                    tool_result = source.get('tool_result', '')
                    
                    # Format args nicely
                    args_str = json.dumps(tool_args, indent=2) if tool_args else "{}"
                    
                    # Truncate result if too long
                    result_preview = tool_result[:500] + "..." if len(tool_result) > 500 else tool_result
                    
                    output += f"{'='*60}\n"
                    output += f"🔧 Tool #{i}: {tool}\n"
                    output += f"📅 Timestamp: {timestamp}\n"
                    output += f"📍 Original message index: {msg_idx}\n"
                    output += f"📋 Arguments:\n{args_str}\n\n"
                    output += f"📊 Result:\n{result_preview}\n"
                    output += f"{'='*60}\n\n"
                
                logger.info(f"Retrieved {len(hits)} archived tool results")
                return output
            else:
                logger.error(f"Failed to retrieve archived tool results: {response.status_code}")
                return f"❌ Failed to retrieve archived tool results: HTTP {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error retrieving archived tool results: {e}")
            return f"❌ Error retrieving archived tool results: {str(e)}"