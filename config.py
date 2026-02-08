#!/usr/bin/env python3
"""
SAM Agent Configuration System
Comprehensive configuration management for Semi-Autonomous Model
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum
import platform

logger = logging.getLogger("SAM.Config")


# ===== CONFIGURATION ENUMS =====
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Theme(Enum):
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"


class ModelProvider(Enum):
    LMSTUDIO = "lmstudio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    CUSTOM = "custom"


# ===== CONFIGURATION DATACLASSES =====

@dataclass
class LMStudioConfig:
    """LM Studio API configuration"""
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    timeout: int = 300
    max_retries: int = 3
    verify_ssl: bool = False
    stream_chunk_size: int = 1024


@dataclass
class ModelConfig:
    """Model-specific configuration"""
    provider: ModelProvider = ModelProvider.LMSTUDIO
    model_name: str = "sam-1"
    context_limit: int = 128000
    temperature: float = 0.3
    max_tokens: int = -1
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)

    # Custom model configurations
    custom_base_url: str = ""
    custom_api_key: str = ""
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ToolConfig:
    """Tool execution configuration"""
    auto_approve: bool = True
    execution_timeout: int = 30
    max_iterations: int = 10
    enable_code_execution: bool = True
    enable_file_operations: bool = True
    enable_web_browsing: bool = True
    enable_system_commands: bool = False

    # Safety settings
    safe_mode: bool = True
    blocked_commands: List[str] = field(default_factory=lambda: [
        "rm -rf", "sudo rm", "mkfs", "dd if=", ":(){ :|:& };:",
        "chmod 777", "passwd", "su -", "sudo su"
    ])

    # Execution limits
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_output_length: int = 50000
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".py", ".js", ".json", ".yaml", ".yml",
        ".html", ".css", ".csv", ".log"
    ])


@dataclass
class MCPConfig:
    """Model Context Protocol configuration"""
    enabled: bool = False
    auto_discover: bool = True
    discovery_timeout: int = 30
    connection_timeout: int = 60
    max_concurrent_sessions: int = 5

    # Server configurations
    servers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Default server paths to try
    default_server_paths: List[str] = field(default_factory=lambda: [
        "./mcp_servers",
        "~/.sam/mcp_servers",
        "/usr/local/share/sam/mcp_servers"
    ])


@dataclass
class MemoryConfig:
    """Memory and context management configuration"""
    enabled: bool = True
    storage_type: str = "local"  # local, redis, sqlite, memory
    storage_path: str = "~/.sam/memory"

    # Vector storage settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: str = "faiss"  # faiss, chroma, pinecone
    collection_name: str = "sam_memories"

    # Context management
    max_context_tokens: int = 100000
    compression_threshold: float = 0.85
    summary_max_length: int = 500
    preserve_recent_messages: int = 10
    auto_compress: bool = True

    # Memory retrieval
    max_results: int = 5
    similarity_threshold: float = 0.7
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class SecurityConfig:
    """Security and safety configuration"""
    enable_sandbox: bool = True
    sandbox_type: str = "docker"  # docker, chroot, none

    # API security
    api_key: Optional[str] = None
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:*"])
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour

    # File system restrictions
    restricted_paths: List[str] = field(default_factory=lambda: [
        "/etc", "/bin", "/sbin", "/boot", "/sys", "/proc"
    ])

    # Network restrictions
    blocked_domains: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=lambda: [80, 443, 8000, 8080])


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    console_enabled: bool = True
    file_enabled: bool = True
    file_path: str = "~/.sam/logs/sam.log"

    # File rotation
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

    # Log formatting
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Special logging
    conversation_logging: bool = True
    tool_execution_logging: bool = True
    api_request_logging: bool = False


@dataclass
class APIConfig:
    """API server configuration"""
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8888

    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Authentication
    auth_enabled: bool = False
    auth_token: Optional[str] = None

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600

    # WebSocket settings
    websocket_enabled: bool = True
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10


@dataclass
class UIConfig:
    """User interface configuration"""
    theme: Theme = Theme.DARK
    font_size: str = "medium"  # small, medium, large
    auto_scroll: bool = True
    sound_notifications: bool = False
    desktop_notifications: bool = True

    # Window settings
    window_width: int = 1400
    window_height: int = 900
    window_min_width: int = 800
    window_min_height: int = 600

    # Behavior
    auto_save_conversations: bool = True
    conversation_history_limit: int = 1000
    typing_indicators: bool = True


@dataclass
class PluginConfig:
    """Plugin system configuration"""
    enabled: bool = True
    auto_load: bool = True
    plugin_directories: List[str] = field(default_factory=lambda: [
        "./plugins",
        "./sam_plugins",
        "~/.sam/plugins"
    ])

    # Plugin security
    verify_signatures: bool = False
    allowed_plugins: List[str] = field(default_factory=list)
    blocked_plugins: List[str] = field(default_factory=list)

    # Plugin settings
    core_plugins: List[str] = field(default_factory=lambda: [
        "core_tools",
        "file_operations",
        "text_processing"
    ])


@dataclass
class PerformanceConfig:
    """Performance and optimization configuration"""
    # Execution settings
    max_concurrent_tools: int = 3
    tool_execution_pool_size: int = 4

    # Memory optimization
    garbage_collection_interval: int = 300  # 5 minutes
    max_memory_usage: int = 2 * 1024 * 1024 * 1024  # 2GB

    # Caching
    response_cache_enabled: bool = True
    response_cache_size: int = 100
    response_cache_ttl: int = 3600  # 1 hour

    # Network optimization
    connection_pool_size: int = 10
    request_timeout: int = 300

@dataclass
class ElasticSearchConfig:
    """ElasticSearch configuration"""
    enabled: bool = True
    host: str = "https://192.168.1.187:9200"
    username: str = "elastic"
    password: str = "zvhWL50+_b6G55Cd+CV9"
    verify_ssl: bool = False
    datastream_name: str = "sam-memories"
    max_results: int = 10
    similarity_threshold: float = 0.75
    index_template_name: str = "sam-memories-template"


@dataclass
class SearXNGConfig:
    """SearXNG search engine configuration"""
    enabled: bool = True
    base_url: str = "http://192.168.1.227/searxng"
    language: str = "en"
    min_score: float = 1.0
    max_results: int = 10
    timeout: int = 15
    verify_ssl: bool = False


@dataclass
class EmbeddingsConfig:
    """Embeddings API configuration"""
    enabled: bool = False
    provider: str = "lmstudio"
    base_url: str = "http://localhost:1234/v1"
    model: str = "nomic-ai/nomic-embed-text-v1.5-GGUF"
    dimension: int = 768
    timeout: int = 30
    batch_size: int = 10


@dataclass
class EmailConfig:
    """Email notification configuration"""
    enabled: bool = False
    provider: str = "ses"  # ses, smtp, sendgrid
    
    # Amazon SES configuration
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    
    # SMTP configuration (alternative)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    
    # Email settings
    from_email: str = ""
    to_email: str = ""
    reply_to: str = ""
    
    # Notification preferences
    notify_on_completion: bool = True
    notify_on_error: bool = True
    notify_on_approval: bool = False
    include_logs: bool = False
    timeout: int = 30


@dataclass
class MultiAgentResearchConfig:
    """Multi-agent web research configuration"""
    enabled: bool = False
    test_mode: bool = True
    lead_agent_timeout: int = 300
    fetcher_agent_timeout: int = 30
    max_concurrent_fetchers: int = 5
    max_results_per_search: int = 10
    relevance_threshold: float = 0.7
    temp_index_prefix: str = "web_research_"


@dataclass
class BrowserConfig:
    """Browser automation configuration"""
    enabled: bool = True
    headless: bool = True
    timeout: int = 30
    max_sessions: int = 5
    window_width: int = 1920
    window_height: int = 1080
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    dismiss_popups: bool = True


# ===== MAIN CONFIGURATION CLASS =====

@dataclass
class SAMConfig:
    """Main SAM Agent configuration"""
    # Meta information
    version: str = "1.0.0"
    agent_name: str = "SAM"
    agent_description: str = "Secret Agent Man AI Agent"

    # Component configurations
    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # ADD THESE TWO LINES:
    elasticsearch: ElasticSearchConfig = field(default_factory=ElasticSearchConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    searxng: SearXNGConfig = field(default_factory=SearXNGConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    multi_agent_research: MultiAgentResearchConfig = field(default_factory=MultiAgentResearchConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)

    # Feature flags
    enable_streaming: bool = True
    enable_context_management: bool = True
    enable_memory: bool = True
    enable_tools: bool = True
    enable_mcp: bool = False
    enable_web_ui: bool = True

    # Paths
    config_dir: str = "~/.sam"
    data_dir: str = "~/.sam/data"
    cache_dir: str = "~/.sam/cache"
    log_dir: str = "~/.sam/logs"

    # Environment-specific settings
    development_mode: bool = False
    debug_mode: bool = False
    verbose_mode: bool = False

    def __post_init__(self):
        """Post-initialization setup"""
        # Expand user paths
        self.config_dir = str(Path(self.config_dir).expanduser())
        self.data_dir = str(Path(self.data_dir).expanduser())
        self.cache_dir = str(Path(self.cache_dir).expanduser())
        self.log_dir = str(Path(self.log_dir).expanduser())

        # Expand paths in sub-configurations
        if self.memory.storage_path.startswith("~"):
            self.memory.storage_path = str(Path(self.memory.storage_path).expanduser())

        if self.logging.file_path.startswith("~"):
            self.logging.file_path = str(Path(self.logging.file_path).expanduser())

        # Create directories if they don't exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.config_dir,
            self.data_dir,
            self.cache_dir,
            self.log_dir,
            str(Path(self.memory.storage_path).parent),
            str(Path(self.logging.file_path).parent)
        ]

        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to create directory {directory}: {e}")


# ===== CONFIGURATION MANAGER =====

class ConfigManager:
    """Configuration management utilities"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(Path("~/.sam/config.json").expanduser())
        self.config = SAMConfig()

    def load_config(self, config_path: Optional[str] = None) -> SAMConfig:
        """Load configuration from file"""
        if config_path:
            self.config_path = config_path

        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)

                # Merge with default config
                self.config = self._dict_to_config(config_dict)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.info("No config file found, using defaults")

        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {e}")
            logger.info("Using default configuration")

        # Apply environment variables
        self._apply_environment_variables()

        return self.config

    def save_config(self, config: Optional[SAMConfig] = None) -> bool:
        """Save configuration to file"""
        if config:
            self.config = config

        try:
            # Ensure config directory exists
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict and save
            config_dict = asdict(self.config)

            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)

            logger.info(f"Configuration saved to {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving config to {self.config_path}: {e}")
            return False

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SAMConfig:
        """Convert dictionary to SAMConfig object"""
        # This is a simplified version - you might want to use a more sophisticated
        # approach like marshmallow or pydantic for complex nested structures
        try:
            return SAMConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Error converting config dict: {e}")
            # Merge with defaults
            default_config = SAMConfig()
            merged_dict = asdict(default_config)

            def merge_dicts(default: dict, custom: dict) -> dict:
                for key, value in custom.items():
                    if key in default:
                        if isinstance(value, dict) and isinstance(default[key], dict):
                            default[key] = merge_dicts(default[key], value)
                        else:
                            default[key] = value
                return default

            merged_dict = merge_dicts(merged_dict, config_dict)
            return SAMConfig(**merged_dict)

    def _apply_environment_variables(self):
        """Apply environment variable overrides"""
        env_mappings = {
            # Model settings
            'SAM_MODEL_NAME': ('model', 'model_name'),
            'SAM_BASE_URL': ('lmstudio', 'base_url'),
            'SAM_API_KEY': ('lmstudio', 'api_key'),
            'SAM_TEMPERATURE': ('model', 'temperature'),
            'SAM_CONTEXT_LIMIT': ('model', 'context_limit'),

            # Tool settings
            'SAM_AUTO_APPROVE': ('tools', 'auto_approve'),
            'SAM_SAFE_MODE': ('tools', 'safe_mode'),
            'SAM_EXECUTION_TIMEOUT': ('tools', 'execution_timeout'),

            # API settings
            'SAM_API_HOST': ('api', 'host'),
            'SAM_API_PORT': ('api', 'port'),
            'SAM_API_ENABLED': ('api', 'enabled'),

            # Logging
            'SAM_LOG_LEVEL': ('logging', 'level'),
            'SAM_LOG_FILE': ('logging', 'file_path'),

            # Feature flags
            'SAM_ENABLE_TOOLS': ('enable_tools',),
            'SAM_ENABLE_MCP': ('enable_mcp',),
            'SAM_DEBUG_MODE': ('debug_mode',),
            'SAM_VERBOSE_MODE': ('verbose_mode',),
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Navigate to the config attribute
                    obj = self.config
                    for attr in config_path[:-1]:
                        obj = getattr(obj, attr)

                    # Get the current value to determine type
                    current_value = getattr(obj, config_path[-1])

                    # Convert value to appropriate type
                    if isinstance(current_value, bool):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    elif isinstance(current_value, LogLevel):
                        value = LogLevel(value.upper())

                    # Set the value
                    setattr(obj, config_path[-1], value)
                    logger.debug(f"Applied environment variable {env_var}={value}")

                except Exception as e:
                    logger.warning(f"Failed to apply environment variable {env_var}: {e}")

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Validate model configuration
        if self.config.model.context_limit <= 0:
            issues.append("Model context_limit must be positive")

        if not (0.0 <= self.config.model.temperature <= 2.0):
            issues.append("Model temperature must be between 0.0 and 2.0")

        # Validate tool configuration
        if self.config.tools.execution_timeout <= 0:
            issues.append("Tool execution_timeout must be positive")

        if self.config.tools.max_iterations <= 0:
            issues.append("Tool max_iterations must be positive")

        # Validate API configuration
        if self.config.api.enabled:
            if not (1 <= self.config.api.port <= 65535):
                issues.append("API port must be between 1 and 65535")

        # Validate memory configuration
        if self.config.memory.enabled:
            if self.config.memory.chunk_size <= 0:
                issues.append("Memory chunk_size must be positive")

        # Validate paths
        try:
            Path(self.config.config_dir).expanduser().mkdir(parents=True, exist_ok=True)
        except Exception:
            issues.append(f"Cannot create config directory: {self.config.config_dir}")

        return issues

    def get_effective_config(self) -> Dict[str, Any]:
        """Get the effective configuration as a dictionary"""
        return asdict(self.config)


# ===== CONFIGURATION UTILITIES =====

def load_config(config_path: Optional[str] = None) -> SAMConfig:
    """Load SAM configuration from file or environment"""
    manager = ConfigManager(config_path)
    return manager.load_config()


def save_config(config: SAMConfig, config_path: Optional[str] = None) -> bool:
    """Save SAM configuration to file"""
    manager = ConfigManager(config_path)
    return manager.save_config(config)


def get_default_config() -> SAMConfig:
    """Get default SAM configuration"""
    return SAMConfig()


def validate_config(config: SAMConfig) -> List[str]:
    """Validate SAM configuration"""
    manager = ConfigManager()
    manager.config = config
    return manager.validate_config()


# ===== PLATFORM-SPECIFIC DEFAULTS =====

def get_platform_defaults() -> Dict[str, Any]:
    """Get platform-specific default configurations"""
    system = platform.system().lower()

    defaults = {}

    if system == "windows":
        defaults.update({
            "config_dir": "~/AppData/Local/SAM",
            "data_dir": "~/AppData/Local/SAM/data",
            "cache_dir": "~/AppData/Local/SAM/cache",
            "log_dir": "~/AppData/Local/SAM/logs"
        })
    elif system == "darwin":  # macOS
        defaults.update({
            "config_dir": "~/Library/Application Support/SAM",
            "data_dir": "~/Library/Application Support/SAM/data",
            "cache_dir": "~/Library/Caches/SAM",
            "log_dir": "~/Library/Logs/SAM"
        })
    else:  # Linux and others
        defaults.update({
            "config_dir": "~/.config/sam",
            "data_dir": "~/.local/share/sam",
            "cache_dir": "~/.cache/sam",
            "log_dir": "~/.local/share/sam/logs"
        })

    return defaults


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Example usage
    print("🔧 SAM Configuration System")

    # Load or create default config
    config = load_config()

    # Validate configuration
    issues = validate_config(config)
    if issues:
        print("⚠️ Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration is valid")

    # Print some key settings
    print(f"\n📋 Key Settings:")
    print(f"  Agent: {config.agent_name} v{config.version}")
    print(f"  Model: {config.model.model_name}")
    print(f"  Base URL: {config.lmstudio.base_url}")
    print(f"  Context Limit: {config.model.context_limit:,} tokens")
    print(f"  Tools Enabled: {config.enable_tools}")
    print(f"  Safe Mode: {config.tools.safe_mode}")
    print(f"  Config Dir: {config.config_dir}")