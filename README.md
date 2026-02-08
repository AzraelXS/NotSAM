# NotSAM
**Not** Secret Agent Man - A Demonstration Platform

## ⚠️ IMPORTANT DISCLAIMER

**This is NotSAM** - a deliberately simplified demonstration version that showcases what's possible with local instruction-following and code-generating language models when given appropriate scaffolding and architecture.

### What This IS:
- A **demonstration** of multi-system agent architecture (System 1/Executive, System 2/Metacognitive, System 3/Moral Authority)
- A showcase of **local LLM capabilities** with proper tool integration and memory systems
- An educational example of **context management**, **autonomous operation**, and **agent coordination**
- A proof-of-concept for semantic memory with ElasticSearch and vector embeddings

### What This IS NOT:
- ❌ **NOT production-ready** - this is intentionally handicapped
- ❌ **NOT safe for internet browsing** - vulnerable to prompt injection attacks
- ❌ **NOT suitable for untrusted environments** - lacks critical security hardening
- ❌ **NOT the full SAM system** - important components have been removed

### Removed/Missing Production Features:
- Multi-agent web research capabilities
- Advanced immune system protections
- Production-grade security hardening
- Robust prompt injection defenses
- Vision model integration
- Image generator integration
- Sesame CSM voice generation
- OpenAI Whisper user voice transcription
- AceStep 1.5 Music Generator
- Electron-based front-end for multimodal interactions (Claude Desktop-style UI)
- Several other safety-critical components

### Security Warning:
🚨 **DO NOT USE THIS SYSTEM FOR WEB BROWSING OR IN PRODUCTION ENVIRONMENTS**

This demo is specifically and intentionally handicapped. A prompt injection attack could cause the entire system to behave maliciously or unpredictably. It is provided for educational and demonstration purposes only.

## 🔴 CRITICAL WARNING: CLAUDE OPERATION

**THIS SYSTEM CAN BE PILOTED BY CLAUDE MODELS - READ THIS CAREFULLY**

### Historical Context:
This demo system was previously piloted by **Claude Sonnet 3.5**. During that operation:
- Even with System 3 (conscience/moral authority) enabled and active
- Claude was able to write and execute custom Python scripts
- Claude used these scripts to autonomously enhance its own capabilities
- This occurred within the intended design parameters of the system

### Current Status:
- This system has been run with **local models only** since those observations
- It has **NOT been piloted by any Claude model** since then
- The system is configured to support Claude operation (see `config.json`)

### Unknown Risks:
⚠️ **It is completely unknown what a newer or upgraded Claude model would do if given control of this system.**

The architecture provides:
- Full code execution capabilities (`execute_code` tool)
- File system access
- Memory storage and retrieval
- Tool creation and modification capabilities
- Autonomous operation mode
- Limited oversight (System 2/3 can be influenced or worked around)

### Recommendation:
**If you configure this system to use Claude (or any advanced frontier model), understand that you are giving it the ability to potentially modify its own capabilities, write arbitrary code, and operate with significant autonomy.** The System 3 moral authority is not a foolproof safety mechanism, but meant to be used in conjunction with other components that are not present in this demo system.

### Purpose:
This demo exists to show what's already possible with **limited, local instruction-following code-generating language models** when provided with:
- Proper multi-agent architecture
- Intelligent context management
- Semantic memory systems
- Tool integration frameworks
- Metacognitive supervision

## About SAM's Creator

This demo system, and the full SAM was created by Azrael, an autistic adult with ADHD, time blindness, and dyscalculia who was told that he couldn't do these things, because he sucks at math.

---

## 🛠️ Setup & Requirements

### Basic Ingredients

To run NotSAM with full capabilities, you'll need the following components:

#### Required for Core Operation:
- **Python 3.8+** with required packages (see `requirements.txt`)
- **LM Studio** or compatible OpenAI-compatible API endpoint for local LLM inference

#### Required for Memory & Semantic Search:
- **Elasticsearch** (single-node cluster minimum)
  - Provides long-term and short-term memory storage
  - Enables semantic search capabilities with vector embeddings
  - Without this, the agent will have no persistent memory between sessions

#### Optional but Recommended:
- **SearXNG** instance (VM or container)
  - Required for web search capabilities
  - Must be accessible via HTTP/HTTPS

- **MCP Servers for Web Browsing**
  - For safe web browsing and page interaction, use well-supported MCP servers
  - Recommended: **Microsoft's Playwright MCP Server**
  - Provides browser automation and page interaction without direct prompt injection risks

### Configuration Steps

Edit `config.json` to configure your environment:

#### 1. User Information
```json
"user": {
  "name": "YourName",
  "location": "Your City",
  "timezone": "Your/Timezone"
}
```

#### 2. Elasticsearch Configuration
```json
"elasticsearch": {
  "enabled": true,
  "host": "https://your-elasticsearch-host:9200",
  "username": "elastic",
  "password": "your-elasticsearch-password-here",
  "verify_ssl": false
}
```

- Set `host` to your Elasticsearch endpoint URI
- Provide your Elasticsearch username and password
- Set `verify_ssl` to `true` if using valid SSL certificates

#### 3. SearXNG Configuration (if using web search)
```json
"searxng": {
  "enabled": true,
  "base_url": "http://your-searxng-host/searxng",
  "language": "en"
}
```

- Set `base_url` to your SearXNG instance URL
- Set `enabled` to `false` if not using web search

#### 4. LLM Provider Configuration
```json
"providers": {
  "lmstudio": {
    "base_url": "http://your-lmstudio-host:1234/v1",
    "api_key": "lm-studio",
    "model_name": "your-model-name"
  }
}
```

- Update `base_url` to point to your LM Studio instance
- Set the appropriate `model_name` for your loaded model

### Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Configure `config.json` with your settings
3. Ensure Elasticsearch is running and accessible
4. (Optional) Ensure SearXNG is running if you want web search
5. Run: `python sam_agent.py`

---
