# SAM WebSocket Broadcasting

Real-time status broadcasting for monitoring SAM's activities via WebSocket connections.

## Overview

The WebSocket broadcaster provides real-time updates for:
- **Tool Calls**: Start, end, results, and errors
- **System 2 Interventions**: Context management, loop breaking, error handling
- **System 3 Evaluations**: Moral authority decisions on tool calls
- **Autonomous Activities**: Heartbeats, mode changes, and exploration events

## Configuration

Add to `config.json`:

```json
{
  "websocket": {
    "enabled": true,
    "host": "localhost",
    "port": 8765,
    "broadcast_tool_calls": true,
    "broadcast_system2": true,
    "broadcast_system3": true,
    "broadcast_autonomous": true
  }
}
```

## Message Types

### Tool Call Events

**tool_call_start**
```json
{
  "type": "tool_call_start",
  "timestamp": "2026-01-17T10:30:00.000Z",
  "data": {
    "tool_name": "execute_code",
    "arguments": {"code": "print('hello')"},
    "context": {}
  }
}
```

**tool_call_end**
```json
{
  "type": "tool_call_end",
  "timestamp": "2026-01-17T10:30:01.000Z",
  "data": {
    "tool_name": "execute_code",
    "success": true,
    "duration": 0.523,
    "error": null
  }
}
```

**tool_call_result**
```json
{
  "type": "tool_call_result",
  "timestamp": "2026-01-17T10:30:01.000Z",
  "data": {
    "tool_name": "execute_code",
    "result": "hello\n",
    "truncated": false
  }
}
```

### System 2 Events

**system2_intervention**
```json
{
  "type": "system2_intervention",
  "timestamp": "2026-01-17T10:30:05.000Z",
  "data": {
    "intervention_type": "token_limit_breach",
    "action": "context_compression",
    "details": {
      "should_break": false,
      "context_modified": true,
      "message": "System 2 intervention: context_compression"
    }
  }
}
```

### System 3 Events

**system3_evaluation**
```json
{
  "type": "system3_evaluation",
  "timestamp": "2026-01-17T10:30:02.000Z",
  "data": {
    "tool_name": "execute_code",
    "decision": "approve",
    "reasoning": "Safe code execution for utility function",
    "confidence": 0.95
  }
}
```

**system3_intervention**
```json
{
  "type": "system3_intervention",
  "timestamp": "2026-01-17T10:30:02.000Z",
  "data": {
    "action": "block_execution",
    "details": {
      "reason": "Protected file access attempt"
    }
  }
}
```

### Autonomous Events

**autonomous_activity**
```json
{
  "type": "autonomous_activity",
  "timestamp": "2026-01-17T10:30:10.000Z",
  "data": {
    "activity_type": "heartbeat",
    "details": {
      "timestamp": 1705488610.123,
      "heartbeat_count": 5
    }
  }
}
```

Activity types:
- `mode_started` - Autonomous mode enabled
- `mode_stopped` - Autonomous mode disabled
- `heartbeat` - Automatic autonomous cycle trigger
- `manual_heartbeat` - User-triggered heartbeat

### Agent Status

**agent_status**
```json
{
  "type": "agent_status",
  "timestamp": "2026-01-17T10:30:15.000Z",
  "data": {
    "status": "processing",
    "details": {
      "iteration": 3,
      "tokens_used": 5432
    }
  }
}
```

### Approval System (Bidirectional)

**approval_request** (Server → Client)
```json
{
  "type": "approval_request",
  "timestamp": "2026-01-17T10:30:20.000Z",
  "data": {
    "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "tool_name": "execute_code",
    "arguments": {
      "code": "print('Hello World')"
    },
    "tool_info": {
      "category": "development",
      "description": "Execute Python code",
      "requires_approval": true
    }
  }
}
```

**approval_response** (Client → Server)
```json
{
  "type": "approval_response",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "response": "approve"
}
```

Response values:
- `"approve"` - Allow the tool to execute
- `"deny"` - Reject the tool execution
- `"stop"` - Stop SAM's execution entirely

## Testing the Broadcaster

### Quick Test with HTML Monitor

1. Start SAM with WebSocket broadcasting enabled
2. Open `websocket_monitor.html` in a web browser
3. Click "Connect" (auto-connects on load)
4. Watch real-time events as SAM operates
5. **NEW**: When SAM requires approval, a modal will appear in the browser
   - Click "✅ Approve" to allow the tool execution
   - Click "❌ Deny" to reject it
   - Click "🛑 Stop SAM" to halt execution entirely

### Test Approval System

Run the test script:
```bash
python test_websocket_approval.py
```

This will:
1. Initialize SAM with safety mode enabled
2. Wait for WebSocket clients to connect
3. Trigger a tool approval request
4. Display the approval in the browser monitor
5. Show the result

### Python Client Example

```python
import asyncio
import websockets
import json

async def monitor_sam():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        print("Connected to SAM WebSocket")
        
        async for message in websocket:
            data = json.loads(message)
            print(f"{data['type']}: {data['data']}")

asyncio.run(monitor_sam())
```

### JavaScript/Node.js Client Example

```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8765');

ws.on('open', () => {
    console.log('Connected to SAM WebSocket');
});

ws.on('message', (data) => {
    const message = JSON.parse(data);
    console.log(`${message.type}:`, message.data);
});

ws.on('close', () => {
    console.log('Disconnected from SAM WebSocket');
});
```

## Integration with Electron

### In main.js

```javascript
const WebSocket = require('ws');

let samWebSocket = null;

function connectToSAM() {
    samWebSocket = new WebSocket('ws://localhost:8765');
    
    samWebSocket.on('message', (data) => {
        const message = JSON.parse(data);
        
        // Forward to renderer process
        if (mainWindow) {
            mainWindow.webContents.send('sam-event', message);
        }
    });
}

// Connect on app ready
app.on('ready', () => {
    createWindow();
    connectToSAM();
});
```

### In renderer process

```javascript
// Listen for SAM events
window.api.receive('sam-event', (message) => {
    switch (message.type) {
        case 'tool_call_start':
            updateToolCallUI(message.data);
            break;
        case 'system2_intervention':
            showSystem2Alert(message.data);
            break;
        // ... handle other event types
    }
});
```

## Architecture

The broadcaster uses:
- **WebSocket Server**: `websockets` library running on background thread
- **Async Queue**: Thread-safe message queue for cross-thread communication
- **Broadcast Pattern**: One-to-many message distribution to all connected clients
- **Auto-reconnect**: Clients can reconnect at any time without data loss

## Requirements

Install the WebSocket library:

```bash
pip install websockets
```

## Troubleshooting

### WebSocket won't start

Check if port 8765 is already in use:
```bash
netstat -ano | findstr :8765
```

Change port in config.json if needed.

### No messages received

1. Verify WebSocket is enabled in config.json
2. Check SAM logs for broadcaster initialization
3. Ensure firewall allows port 8765
4. Test connection with `websocket_monitor.html`

### Connection drops

The broadcaster automatically handles disconnections. Clients can reconnect at any time without affecting SAM's operation.

## Performance

- **Minimal overhead**: Broadcasting happens on separate thread
- **Non-blocking**: Tool execution is never delayed by WebSocket operations
- **Efficient serialization**: JSON messages are pre-formatted
- **Auto-truncation**: Large results are truncated to prevent memory issues

## Security Notes

- Currently binds to `localhost` by default (safe for local use)
- For remote access, use SSH tunneling or VPN
- No authentication currently implemented
- Consider adding WSS (WebSocket Secure) for production use

## Future Enhancements

- [ ] Authentication/authorization
- [ ] Message filtering by type
- [ ] Historical message replay
- [ ] WebSocket compression
- [ ] SSL/TLS support
- [ ] Rate limiting per client
- [ ] Message persistence
