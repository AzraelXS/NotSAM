#!/usr/bin/env python3
"""
WebSocket Broadcaster for SAM
Real-time status broadcasting for Electron UI integration
"""

import asyncio
import json
import logging
import threading
import time
from typing import Dict, Any, List, Set, Optional
from datetime import datetime

try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger("SAM.WebSocket")


class WebSocketBroadcaster:
    """Broadcasts SAM status messages to connected WebSocket clients"""

    def __init__(self, host: str = "localhost", port: int = 8765, enabled: bool = True):
        self.host = host
        self.port = port
        self.enabled = enabled and WEBSOCKETS_AVAILABLE
        self.clients: Set[WebSocketServerProtocol] = set()
        self.server = None
        self.loop = None
        self.thread = None
        self.running = False
        self.message_queue = asyncio.Queue() if self.enabled else None
        
        # Approval request/response system
        self.pending_approvals = {}  # request_id -> asyncio.Event
        self.approval_responses = {}  # request_id -> response
        self.approval_timeout = 300  # 5 minutes default

        if not WEBSOCKETS_AVAILABLE:
            logger.warning("websockets package not available - broadcasting disabled")
            self.enabled = False

    def start(self):
        """Start the WebSocket server in a background thread"""
        if not self.enabled:
            logger.info("WebSocket broadcasting disabled")
            return

        if self.running:
            logger.warning("WebSocket server already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True, name="WebSocket-Server")
        self.thread.start()
        logger.info(f"WebSocket broadcaster starting on ws://{self.host}:{self.port}")

    def _run_server(self):
        """Run the WebSocket server event loop"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._start_server())
        except Exception as e:
            logger.error(f"Error in WebSocket server: {e}")
        finally:
            if self.loop:
                self.loop.close()

    async def _start_server(self):
        """Start the WebSocket server"""
        try:
            async with serve(self._handle_client, self.host, self.port):
                logger.info(f"✅ WebSocket server running on ws://{self.host}:{self.port}")
                # Start message processor
                asyncio.create_task(self._process_messages())
                # Keep server running
                await asyncio.Future()  # Run forever
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            self.running = False

    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a new WebSocket client connection"""
        self.clients.add(websocket)
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"📡 WebSocket client connected: {client_id} (total: {len(self.clients)})")

        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connection",
                "status": "connected",
                "timestamp": datetime.now().isoformat(),
                "message": "Connected to SAM WebSocket broadcaster"
            }))

            # Keep connection alive and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(data, client_id)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {client_id}: {message}")
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"📡 WebSocket client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket client {client_id}: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"📡 Client {client_id} removed (remaining: {len(self.clients)})")

    async def _process_messages(self):
        """Process messages from the queue and broadcast to clients"""
        while self.running:
            try:
                # Wait for messages with timeout
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                    await self._broadcast(message)
                except asyncio.TimeoutError:
                    continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await asyncio.sleep(0.1)

    async def _handle_client_message(self, data: Dict[str, Any], client_id: str):
        """Handle incoming messages from clients"""
        msg_type = data.get('type')
        
        if msg_type == 'approval_response':
            request_id = data.get('request_id')
            response = data.get('response')  # 'approve', 'deny', 'stop'
            
            if request_id in self.pending_approvals:
                self.approval_responses[request_id] = response
                self.pending_approvals[request_id].set()  # Wake up waiting thread
                logger.info(f"📨 Approval response from {client_id}: {response} for {request_id}")
            else:
                logger.warning(f"Received approval response for unknown request: {request_id}")
        
        elif msg_type == 'ping':
            # Simple ping/pong for keepalive
            logger.debug(f"Ping from {client_id}")
        
        else:
            logger.debug(f"Unknown message type from {client_id}: {msg_type}")
    
    async def _broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self.clients:
            return

        message_json = json.dumps(message)
        disconnected = set()

        for client in self.clients:
            try:
                await client.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        self.clients -= disconnected

    def broadcast_sync(self, message_type: str, data: Dict[str, Any]):
        """Thread-safe method to broadcast a message from any thread"""
        if not self.enabled or not self.running:
            return

        message = {
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

        # Add to queue from any thread
        if self.loop and self.message_queue:
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(message),
                self.loop
            )

    # Convenience methods for specific message types

    def tool_call_start(self, tool_name: str, arguments: Dict[str, Any], context: Optional[Dict] = None):
        """Broadcast when a tool call begins"""
        self.broadcast_sync("tool_call_start", {
            "tool_name": tool_name,
            "arguments": arguments,
            "context": context or {}
        })

    def tool_call_end(self, tool_name: str, success: bool, duration: float, error: Optional[str] = None):
        """Broadcast when a tool call ends"""
        self.broadcast_sync("tool_call_end", {
            "tool_name": tool_name,
            "success": success,
            "duration": duration,
            "error": error
        })

    def tool_call_result(self, tool_name: str, result: Any, truncated: bool = False):
        """Broadcast the raw result from a tool call"""
        # Convert result to JSON-serializable format
        try:
            if isinstance(result, (str, int, float, bool, type(None))):
                result_data = result
            elif isinstance(result, (dict, list)):
                result_data = result
            else:
                result_data = str(result)

            # Truncate very large results
            result_str = json.dumps(result_data) if not isinstance(result_data, str) else result_data
            if len(result_str) > 10000:
                result_data = result_str[:10000] + "... [truncated]"
                truncated = True

            self.broadcast_sync("tool_call_result", {
                "tool_name": tool_name,
                "result": result_data,
                "truncated": truncated
            })
        except Exception as e:
            logger.error(f"Error broadcasting tool result: {e}")

    def system2_intervention(self, intervention_type: str, action: str, details: Dict[str, Any]):
        """Broadcast when System 2 intervenes"""
        self.broadcast_sync("system2_intervention", {
            "intervention_type": intervention_type,
            "action": action,
            "details": details
        })

    def system3_evaluation(self, tool_name: str, decision: str, reasoning: str, confidence: float):
        """Broadcast when System 3 evaluates a tool call"""
        self.broadcast_sync("system3_evaluation", {
            "tool_name": tool_name,
            "decision": decision,
            "reasoning": reasoning,
            "confidence": confidence
        })

    def system3_intervention(self, action: str, details: Dict[str, Any]):
        """Broadcast when System 3 takes an intervention action"""
        self.broadcast_sync("system3_intervention", {
            "action": action,
            "details": details
        })

    def autonomous_activity(self, activity_type: str, details: Dict[str, Any]):
        """Broadcast autonomous manager activities"""
        self.broadcast_sync("autonomous_activity", {
            "activity_type": activity_type,
            "details": details
        })

    def agent_status(self, status: str, details: Dict[str, Any]):
        """Broadcast general agent status updates"""
        self.broadcast_sync("agent_status", {
            "status": status,
            "details": details
        })
    
    def approval_request(self, request_id: str, tool_name: str, arguments: Dict[str, Any], 
                        tool_info: Optional[Dict[str, Any]] = None):
        """Broadcast an approval request to clients"""
        self.broadcast_sync("approval_request", {
            "request_id": request_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "tool_info": tool_info or {}
        })
    
    def wait_for_approval(self, request_id: str, timeout: Optional[float] = None) -> Optional[str]:
        """Wait for approval response from a client (blocking, thread-safe)
        
        Returns: 'approve', 'deny', 'stop', or None on timeout
        """
        if not self.enabled or not self.running:
            return None
        
        # Create event for this request
        event = threading.Event()
        self.pending_approvals[request_id] = event
        
        # Wait for response
        timeout_val = timeout or self.approval_timeout
        got_response = event.wait(timeout=timeout_val)
        
        # Get response
        response = self.approval_responses.pop(request_id, None) if got_response else None
        
        # Cleanup
        self.pending_approvals.pop(request_id, None)
        
        if not got_response:
            logger.warning(f"Approval request {request_id} timed out after {timeout_val}s")
        
        return response

    def stop(self):
        """Stop the WebSocket server"""
        if not self.running:
            return

        logger.info("Stopping WebSocket broadcaster...")
        self.running = False

        if self.loop:
            # Close all client connections
            for client in list(self.clients):
                try:
                    asyncio.run_coroutine_threadsafe(client.close(), self.loop)
                except:
                    pass

        if self.thread:
            self.thread.join(timeout=2)

        logger.info("WebSocket broadcaster stopped")


# Global broadcaster instance
_broadcaster: Optional[WebSocketBroadcaster] = None


def get_broadcaster() -> Optional[WebSocketBroadcaster]:
    """Get the global broadcaster instance"""
    return _broadcaster


def init_broadcaster(host: str = "localhost", port: int = 8765, enabled: bool = True) -> WebSocketBroadcaster:
    """Initialize the global broadcaster"""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = WebSocketBroadcaster(host, port, enabled)
        if enabled:
            _broadcaster.start()
    return _broadcaster


def stop_broadcaster():
    """Stop the global broadcaster"""
    global _broadcaster
    if _broadcaster:
        _broadcaster.stop()
        _broadcaster = None
