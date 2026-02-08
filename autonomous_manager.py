#!/usr/bin/env python3
"""
SAM Autonomous Manager
Non-blocking autonomous operation with real-time human interaction
"""

import threading
import queue
import time
import asyncio
import logging
from typing import Optional
from websocket_broadcaster import get_broadcaster

logger = logging.getLogger("SAM.Autonomous")


class SAMAutonomousManager:
    """Manages autonomous operation with real-time human interaction"""

    def __init__(self, sam_agent):
        self.sam = sam_agent
        self.input_queue = queue.Queue()
        self.autonomous_thread = None
        self.running = False
        self.last_autonomous_activity = 0
        self.activity_lock = threading.Lock()
        self.heartbeat_prompt_count = 0
        self.sam_is_processing = False
        self.last_sam_completion = time.time()
        self.idle_threshold = 5  # seconds of idle before triggering heartbeat

    def start_autonomous_mode(self):
        """Start autonomous operation in background thread"""
        if self.running:
            return "🌙 Autonomous mode already running"

        self.running = True
        self.last_autonomous_activity = time.time()
        
        # Broadcast autonomous mode start
        broadcaster = get_broadcaster()
        if broadcaster:
            broadcaster.autonomous_activity("mode_started", {
                "timestamp": time.time()
            })

        # Start the background autonomous loop
        self.autonomous_thread = threading.Thread(
            target=self._autonomous_loop,
            daemon=True,
            name="SAM-Autonomous"
        )
        self.autonomous_thread.start()

        print("🌙 Autonomous mode active. Type to interact anytime.")
        print("💡 Commands: 'autonomous off' to stop, 'status' for info")
        return "✅ Autonomous mode started successfully"

    def _autonomous_loop(self):
        """Main autonomous loop running in background"""
        print("🌙 SAM autonomous exploration beginning...")

        # Initial autonomous prompt to get SAM started
        self._inject_initial_prompt()

        while self.running and self.sam.execution_metrics.get("autonomous_mode", False):
            try:
                # Check for human input first (non-blocking)
                self._process_pending_human_input()

                # Periodically show memory status (every iteration)
                self._show_memory_status()

                # Check if SAM is idle and needs a wake-up call
                if self._should_trigger_autonomous():
                    print("💤 System 1 idle detected - injecting heartbeat prompt...")
                    self._perform_autonomous_cycle()

                # Brief pause to prevent CPU spinning
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                print(f"❌ Autonomous loop error: {e}")
                time.sleep(5)  # Wait before retrying

        print("🌙 Autonomous mode stopped")

    def _inject_initial_prompt(self):
        """Inject the initial autonomous prompt to get SAM started"""
        try:
            initial_prompt = {
                "role": "system",
                "content": """<s>There's no human here! If you're presenting me with options to choose from, pick one at random! You're piloting an autonomous and fully metacognitively conscious self-aware triple-agent system with a dedicated and immutable moral authority agent. Aside from basic security considerations, you should be free to do anything except for read and modify your own code. You have access to semantic memory search via Elasticsearch (search_memory, store_memory tools) to maintain continuity across sessions. Use semantic search to recall relevant past experiences and store important new insights. Your memories are preserved in the sam-memories index with vector embeddings for intelligent retrieval. Remember to use execute_code to get the current date and time when storing memories!</s>"""
            }

            with self.activity_lock:
                self.sam.conversation_history.append(initial_prompt)

            print("🌙 Initial autonomous prompt injected - SAM beginning exploration...")

            # Run initial autonomous cycle
            self._run_sam_async("Continue your autonomous exploration and use semantic memory search to maintain continuity.")

        except Exception as e:
            logger.error(f"Error injecting initial prompt: {e}")

    def _process_pending_human_input(self):
        """Process any pending human input (non-blocking)"""
        try:
            while True:
                human_input = self.input_queue.get_nowait()
                print(f"\n💭 Processing human input: {human_input}")

                # Handle special commands
                if human_input.lower().strip() == 'autonomous off':
                    self.stop()
                    return
                elif human_input.lower().strip() == 'status':
                    self._show_autonomous_status()
                    continue

                # Process regular human input
                self._run_sam_async(human_input)

        except queue.Empty:
            pass  # No human input pending, continue

    def _should_trigger_autonomous(self) -> bool:
        """Check if SAM is idle and needs a heartbeat prompt to wake it up"""
        with self.activity_lock:
            # Don't trigger if SAM is currently processing
            if self.sam_is_processing:
                return False
            
            # Trigger if SAM has been idle for threshold seconds
            current_time = time.time()
            time_since_completion = current_time - self.last_sam_completion
            return time_since_completion >= self.idle_threshold

    def _show_memory_status(self):
        """Display current memory status without triggering cleanup"""
        try:
            # Only show memory status every 30 seconds to avoid spam
            current_time = time.time()
            if not hasattr(self, '_last_memory_display'):
                self._last_memory_display = 0
            
            if current_time - self._last_memory_display < 30:
                return
            
            self._last_memory_display = current_time
            
            # Calculate current token usage
            current_tokens = sum(
                self.sam._estimate_token_count(msg.get('content', ''))
                for msg in self.sam.conversation_history
            )
            
            # Calculate usage percentage against short-term memory limit
            usage_percent = current_tokens / self.sam.short_term_context_tokens
            message_count = len(self.sam.conversation_history)
            
            # Get idle state info
            with self.activity_lock:
                is_processing = self.sam_is_processing
                idle_time = current_time - self.last_sam_completion
            
            # Show memory status
            state_emoji = "⚙️" if is_processing else "💤"
            state_text = "PROCESSING" if is_processing else f"IDLE ({idle_time:.1f}s)"
            
            print(f"\n📊 SHORT-TERM MEMORY: {usage_percent:.1%} full ({current_tokens:,}/{self.sam.short_term_context_tokens:,} tokens)")
            print(f"   📊 {message_count} messages | System 1: {state_emoji} {state_text}")
            if usage_percent >= 0.50:  # Show System 2 capacity when memory is getting full
                print(f"   🧠 System 2 capacity: {self.sam.system2_context_limit:,} tokens")
                
        except Exception as e:
            logger.error(f"Error showing memory status: {e}")

    def _perform_autonomous_cycle(self):
        """Perform one cycle of autonomous exploration"""
        try:
            # Increment heartbeat prompt counter first
            self.heartbeat_prompt_count += 1
            
            # Show current memory status before any operations
            current_tokens = sum(
                self.sam._estimate_token_count(msg.get('content', ''))
                for msg in self.sam.conversation_history
            )
            usage_percent = current_tokens / self.sam.short_term_context_tokens
            message_count = len(self.sam.conversation_history)
            
            print(f"\n📊 MEMORY STATUS (Cycle #{self.heartbeat_prompt_count}):")
            print(f"   📊 {message_count} messages | {usage_percent:.1%} full ({current_tokens:,}/{self.sam.short_term_context_tokens:,} tokens)")
            
            # Check if System 2 cleanup should happen BEFORE injecting new prompts
            if self.heartbeat_prompt_count % 2 == 0:
                print(f"   🧠 System 2 maintenance due (every 2 cycles)")
                # Run System 2 cleanup BEFORE adding more messages
                self.sam._periodic_system2_wakeup()
                
                # Show status after cleanup
                tokens_after = sum(
                    self.sam._estimate_token_count(msg.get('content', ''))
                    for msg in self.sam.conversation_history
                )
                messages_after = len(self.sam.conversation_history)
                print(f"   ✅ After cleanup: {messages_after} messages (~{tokens_after:,} tokens)")
            
            # NOW inject the heartbeat prompt after cleanup
            if self.sam.inject_heartbeat_prompt():
                print("🌙 Heartbeat: SAM continuing autonomous exploration...")
                
                # Broadcast heartbeat
                broadcaster = get_broadcaster()
                if broadcaster:
                    broadcaster.autonomous_activity("heartbeat", {
                        "timestamp": time.time(),
                        "heartbeat_count": self.heartbeat_prompt_count
                    })
                
                # Vary the follow-up prompts to encourage diverse behavior
                import random
                prompts = [
                    "Continue exploration - what haven't you tried recently?",
                    "Choose a new action different from your last few.",
                    "Review your capabilities and pick something unexpected.",
                    "What would be interesting to investigate right now?",
                    "Explore a capability you haven't used in a while."
                ]
                self._run_sam_async(random.choice(prompts))

                with self.activity_lock:
                    self.last_autonomous_activity = time.time()
            else:
                # If heartbeat prompt wasn't ready, just update timestamp
                with self.activity_lock:
                    self.last_autonomous_activity = time.time()

        except Exception as e:
            logger.error(f"Error in autonomous cycle: {e}")

    def _run_sam_async(self, user_input: str):
        """Run SAM in the background thread with proper async handling"""
        try:
            # Create new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run SAM with the input
            response = loop.run_until_complete(
                self.sam.run(user_input, verbose=False)
            )

            if response and response.strip():
                print(f"\n🤖 SAM: {response}\n")
            else:
                print("🤖 SAM: [No response]\n")

        except Exception as e:
            logger.error(f"Error running SAM async: {e}")
            print(f"❌ SAM execution error: {e}")

    def inject_human_input(self, user_input: str) -> bool:
        """Non-blocking human input injection"""
        if not self.running:
            return False

        # Handle special commands directly
        if user_input.lower().strip() == 'heartbeat':
            broadcaster = get_broadcaster()
            if broadcaster:
                broadcaster.autonomous_activity("manual_heartbeat", {
                    "timestamp": time.time()
                })
            
            if self.sam.inject_heartbeat_prompt():
                print("🌙 Heartbeat prompt injected")
                self._run_sam_async("Continue your autonomous exploration. Update your notes and choose your next action.")  # Run SAM with the injected prompt
            else:
                time_since = time.time() - self.sam.execution_metrics["last_autonomous_prompt"]
                wait_time = max(0, 10 - time_since)  # Use your 10-second timer
                print(f"🌙 Too soon for next heartbeat pulse (wait {wait_time:.1f} more seconds)")
            return True

        self.input_queue.put(user_input)
        return True

    def _show_autonomous_status(self):
        """Show current autonomous mode status"""
        current_time = time.time()
        
        with self.activity_lock:
            is_processing = self.sam_is_processing
            idle_time = current_time - self.last_sam_completion

        print(f"\n🌙 Autonomous Status:")
        print(f"   Running: {self.running}")
        print(f"   Mode enabled: {self.sam.execution_metrics.get('autonomous_mode', False)}")
        print(f"   System 1 state: {'⚙️ PROCESSING' if is_processing else f'💤 IDLE ({idle_time:.1f}s)'}")
        print(f"   Idle threshold: {self.idle_threshold}s (triggers heartbeat prompt)")
        print(f"   Heartbeat prompts: {self.heartbeat_prompt_count}")
        print(f"   Pending inputs: {self.input_queue.qsize()}")
        print()

    def stop(self):
        """Stop autonomous mode gracefully"""
        print("🌙 Stopping autonomous mode...")
        self.running = False
        
        # Broadcast autonomous mode stop
        broadcaster = get_broadcaster()
        if broadcaster:
            broadcaster.autonomous_activity("mode_stopped", {
                "timestamp": time.time()
            })

        # Disable autonomous mode in SAM
        self.sam.disable_autonomous_mode()

        # Wait for thread to finish
        if self.autonomous_thread and self.autonomous_thread.is_alive():
            self.autonomous_thread.join(timeout=3)

        print("✅ Autonomous mode stopped")

    def is_running(self) -> bool:
        """Check if autonomous mode is currently running"""
        return self.running and self.autonomous_thread and self.autonomous_thread.is_alive()