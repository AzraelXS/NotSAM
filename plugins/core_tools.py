#!/usr/bin/env python3
"""
Core Tools Plugin for SAM Agent
Essential tools for code execution, documentation, and basic utilities
"""

import os
import sys
import json
import math
import re
import time
import random
import subprocess
import platform
import signal
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional
import inspect
import ast
import traceback
import urllib.request
import urllib.parse

# Simple, clean import - no complex fallback logic needed
from sam_agent import SAMPlugin, ToolCategory


class CoreToolsPlugin(SAMPlugin):
    """Core tools for SAM Agent"""

    def __init__(self):
        super().__init__(
            name="Core Tools",
            version="1.0.0",
            description="Essential tools including code execution, search, and calculations"
        )
        self.execution_history = []

    def register_tools(self, agent):
        """Register all core tools with the agent"""

        agent.register_local_tool(
            self.execute_code,
            category=ToolCategory.DEVELOPMENT,
            requires_approval=False
        )

        agent.register_local_tool(
            self.get_current_time,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

        agent.register_local_tool(
            self.get_weather,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

        agent.register_local_tool(
            self.get_system_info,
            category=ToolCategory.SYSTEM,
            requires_approval=False
        )

        agent.register_local_tool(
            self.add_steps_to_plan,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

    def add_steps_to_plan(self, new_steps: list) -> str:
        """
        Add new steps to the current active plan.
        
        Use this when you realize additional steps are needed that weren't in your original plan.
        For example, if you planned to gather data but forgot to include the final email step.
        
        Args:
            new_steps: List of step descriptions (strings) to add
                      Example: ["send_email to user with results", "cleanup temporary files"]
            
        Returns:
            Success message or error
        """
        try:
            agent = self.agent if hasattr(self, 'agent') else None
            if not agent:
                return "❌ Error: Agent reference not available"
            
            # Check if there's an active plan
            if "current_plan_id" not in agent.execution_metrics:
                return "❌ Error: No active plan to update. Create a plan first by outputting multiple tool calls."
            
            if "current_plan_steps" not in agent.execution_metrics:
                return "❌ Error: No plan steps found in execution metrics"
            
            plan_id = agent.execution_metrics["current_plan_id"]
            current_steps = agent.execution_metrics["current_plan_steps"]
            original_count = len(current_steps)
            
            # Validate new_steps format
            if not isinstance(new_steps, list) or len(new_steps) == 0:
                return "❌ Error: new_steps must be a non-empty list of step descriptions (strings)"
            
            # Add new steps to the plan
            next_order = original_count + 1
            added_steps = []
            
            for step_desc in new_steps:
                if not isinstance(step_desc, str):
                    return f"❌ Error: Each step must be a string description. Got: {type(step_desc)}"
                
                new_step = {
                    "description": step_desc,
                    "status": "pending",
                    "order": next_order,
                    "tool_name": "unknown",
                    "tool_args": {},
                    "added_during": "runtime_extension"
                }
                
                current_steps.append(new_step)
                added_steps.append(step_desc)
                next_order += 1
            
            # Update the cached plan
            agent.execution_metrics["current_plan_steps"] = current_steps
            
            # Also update in Elasticsearch if memory manager available
            if hasattr(agent, 'memory_manager') and agent.memory_manager:
                try:
                    from sam_agent import logger
                    # Fetch current plan from ES
                    plan = agent.memory_manager.get_plan(plan_id)
                    if plan:
                        # Update steps
                        agent.memory_manager.es.update(
                            index='sam_plans',
                            id=plan_id,
                            body={
                                'doc': {
                                    'steps': current_steps,
                                    'total_steps': len(current_steps),
                                    'updated_at': time.time()
                                }
                            }
                        )
                        logger.info(f"📋 Updated plan {plan_id} in Elasticsearch with {len(new_steps)} new steps")
                except Exception as es_error:
                    # Continue anyway - cached version is updated
                    pass
            
            result = []
            result.append(f"✅ Added {len(new_steps)} step(s) to current plan")
            result.append(f"   📊 Plan now has {len(current_steps)} total steps (was {original_count})")
            result.append(f"   New steps:")
            for i, step in enumerate(added_steps, original_count + 1):
                result.append(f"   {i}. {step}")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"❌ Error adding steps to plan: {str(e)}"

    def execute_code(self, code: str, language: str = "python", timeout: int = 30) -> str:
        """
        Execute code safely in a controlled environment.

        Args:
            code: Code to execute
            language: Programming language (python, javascript, bash, powershell)
            timeout: Maximum execution time in seconds

        Returns:
            Execution result or error message
        """
        if language.lower() not in ["python", "javascript", "bash", "shell", "powershell", "pwsh"]:
            return f"❌ Unsupported language: {language}"

        try:
            if language.lower() == "python":
                return self._execute_python_code(code, timeout)
            elif language.lower() == "javascript":
                return self._execute_javascript_code(code)
            elif language.lower() in ["bash", "shell"]:
                return self._execute_shell_code(code)
            elif language.lower() in ["powershell", "pwsh"]:
                return self._execute_powershell_code(code)
        except Exception as e:
            return f"❌ Execution error: {str(e)}"

    def _execute_python_code(self, code: str, timeout: int = 30) -> str:
        """Execute Python code with proper import support"""
        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            result = None
            error = None
            start_time = time.time()

            try:
                # Create execution environment with proper imports
                exec_globals = {
                    '__builtins__': __builtins__,
                    # Pre-import commonly needed modules
                    'os': os,
                    'sys': sys,
                    'json': json,
                    'math': math,
                    'random': random,
                    'datetime': datetime,
                    'time': time,
                    're': re,
                    'pathlib': Path,
                    'platform': platform,
                    'subprocess': subprocess,
                }

                # Parse and execute code
                tree = ast.parse(code)

                # If last node is an expression, evaluate it for return value
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    # Execute all but last statement
                    if len(tree.body) > 1:
                        statements = ast.Module(body=tree.body[:-1], type_ignores=[])
                        exec(compile(statements, '<string>', 'exec'), exec_globals)

                    # Evaluate final expression
                    expr = ast.Expression(body=tree.body[-1].value)
                    result = eval(compile(expr, '<string>', 'eval'), exec_globals)
                else:
                    # Execute as statements
                    exec(code, exec_globals)

            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"

            execution_time = time.time() - start_time

            # Collect output
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()

            # Format response
            output_parts = [
                f"🐍 Python Code Executed ({execution_time:.3f}s)"
            ]

            if stdout_text.strip():
                output_parts.append(f"\n📤 Output:\n{stdout_text}")

            if result is not None:
                output_parts.append(f"\n🔢 Return Value: {repr(result)}")

            if stderr_text.strip():
                output_parts.append(f"\n⚠️  Stderr:\n{stderr_text}")

            if error:
                output_parts.append(f"\n❌ Error: {error}")

            # Show subprocess results if no output
            if not stdout_text.strip() and result is None and not stderr_text.strip() and not error:
                output_parts.append("\n⚠️  No output captured - check if subprocess commands are working correctly")

            return "".join(output_parts)

        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _execute_javascript_code(self, code: str) -> str:
        """Execute JavaScript code using Node.js"""
        try:
            # Write code to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Execute with Node.js
                result = subprocess.run(
                    ['node', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                output = ""
                if result.stdout:
                    output += f"📤 Output:\n{result.stdout}"
                if result.stderr:
                    output += f"\n⚠️  Errors:\n{result.stderr}"

                if result.returncode != 0:
                    output += f"\n❌ Exit code: {result.returncode}"

                return output or "✅ JavaScript executed successfully (no output)"

            finally:
                os.unlink(temp_file)

        except subprocess.TimeoutExpired:
            return "❌ JavaScript execution timed out"
        except FileNotFoundError:
            return "❌ Node.js not found. Please install Node.js to execute JavaScript."
        except Exception as e:
            return f"❌ JavaScript execution error: {str(e)}"

    def _execute_shell_code(self, code: str) -> str:
        """Execute shell commands safely"""
        # Basic safety checks
        dangerous_commands = ['rm -rf', 'sudo', 'passwd', 'chmod 777', 'dd if=', 'format', 'del /']
        if any(cmd in code.lower() for cmd in dangerous_commands):
            return "❌ Potentially dangerous command detected. Execution blocked for safety."

        try:
            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            output = ""
            if result.stdout:
                output += f"📤 Output:\n{result.stdout}"
            if result.stderr:
                output += f"\n⚠️  Errors:\n{result.stderr}"

            if result.returncode != 0:
                output += f"\n❌ Exit code: {result.returncode}"

            return output or "✅ Shell command executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return "❌ Shell command timed out"
        except Exception as e:
            return f"❌ Shell execution error: {str(e)}"

    def _execute_powershell_code(self, code: str) -> str:
        """Execute PowerShell commands"""
        try:
            # Use PowerShell to execute the command
            result = subprocess.run(
                ["powershell", "-Command", code],
                capture_output=True,
                text=True,
                timeout=30
            )

            output = ""
            if result.stdout:
                output += f"📤 Output:\n{result.stdout}"
            if result.stderr:
                output += f"\n⚠️  Errors:\n{result.stderr}"

            if result.returncode != 0:
                output += f"\n❌ Exit code: {result.returncode}"

            return output or "✅ PowerShell command executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return "❌ PowerShell command timed out"
        except FileNotFoundError:
            return "❌ PowerShell not found. Please ensure PowerShell is installed and in PATH."
        except Exception as e:
            return f"❌ PowerShell execution error: {str(e)}"

    def get_current_time(self, format: str = "%m-%d-%Y %H:%M:%S", timezone: str = "local") -> str:
        """
        Get the current date and time.

        Args:
            format: Time format string (default: "%m-%d-%Y %H:%M:%S")
            timezone: Timezone (currently only supports "local")

        Returns:
            Formatted current time
        """
        try:
            now = datetime.now()
            formatted_time = now.strftime(format)
            return f"🕒 Current time: {formatted_time}"
        except Exception as e:
            return f"❌ Time format error: {str(e)}"



    def get_system_info(self) -> str:
        """
        Get system information.

        Returns:
            System information summary
        """
        try:
            info = {
                "platform": platform.platform(),
                "system": platform.system(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            }

            result = "💻 System Information:\n"
            result += f"🖥️  Platform: {info['platform']}\n"
            result += f"⚙️  System: {info['system']}\n"
            result += f"🔧 Processor: {info['processor']}\n"
            result += f"🏗️  Architecture: {info['architecture'][0]}\n"
            result += f"🐍 Python: {info['python_version']}\n"
            result += f"🌐 Hostname: {info['hostname']}"

            return result

        except Exception as e:
            return f"❌ Error getting system info: {str(e)}"

    def get_weather(self, location: str = None) -> str:
        """
        Get current weather information using Open-Meteo API.
        
        Call without arguments to use the user's default location from config.json,
        or specify a location to get weather for a different place.

        Args:
            location: City name or location (optional - if not provided, uses location from config.json)

        Returns:
            Current weather information
            
        Examples:
            get_weather() - Gets weather for default location in config.json
            get_weather("Tokyo") - Gets weather for Tokyo
            get_weather("London") - Gets weather for London
        """
        try:
            # Load config for default location
            if location is None:
                config_path = Path("config.json")
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        location = config.get("user", {}).get("location", "New York")
                else:
                    location = "New York"

            # Try to geocode location - if it fails, try just the first word (city name)
            geocode_data = None
            search_terms = [location]
            
            # If location has commas or multiple words, also try just the first word
            if ',' in location or ' ' in location:
                first_word = location.split(',')[0].split()[0].strip()
                if first_word != location:
                    search_terms.append(first_word)
            
            for search_term in search_terms:
                geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(search_term)}&count=1&language=en&format=json"
                
                with urllib.request.urlopen(geocode_url, timeout=10) as response:
                    data = json.loads(response.read().decode())
                
                if data.get("results"):
                    geocode_data = data
                    break
            
            if not geocode_data or not geocode_data.get("results"):
                return f"❌ Location not found: {location}"
            
            result = geocode_data["results"][0]
            lat = result["latitude"]
            lon = result["longitude"]
            location_name = result["name"]
            country = result.get("country", "")
            admin1 = result.get("admin1", "")  # State/province
            
            # Get weather data
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m&temperature_unit=fahrenheit&wind_speed_unit=mph"
            
            with urllib.request.urlopen(weather_url, timeout=10) as response:
                weather_data = json.loads(response.read().decode())
            
            current = weather_data["current"]
            
            # Weather code descriptions (WMO codes)
            weather_codes = {
                0: "☀️ Clear sky",
                1: "🌤️ Mainly clear",
                2: "⛅ Partly cloudy",
                3: "☁️ Overcast",
                45: "🌫️ Fog",
                48: "🌫️ Depositing rime fog",
                51: "🌦️ Light drizzle",
                53: "🌦️ Moderate drizzle",
                55: "🌧️ Dense drizzle",
                61: "🌧️ Slight rain",
                63: "🌧️ Moderate rain",
                65: "🌧️ Heavy rain",
                71: "🌨️ Slight snow",
                73: "🌨️ Moderate snow",
                75: "🌨️ Heavy snow",
                77: "🌨️ Snow grains",
                80: "🌦️ Slight rain showers",
                81: "🌧️ Moderate rain showers",
                82: "⛈️ Violent rain showers",
                85: "🌨️ Slight snow showers",
                86: "🌨️ Heavy snow showers",
                95: "⛈️ Thunderstorm",
                96: "⛈️ Thunderstorm with slight hail",
                99: "⛈️ Thunderstorm with heavy hail"
            }
            
            weather_desc = weather_codes.get(current["weather_code"], "❓ Unknown")
            
            # Build location string
            location_str = location_name
            if admin1:
                location_str += f", {admin1}"
            if country:
                location_str += f", {country}"
            
            result_text = f"🌍 Weather for {location_str}\n"
            result_text += f"\n{weather_desc}\n"
            result_text += f"🌡️ Temperature: {current['temperature_2m']}°F\n"
            result_text += f"🤔 Feels like: {current['apparent_temperature']}°F\n"
            result_text += f"💧 Humidity: {current['relative_humidity_2m']}%\n"
            result_text += f"💨 Wind: {current['wind_speed_10m']} mph\n"
            result_text += f"🌧️ Precipitation: {current['precipitation']} mm"
            
            return result_text
            
        except urllib.error.URLError as e:
            return f"❌ Network error: {str(e)}"
        except json.JSONDecodeError as e:
            return f"❌ Failed to parse weather data: {str(e)}"
        except Exception as e:
            return f"❌ Error getting weather: {str(e)}"


# Create the plugin instance
def get_plugin():
    """Plugin entry point"""
    return CoreToolsPlugin()


# For direct execution testing
if __name__ == "__main__":
    plugin = CoreToolsPlugin()

    # Test the execute_code function
    test_code = """
from datetime import datetime
current_time = datetime.now()
print(current_time.strftime('%Y-%m-%d %H:%M:%S'))
"""

    print("Testing execute_code function:")
    result = plugin.execute_code(test_code, "python")
    print(result)

def create_plugin():
    """Factory function to create the plugin instance"""
    return CoreToolsPlugin()