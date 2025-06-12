#!/usr/bin/env python3
"""
Claude-OpenRouter MCP Server
Enables Claude Code to collaborate with various AI models via OpenRouter
"""

import json
import sys
import os
import requests
from typing import Dict, Any, Optional, List

# Ensure unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# Server version
__version__ = "2.0.0"

# OpenRouter configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_API_KEY_HERE")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default model - can be overridden
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash")

# Available models (you can extend this list based on OpenRouter's offerings)
AVAILABLE_MODELS = [
    "google/gemini-2.0-flash",
    "google/gemini-pro",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
    "openai/gpt-4-turbo",
    "openai/gpt-4",
    "openai/gpt-3.5-turbo",
    "meta-llama/llama-3-70b-instruct",
    "mistralai/mixtral-8x7b-instruct",
]

# Check if API key is configured
API_CONFIGURED = OPENROUTER_API_KEY != "YOUR_API_KEY_HERE"

def send_response(response: Dict[str, Any]):
    """Send a JSON-RPC response"""
    print(json.dumps(response), flush=True)

def handle_initialize(request_id: Any) -> Dict[str, Any]:
    """Handle initialization"""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "claude-openrouter-mcp",
                "version": __version__
            }
        }
    }

def handle_tools_list(request_id: Any) -> Dict[str, Any]:
    """List available tools"""
    tools = []
    
    if API_CONFIGURED:
        tools = [
            {
                "name": "ask_ai",
                "description": "Ask any AI model via OpenRouter and get the response directly in Claude's context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The question or prompt for the AI model"
                        },
                        "model": {
                            "type": "string",
                            "description": f"Model to use (default: {DEFAULT_MODEL})",
                            "enum": AVAILABLE_MODELS,
                            "default": DEFAULT_MODEL
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Temperature for response (0.0-1.0)",
                            "default": 0.5
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens in response",
                            "default": 4096
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "ai_code_review",
                "description": "Have an AI model review code and return feedback directly to Claude",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to review"
                        },
                        "focus": {
                            "type": "string",
                            "description": "Specific focus area (security, performance, etc.)",
                            "default": "general"
                        },
                        "model": {
                            "type": "string",
                            "description": f"Model to use (default: {DEFAULT_MODEL})",
                            "enum": AVAILABLE_MODELS,
                            "default": DEFAULT_MODEL
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "ai_brainstorm",
                "description": "Brainstorm solutions with any AI model, response visible to Claude",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic to brainstorm about"
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context",
                            "default": ""
                        },
                        "model": {
                            "type": "string",
                            "description": f"Model to use (default: {DEFAULT_MODEL})",
                            "enum": AVAILABLE_MODELS,
                            "default": DEFAULT_MODEL
                        }
                    },
                    "required": ["topic"]
                }
            },
            {
                "name": "list_models",
                "description": "List all available AI models",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    else:
        tools = [
            {
                "name": "server_info",
                "description": "Get server status and configuration information",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "tools": tools
        }
    }

def call_openrouter(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL, 
                   temperature: float = 0.5, max_tokens: int = 4096) -> str:
    """Call OpenRouter API and return response"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/your-repo/claude-openrouter-mcp",
            "X-Title": "Claude-OpenRouter MCP Server",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error calling OpenRouter API: {str(e)}"
    except KeyError as e:
        return f"Unexpected response format: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def handle_tool_call(request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tool execution"""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    try:
        result = ""
        
        if tool_name == "server_info":
            if API_CONFIGURED:
                result = f"Server v{__version__} - OpenRouter connected with default model: {DEFAULT_MODEL}"
            else:
                result = f"Server v{__version__} - Please set OPENROUTER_API_KEY environment variable"
        
        elif tool_name == "list_models":
            result = "Available models:\n" + "\n".join(f"- {model}" for model in AVAILABLE_MODELS)
            result += f"\n\nCurrent default: {DEFAULT_MODEL}"
        
        elif tool_name == "ask_ai":
            if not API_CONFIGURED:
                result = "OpenRouter API key not configured. Please set OPENROUTER_API_KEY environment variable."
            else:
                prompt = arguments.get("prompt", "")
                model = arguments.get("model", DEFAULT_MODEL)
                temperature = arguments.get("temperature", 0.5)
                max_tokens = arguments.get("max_tokens", 4096)
                
                messages = [{"role": "user", "content": prompt}]
                result = call_openrouter(messages, model, temperature, max_tokens)
                result = f"[Model: {model}]\n\n{result}"
            
        elif tool_name == "ai_code_review":
            if not API_CONFIGURED:
                result = "OpenRouter API key not configured. Please set OPENROUTER_API_KEY environment variable."
            else:
                code = arguments.get("code", "")
                focus = arguments.get("focus", "general")
                model = arguments.get("model", DEFAULT_MODEL)
                
                prompt = f"""Please review this code with a focus on {focus}:

```
{code}
```

Provide specific, actionable feedback on:
1. Potential issues or bugs
2. Security concerns
3. Performance optimizations
4. Best practices
5. Code clarity and maintainability"""
                
                messages = [{"role": "user", "content": prompt}]
                result = call_openrouter(messages, model, 0.2, 8192)
                result = f"[Model: {model}]\n\n{result}"
            
        elif tool_name == "ai_brainstorm":
            if not API_CONFIGURED:
                result = "OpenRouter API key not configured. Please set OPENROUTER_API_KEY environment variable."
            else:
                topic = arguments.get("topic", "")
                context = arguments.get("context", "")
                model = arguments.get("model", DEFAULT_MODEL)
                
                prompt = f"Let's brainstorm about: {topic}"
                if context:
                    prompt += f"\n\nContext: {context}"
                prompt += "\n\nProvide creative ideas, alternatives, and considerations."
                
                messages = [{"role": "user", "content": prompt}]
                result = call_openrouter(messages, model, 0.7, 8192)
                result = f"[Model: {model}]\n\n{result}"
            
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": f"🤖 AI RESPONSE:\n\n{result}"
                    }
                ]
            }
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }

def main():
    """Main server loop"""
    # Check configuration on startup
    if not API_CONFIGURED:
        sys.stderr.write("Warning: OPENROUTER_API_KEY not configured. Limited functionality available.\n")
        sys.stderr.write("Please set OPENROUTER_API_KEY environment variable to enable AI features.\n")
        sys.stderr.flush()
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            request = json.loads(line.strip())
            method = request.get("method")
            request_id = request.get("id")
            params = request.get("params", {})
            
            if method == "initialize":
                response = handle_initialize(request_id)
            elif method == "tools/list":
                response = handle_tools_list(request_id)
            elif method == "tools/call":
                response = handle_tool_call(request_id, params)
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            
            send_response(response)
            
        except json.JSONDecodeError:
            continue
        except EOFError:
            break
        except Exception as e:
            if 'request_id' in locals():
                send_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                })

if __name__ == "__main__":
    main()