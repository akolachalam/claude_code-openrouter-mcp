# Claude Code + OpenRouter MCP Server

Connect Claude Code with any AI model through OpenRouter! Use GPT-4, Claude, Gemini, Llama, and more - all within Claude Code!

## 🚀 Quick Start (2 minutes)

### Prerequisites
- Python 3.8+ installed
- Claude Code CLI installed
- OpenRouter API key ([Get one here](https://openrouter.ai/keys))

### Setup

1. **Clone this repo:**
```bash
git clone https://github.com/your-username/claude_code-openrouter-mcp.git
cd claude_code-openrouter-mcp
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set your OpenRouter API key:**
```bash
export OPENROUTER_API_KEY="your_api_key_here"
# Optional: Set default model
export OPENROUTER_MODEL="google/gemini-2.0-flash"
```

4. **Add to Claude Code:**
```bash
claude mcp add openrouter python3 /path/to/claude_code-openrouter-mcp/server.py
```

That's it! 🎉

## 📖 Usage Examples

Start Claude Code and use these tools:

```bash
claude

# Ask any AI model
mcp__openrouter__ask_ai
  prompt: "Explain quantum computing"
  model: "openai/gpt-4-turbo"

# Get code reviews
mcp__openrouter__ai_code_review
  code: "def auth(u): return u.pwd == 'admin'"
  focus: "security"
  model: "anthropic/claude-3-opus"

# Brainstorm with different models
mcp__openrouter__ai_brainstorm
  topic: "How to scale a web app"
  model: "meta-llama/llama-3-70b-instruct"

# List available models
mcp__openrouter__list_models
```

## 🤖 Available Models

- **Google**: gemini-2.0-flash, gemini-pro
- **Anthropic**: claude-3-opus, claude-3-sonnet, claude-3-haiku
- **OpenAI**: gpt-4-turbo, gpt-4, gpt-3.5-turbo
- **Meta**: llama-3-70b-instruct
- **Mistral**: mixtral-8x7b-instruct
- And many more on [OpenRouter](https://openrouter.ai/models)

## 🛠️ Available Tools

- **ask_ai** - Ask any AI model a question
- **ai_code_review** - Get code reviews from any model
- **ai_brainstorm** - Brainstorm ideas with any model
- **list_models** - See all available models
- **server_info** - Check server status

## 🔧 Configuration

### Environment Variables
- `OPENROUTER_API_KEY` - Your OpenRouter API key (required)
- `OPENROUTER_MODEL` - Default model to use (optional, defaults to gemini-2.0-flash)

### Using .env file
Copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
# Edit .env with your API key
```

## 🧪 Testing

Test the server directly:
```bash
# Check if it works
python3 server.py < test_request.json
```

Or test in Claude Code:
```bash
claude
# Then type: mcp__openrouter__server_info
```

## 🐛 Troubleshooting

**MCP not showing up?**
```bash
# Check if it's installed
claude mcp list

# Remove and re-add
claude mcp remove openrouter
claude mcp add openrouter python3 /full/path/to/server.py
```

**API key errors?**
- Ensure `OPENROUTER_API_KEY` is set in your environment
- Check your API key is valid at [OpenRouter](https://openrouter.ai/keys)

**Connection errors?**
- Ensure `requests` is installed: `pip install requests`
- Check your internet connection

## 🤝 Contributing

Pull requests welcome! Feel free to add more models or features.

## 📜 License

MIT - Use freely!

---

Made with ❤️ for the Claude Code community