# ollama_tui

A minimal terminal UI for chatting with local Ollama models, with streaming output.

Assistant responses are shown as plain text (Markdown rendering is disabled).

## Requirements
- Python 3.8+
- Ollama running locally (`ollama serve`)

## Run
```bash
python3 ollama_tui.py
```

Optionally set a different host:
```bash
OLLAMA_HOST=http://localhost:11434 python3 ollama_tui.py
```

## Keys
- Enter: send message
- Ctrl+Q: quit
- Ctrl+L: clear chat
- Up/Down or PageUp/PageDown: scroll

## Commands
- `/help` – show commands
- `/models` – list models
- `/model <name>` – select model
- `/clear` – clear chat
- `/system <text|clear>` – set per-chat system prompt (appended after default plain-text guidance)
- `/save [path]` – save transcript (defaults to `~/.ollama_tui/exports/`)
- `/retry` – resend last user message
- `/host <url>` – set Ollama host
- `/quit` – quit

## Tool Execution (Assistant-Initiated)
The assistant can request terminal commands using a JSON tool call. You will be prompted to approve every
command before it runs. A preview dialog shows the command and working directory. Output is streamed live
and sent back to the assistant automatically so it can continue.

Multiple tool calls can be included in a single assistant response; they will be executed in order.

Example:
```json
{"tool":"terminal.exec","command":"ls -la","cwd":"/path/to/project"}
```

## Persistence
Chats auto-save to `~/.ollama_tui/chats.json` and load on startup.
# ollama-TUI
