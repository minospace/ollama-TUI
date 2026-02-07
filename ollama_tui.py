#!/usr/bin/env python3
import curses
import json
import os
import re
import textwrap
import time
import urllib.error
import urllib.request
import subprocess
import selectors
import argparse


DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".ollama_tui")
DEFAULT_SESSION_PATH = os.path.join(DEFAULT_DATA_DIR, "chats.json")
DEFAULT_EXPORT_DIR = os.path.join(DEFAULT_DATA_DIR, "exports")
DEFAULT_SYSTEM_PROMPT = (
    "Respond in plain text. "
    "Do not use Markdown unless the user explicitly asks for it. "
    "If Markdown is requested, keep it minimal and only as required (e.g., fenced code blocks). "
    "Keep replies concise and direct."
)
TOOL_NAME = "terminal.exec"
TOOL_OUTPUT_PREFIX = "TOOL OUTPUT (terminal.exec):"
TOOL_DENIED_PREFIX = "TOOL DENIED (terminal.exec):"
TOOL_SYSTEM_PROMPT = (
    "Tool use: You can execute terminal commands via a tool. "
    "When the user asks you to inspect local files, list directories, check git status, "
    "or verify environment details, you should use the tool instead of guessing.\n"
    "To call the tool, reply with ONLY a JSON object (or a JSON array of objects) and no other text:\n"
    '{"tool":"terminal.exec","command":"<shell command>","cwd":"<optional working dir>"}\n'
    "If multiple commands are required, reply with a JSON array of those objects.\n"
    "The user must approve every command before it runs. After approval or denial, "
    "you will receive a system message starting with "
    '"TOOL OUTPUT (terminal.exec):" or "TOOL DENIED (terminal.exec):". '
    "Then continue your response normally."
)
TOOL_MAX_CALLS = 3
TOOL_TIMEOUT_SEC = 120

MARKDOWN_PREFIX = "| "

COMMANDS = [
    {
        "name": "/help",
        "aliases": ["/h"],
        "desc": "Show help",
        "takes_arg": False,
        "arg_hint": "",
    },
    {
        "name": "/models",
        "aliases": [],
        "desc": "List available models",
        "takes_arg": False,
        "arg_hint": "",
    },
    {
        "name": "/model",
        "aliases": [],
        "desc": "Set model",
        "takes_arg": True,
        "arg_hint": "<name>",
    },
    {
        "name": "/clear",
        "aliases": ["/c"],
        "desc": "Clear history",
        "takes_arg": False,
        "arg_hint": "",
    },
    {
        "name": "/chats",
        "aliases": [],
        "desc": "List chats",
        "takes_arg": False,
        "arg_hint": "",
    },
    {
        "name": "/chat",
        "aliases": [],
        "desc": "Manage chats",
        "takes_arg": True,
        "arg_hint": "<id|new|title|delete|cancel> [args]",
    },
    {
        "name": "/system",
        "aliases": [],
        "desc": "Set per-chat system prompt (appended)",
        "takes_arg": True,
        "arg_hint": "<text|clear>",
    },
    {
        "name": "/retry",
        "aliases": [],
        "desc": "Resend last user message",
        "takes_arg": False,
        "arg_hint": "",
    },
    {
        "name": "/host",
        "aliases": [],
        "desc": "Set Ollama host",
        "takes_arg": True,
        "arg_hint": "<url>",
    },
    {
        "name": "/save",
        "aliases": [],
        "desc": "Save chat to file",
        "takes_arg": False,
        "arg_hint": "[path]",
    },
    {
        "name": "/offload",
        "aliases": ["/stop"],
        "desc": "Offload model (ollama stop)",
        "takes_arg": True,
        "arg_hint": "[name]",
    },
    {
        "name": "/quit",
        "aliases": ["/q"],
        "desc": "Quit",
        "takes_arg": False,
        "arg_hint": "",
    },
]

def http_json(path, payload=None, timeout=30, host=None):
    base = host or DEFAULT_HOST
    url = base.rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def list_models(host=None):
    try:
        data = http_json("/api/tags", host=host)
        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        return models
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []


def offload_model(name, host=None):
    try:
        http_json(
            "/api/generate",
            payload={"model": name, "prompt": " ", "stream": False, "keep_alive": 0},
            host=host,
        )
        return True, ""
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        api_detail = str(exc)

    try:
        proc = subprocess.run(
            ["ollama", "stop", name],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, api_detail or "ollama CLI not found in PATH"
    if proc.returncode == 0:
        return True, ""
    detail = (proc.stderr or proc.stdout or "").strip()
    if detail:
        return False, detail
    return False, f"ollama stop failed (exit {proc.returncode})"


def load_model(name, host=None):
    try:
        http_json(
            "/api/generate",
            payload={"model": name, "prompt": " ", "stream": False},
            host=host,
        )
        return True, ""
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        return False, str(exc)


REASONING_FIELDS = ("reasoning", "thinking", "analysis", "thought", "thoughts")

STYLES = {
    "label": curses.A_BOLD,
    "heading": curses.A_BOLD,
    "code": curses.A_DIM,
    "code_inline": curses.A_DIM,
    "quote": curses.A_DIM,
    "rule": curses.A_DIM,
    "bold": curses.A_BOLD,
    "italic": curses.A_UNDERLINE,
    "link": curses.A_UNDERLINE,
    "bullet": 0,
}

INLINE_PATTERNS = [
    ("code", re.compile(r"`([^`]+)`")),
    ("link", re.compile(r"\[([^\]]+)\]\(([^)]+)\)")),
    ("bold", re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")),
    ("italic", re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_(?!_)")),
]

MARKDOWN_DETECT_RE = re.compile(
    r"(^\s*#{1,6}\s+\S)|(^\s*([-*+]|\d+[.)])\s+\S)|(^\s*> )|(```)|(\[[^\]]+\]\([^)]+\))|(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)",
    re.M,
)


def init_styles():
    if not curses.has_colors():
        return
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_GREEN, -1)
    curses.init_pair(4, curses.COLOR_BLUE, -1)
    STYLES["heading"] |= curses.color_pair(1)
    STYLES["code"] |= curses.color_pair(2)
    STYLES["code_inline"] |= curses.color_pair(2)
    STYLES["quote"] |= curses.color_pair(3)
    STYLES["label"] |= curses.color_pair(4)
    STYLES["link"] |= curses.color_pair(4)


def merge_segments(segments):
    merged = []
    for text, attr in segments:
        if not text:
            continue
        if merged and merged[-1][1] == attr:
            merged[-1] = (merged[-1][0] + text, attr)
        else:
            merged.append((text, attr))
    return merged


def wrap_segments(segments, width):
    if width <= 0:
        return [[("", 0)]]
    tokens = []
    for text, attr in segments:
        if not text:
            continue
        for token in re.findall(r"\s+|\S+", text):
            tokens.append((token, attr))
    lines = []
    current = []
    cur_len = 0
    for token, attr in tokens:
        if token.isspace() and cur_len == 0:
            continue
        token_len = len(token)
        if cur_len + token_len > width and cur_len > 0:
            lines.append(merge_segments(current))
            current = []
            cur_len = 0
            if token.isspace():
                continue
        current.append((token, attr))
        cur_len += token_len
    if current or not lines:
        lines.append(merge_segments(current))
    return lines


def parse_inline(text):
    segments = []
    idx = 0
    while idx < len(text):
        next_match = None
        next_kind = None
        next_start = None
        next_end = None
        for kind, pattern in INLINE_PATTERNS:
            match = pattern.search(text, idx)
            if not match:
                continue
            if next_match is None or match.start() < next_start:
                next_match = match
                next_kind = kind
                next_start = match.start()
                next_end = match.end()
        if not next_match:
            segments.append((text[idx:], 0))
            break
        if next_start > idx:
            segments.append((text[idx:next_start], 0))
        if next_kind == "code":
            segments.append((next_match.group(1), STYLES["code_inline"]))
        elif next_kind == "link":
            label = next_match.group(1)
            url = next_match.group(2)
            segments.append((label, STYLES["link"]))
            segments.append((f" ({url})", STYLES["code_inline"]))
        elif next_kind == "bold":
            value = next_match.group(1) or next_match.group(2) or ""
            segments.append((value, STYLES["bold"]))
        elif next_kind == "italic":
            value = next_match.group(1) or next_match.group(2) or ""
            segments.append((value, STYLES["italic"]))
        idx = next_end
    return merge_segments(segments)


def is_markdown(text):
    # Markdown rendering disabled; always treat as plain text.
    return False


def parse_markdown_blocks(text):
    blocks = []
    para = []
    in_code = False
    code_fence = None
    code_lines = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if in_code:
            if stripped.startswith(code_fence):
                blocks.append(("code", code_lines))
                code_lines = []
                in_code = False
                code_fence = None
            else:
                code_lines.append(line)
            continue
        if stripped.startswith(("```", "~~~")):
            if para:
                blocks.append(("paragraph", " ".join(para).strip()))
                para = []
            in_code = True
            code_fence = stripped[:3]
            continue
        if not stripped:
            if para:
                blocks.append(("paragraph", " ".join(para).strip()))
                para = []
            blocks.append(("blank", ""))
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            if para:
                blocks.append(("paragraph", " ".join(para).strip()))
                para = []
            blocks.append(("heading", heading_match.group(2).strip(), len(heading_match.group(1))))
            continue
        if re.match(r"^(-{3,}|_{3,}|\*{3,})$", stripped):
            if para:
                blocks.append(("paragraph", " ".join(para).strip()))
                para = []
            blocks.append(("rule", ""))
            continue
        if stripped.startswith(">"):
            if para:
                blocks.append(("paragraph", " ".join(para).strip()))
                para = []
            quote_text = stripped[1:].lstrip()
            if blocks and blocks[-1][0] == "quote":
                blocks[-1] = ("quote", (blocks[-1][1] + " " + quote_text).strip())
            else:
                blocks.append(("quote", quote_text))
            continue
        list_match = re.match(r"^([-*+]|\d+[.)])\s+(.*)$", stripped)
        if list_match:
            if para:
                blocks.append(("paragraph", " ".join(para).strip()))
                para = []
            bullet = list_match.group(1)
            item_text = list_match.group(2).strip()
            blocks.append(("list", bullet, item_text))
            continue
        if blocks and blocks[-1][0] == "list" and line.startswith(("  ", "\t")):
            bullet, item_text = blocks[-1][1], blocks[-1][2]
            blocks[-1] = ("list", bullet, f"{item_text} {stripped}".strip())
            continue
        para.append(stripped)
    if para:
        blocks.append(("paragraph", " ".join(para).strip()))
    if in_code:
        blocks.append(("code", code_lines))
    return blocks


def wrap_code_line(line, width):
    if width <= 0:
        return [""]
    if not line:
        return [""]
    return [line[i : i + width] for i in range(0, len(line), width)]


def render_markdown(content, width):
    lines = []
    blocks = parse_markdown_blocks(content)
    for block in blocks:
        kind = block[0]
        if kind == "blank":
            lines.append([])
            continue
        if kind == "heading":
            text = block[1]
            segments = [(text, STYLES["heading"])]
            lines.extend(wrap_segments(segments, width))
            continue
        if kind == "paragraph":
            segments = parse_inline(block[1])
            lines.extend(wrap_segments(segments, width))
            continue
        if kind == "list":
            bullet = block[1]
            text = block[2]
            bullet_text = f"{bullet} "
            body_width = max(1, width - len(bullet_text))
            body_lines = wrap_segments(parse_inline(text), body_width)
            for idx, body_line in enumerate(body_lines):
                if idx == 0:
                    lines.append([(bullet_text, STYLES["bullet"])] + body_line)
                else:
                    lines.append([(" " * len(bullet_text), STYLES["bullet"])] + body_line)
            continue
        if kind == "quote":
            quote_prefix = "| "
            body_width = max(1, width - len(quote_prefix))
            body_lines = wrap_segments(parse_inline(block[1]), body_width)
            for body_line in body_lines:
                lines.append([(quote_prefix, STYLES["quote"])] + body_line)
            continue
        if kind == "code":
            for code_line in block[1]:
                for wrapped in wrap_code_line(code_line, width):
                    lines.append([(wrapped, STYLES["code"])])
            if not block[1]:
                lines.append([("", STYLES["code"])])
            continue
        if kind == "rule":
            rule_text = "-" * max(1, width)
            lines.append([(rule_text, STYLES["rule"])])
            continue
    return lines


def render_markdown_message(content, width):
    label_line = [(f"Assistant (Markdown):", STYLES["label"])]
    if width <= len(MARKDOWN_PREFIX) + 1:
        return [label_line]
    md_width = max(1, width - len(MARKDOWN_PREFIX))
    md_lines = render_markdown(content, md_width)
    prefixed = []
    for line in md_lines:
        prefixed.append([(MARKDOWN_PREFIX, STYLES["label"])] + line)
    return [label_line] + prefixed


def coerce_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.append(coerce_text(item))
        return "".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return coerce_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_message_parts(msg):
    parts = []
    if not isinstance(msg, dict):
        return parts

    content = msg.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                item_type = (item.get("type") or "").lower()
                item_text = item.get("text") or item.get("content") or item.get("value")
                if not item_text:
                    continue
                if item_type in REASONING_FIELDS:
                    parts.append(("reasoning", coerce_text(item_text)))
                else:
                    parts.append(("content", coerce_text(item_text)))
            else:
                parts.append(("content", coerce_text(item)))
    elif isinstance(content, dict):
        item_type = (content.get("type") or "").lower()
        item_text = content.get("text") or content.get("content") or content.get("value")
        if item_text:
            if item_type in REASONING_FIELDS:
                parts.append(("reasoning", coerce_text(item_text)))
            else:
                parts.append(("content", coerce_text(item_text)))
    elif content:
        parts.append(("content", coerce_text(content)))

    for key in REASONING_FIELDS:
        if key in msg:
            text = coerce_text(msg.get(key))
            if text:
                parts.append(("reasoning", text))
    return parts


def chat_stream(model, messages, host=None):
    payload = {"model": model, "messages": messages, "stream": True}
    base = host or DEFAULT_HOST
    url = base.rstrip("/") + "/api/chat"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            if not raw:
                continue
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in obj:
                raise RuntimeError(obj["error"])
            msg = obj.get("message", {})
            parts = extract_message_parts(msg)
            if not parts:
                for key in REASONING_FIELDS:
                    text = coerce_text(obj.get(key))
                    if text:
                        parts.append(("reasoning", text))
                response_text = coerce_text(obj.get("response"))
                if response_text:
                    parts.append(("content", response_text))
            for kind, text in parts:
                if text:
                    yield (kind, text)
            if obj.get("done"):
                break


def normalize_cwd(cwd):
    if not cwd:
        return None
    expanded = os.path.expanduser(cwd.strip())
    if not os.path.isabs(expanded):
        expanded = os.path.abspath(expanded)
    return expanded


def parse_tool_requests(text):
    if not text:
        return []
    stripped = text.strip()
    if not stripped:
        return []
    if TOOL_NAME not in stripped:
        return []

    def coerce_tool_obj(obj):
        if not isinstance(obj, dict):
            return None
        if obj.get("tool") != TOOL_NAME:
            return None
        command = obj.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        cwd = obj.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            cwd = None
        command = command.strip()
        cwd = normalize_cwd(cwd) if isinstance(cwd, str) and cwd.strip() else None
        return {"command": command, "cwd": cwd}

    decoder = json.JSONDecoder()
    idx = 0
    found = []
    payload = stripped
    while True:
        brace = payload.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(payload[brace:])
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        tool = coerce_tool_obj(obj)
        if tool:
            found.append(tool)
        idx = brace + max(end, 1)
    return found


def format_tool_request(command, cwd=None):
    if cwd:
        return f"Tool request ({TOOL_NAME}): {command} (cwd: {cwd})"
    return f"Tool request ({TOOL_NAME}): {command}"


def format_tool_denied(command):
    return f"{TOOL_DENIED_PREFIX}\n$ {command}\nreason: user denied"


def format_tool_output_status(command, status, stdout, stderr):
    lines = [TOOL_OUTPUT_PREFIX, f"$ {command}", status]
    if stdout:
        lines.append("stdout:")
        lines.append(stdout.rstrip("\n"))
    if stderr:
        lines.append("stderr:")
        lines.append(stderr.rstrip("\n"))
    return "\n".join(lines).strip()


def run_terminal_command_stream(command, cwd=None, on_update=None):
    if cwd and not os.path.isdir(cwd):
        return None, "", "", f"cwd not found: {cwd}"
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            cwd=cwd,
        )
    except Exception as exc:
        return None, "", "", str(exc)

    stdout_parts = []
    stderr_parts = []
    selector = selectors.DefaultSelector()
    if proc.stdout:
        selector.register(proc.stdout, selectors.EVENT_READ)
    if proc.stderr:
        selector.register(proc.stderr, selectors.EVENT_READ)

    if on_update:
        on_update("".join(stdout_parts), "".join(stderr_parts), "running...")

    start_time = time.time()
    while True:
        if time.time() - start_time > TOOL_TIMEOUT_SEC:
            proc.kill()
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
            selector.close()
            return None, "".join(stdout_parts), "".join(stderr_parts), f"command timed out after {TOOL_TIMEOUT_SEC}s"

        if proc.poll() is not None and not selector.get_map():
            break

        events = selector.select(timeout=0.1)
        for key, _ in events:
            try:
                data = os.read(key.fileobj.fileno(), 4096)
            except Exception:
                data = b""
            if data:
                text = data.decode("utf-8", "replace")
                if key.fileobj is proc.stdout:
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            else:
                try:
                    selector.unregister(key.fileobj)
                except KeyError:
                    pass
                try:
                    key.fileobj.close()
                except Exception:
                    pass
        if events and on_update:
            on_update("".join(stdout_parts), "".join(stderr_parts), "running...")

    returncode = proc.wait()
    return returncode, "".join(stdout_parts), "".join(stderr_parts), ""


def prompt_tool_approval(
    stdscr,
    history,
    input_buf,
    cursor_idx,
    model,
    chat_title,
    scroll,
    suggestions,
    suggest_idx,
    command,
    cwd=None,
):
    cwd_display = cwd or os.getcwd()
    draw(
        stdscr,
        history,
        input_buf,
        cursor_idx,
        "",
        model,
        chat_title,
        scroll,
        suggestions,
        suggest_idx,
    )
    modal_ok = draw_approval_dialog(stdscr, command, cwd_display)
    if not modal_ok:
        status_msg = f"Approve command? (y/n) {command}"
        draw(
            stdscr,
            history,
            input_buf,
            cursor_idx,
            status_msg,
            model,
            chat_title,
            scroll,
            suggestions,
            suggest_idx,
        )
    while True:
        ch = stdscr.getch()
        if ch in (ord("y"), ord("Y")):
            return True
        if ch in (ord("n"), ord("N"), 27):
            return False


def stream_assistant_reply(
    stdscr,
    history,
    messages,
    model,
    host,
    chat_title,
    input_buf,
    cursor_idx,
    scroll,
    suggestions,
    suggest_idx,
    system_prompt,
):
    history.append(("assistant", ""))
    assistant_idx = len(history) - 1
    reasoning_idx = None
    reasoning_text = ""
    status_msg = "streaming..."
    last_status_time = time.time()
    scroll = 0
    draw(
        stdscr,
        history,
        input_buf,
        cursor_idx,
        status_msg,
        model,
        chat_title,
        scroll,
        suggestions,
        suggest_idx,
    )

    try:
        reply = ""
        if system_prompt:
            chat_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            chat_messages = messages
        for kind, chunk in chat_stream(model, chat_messages, host):
            if kind == "reasoning":
                if reasoning_idx is None:
                    history.insert(assistant_idx, ("reasoning", ""))
                    reasoning_idx = assistant_idx
                    assistant_idx += 1
                reasoning_text += chunk
                history[reasoning_idx] = ("reasoning", reasoning_text)
            else:
                reply += chunk
                history[assistant_idx] = ("assistant", reply)
            draw(
                stdscr,
                history,
                input_buf,
                cursor_idx,
                status_msg,
                model,
                chat_title,
                scroll,
                suggestions,
                suggest_idx,
            )
        messages.append({"role": "assistant", "content": reply})
        if reasoning_idx is not None and not reasoning_text:
            history.pop(reasoning_idx)
        status_msg = ""
        return reply, status_msg, last_status_time, scroll, None
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
        RuntimeError,
    ) as exc:
        if reply:
            history[assistant_idx] = ("assistant", reply)
        else:
            history.pop()
        history.append(("system", f"error: {exc}"))
        status_msg = "request failed"
        last_status_time = time.time()
        return reply, status_msg, last_status_time, scroll, exc


def wrap_plain_message(role, content, width):
    prefix = {
        "user": "You: ",
        "assistant": "Assistant: ",
        "system": "System: ",
        "reasoning": "Reasoning: ",
    }.get(role, "")

    if width <= len(prefix) + 1:
        return [[(prefix.rstrip(), 0)]]

    first_width = width - len(prefix)
    indent = " " * len(prefix)
    out = []
    raw_lines = content.splitlines() or [""]
    for line_idx, raw_line in enumerate(raw_lines):
        lines = textwrap.wrap(raw_line, width=first_width) or [""]
        for wrapped_idx, line in enumerate(lines):
            if line_idx == 0 and wrapped_idx == 0:
                out.append([(prefix + line, 0)])
            else:
                out.append([(indent + line, 0)])
    return out


def render_history(history, width):
    lines = []
    for role, content in history:
        if role == "assistant" and is_markdown(content):
            lines.extend(render_markdown_message(content, width))
        else:
            lines.extend(wrap_plain_message(role, content, width))
        lines.append([])
    if lines:
        lines.pop()
    return lines


def save_history(path, history):
    with open(path, "w", encoding="utf-8") as f:
        for role, content in history:
            label = role.upper()
            f.write(f"[{label}]\n{content}\n\n")


def format_command_usage(cmd):
    usage = cmd["name"]
    if cmd["takes_arg"]:
        usage += f" {cmd['arg_hint']}"
    return usage


def build_help_text():
    items = []
    for cmd in COMMANDS:
        usage = format_command_usage(cmd)
        if cmd["aliases"]:
            usage += f" ({', '.join(cmd['aliases'])})"
        items.append(usage)
    return "Commands: " + ", ".join(items)


def build_system_prompt(user_prompt):
    base_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{TOOL_SYSTEM_PROMPT}"
    if user_prompt:
        return f"{base_prompt}\n\n{user_prompt}"
    return base_prompt


def get_command_suggestions(input_buf):
    if not input_buf.startswith("/"):
        return []
    if " " in input_buf:
        return []
    prefix = input_buf.lower()
    matches = []
    for cmd in COMMANDS:
        triggers = [cmd["name"]] + cmd["aliases"]
        if prefix == "/" or any(t.startswith(prefix) for t in triggers):
            matches.append(cmd)
    return matches


def draw(
    stdscr,
    history,
    input_buf,
    cursor_idx,
    status_msg,
    model,
    chat_title,
    scroll,
    suggestions,
    suggest_idx,
):
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    history_height = max(1, height - 2)

    lines = render_history(history, width)
    max_scroll = max(0, len(lines) - history_height)
    scroll = min(scroll, max_scroll)

    start = max(0, len(lines) - history_height - scroll)
    end = start + history_height
    view = lines[start:end]

    for i, line in enumerate(view):
        x = 0
        for text, attr in line:
            if x >= width - 1:
                break
            remaining = width - 1 - x
            if remaining <= 0:
                break
            stdscr.addnstr(i, x, text, remaining, attr)
            x += min(len(text), remaining)

    if suggestions:
        menu_height = min(len(suggestions), max(0, height - 2))
        menu_start = max(0, height - 2 - menu_height)
        for i in range(menu_height):
            cmd = suggestions[i]
            alias = f" ({', '.join(cmd['aliases'])})" if cmd["aliases"] else ""
            label = f"{format_command_usage(cmd):<18} {cmd['desc']}{alias}"
            attr = curses.A_REVERSE if i == suggest_idx else curses.A_NORMAL
            stdscr.addnstr(menu_start + i, 0, label.ljust(width), width - 1, attr)

    status = status_msg or ""
    status_base = f"Chat: {chat_title}  |  Model: {model}  |  /help for commands"
    if status:
        status = f"{status_base}  |  {status}"
    else:
        status = status_base
    stdscr.addnstr(height - 2, 0, status.ljust(width), width - 1, curses.A_REVERSE)

    prompt = "> "
    max_input = max(0, width - len(prompt) - 1)
    if len(input_buf) <= max_input:
        view_start = 0
        visible = input_buf
        cursor_x = len(prompt) + cursor_idx
    else:
        view_start = max(0, min(cursor_idx - max_input + 1, len(input_buf) - max_input))
        visible = input_buf[view_start : view_start + max_input]
        cursor_x = len(prompt) + (cursor_idx - view_start)
    stdscr.addnstr(height - 1, 0, prompt + visible, width - 1)
    stdscr.move(height - 1, min(cursor_x, width - 1))
    stdscr.refresh()
    return scroll


def draw_modal(stdscr, lines, title=None):
    height, width = stdscr.getmaxyx()
    if height < 6 or width < 24:
        return False

    content = []
    if title:
        content.append(title)
    for line in lines:
        if line is None:
            continue
        content.append(str(line))

    max_inner_width = max(10, width - 6)
    wrapped = []
    for line in content:
        wrapped.extend(textwrap.wrap(line, width=max_inner_width) or [""])

    inner_width = max(len(line) for line in wrapped) if wrapped else 10
    inner_width = min(inner_width, max_inner_width)
    inner_height = max(1, len(wrapped))
    box_width = inner_width + 2
    box_height = inner_height + 2

    top = max(0, (height - box_height) // 2)
    left = max(0, (width - box_width) // 2)

    try:
        stdscr.addnstr(top, left, "+" + "-" * inner_width + "+", box_width)
        for i in range(inner_height):
            stdscr.addnstr(top + 1 + i, left, "|", 1)
            line = wrapped[i] if i < len(wrapped) else ""
            stdscr.addnstr(top + 1 + i, left + 1, line.ljust(inner_width), inner_width)
            stdscr.addnstr(top + 1 + i, left + box_width - 1, "|", 1)
        stdscr.addnstr(top + box_height - 1, left, "+" + "-" * inner_width + "+", box_width)
        stdscr.refresh()
    except curses.error:
        return False
    return True


def draw_approval_dialog(stdscr, command, cwd):
    height, width = stdscr.getmaxyx()
    if height < 10 or width < 40:
        return False

    max_inner_width = max(24, min(width - 10, 100))
    lines = []

    lines.append([("Command:", curses.A_BOLD)])
    for line in textwrap.wrap(f"$ {command}", width=max_inner_width):
        lines.append([(line, curses.A_DIM)])
    lines.append([("", 0)])
    lines.append([("Working directory:", curses.A_BOLD)])
    for line in textwrap.wrap(cwd, width=max_inner_width):
        lines.append([(line, 0)])
    lines.append([("", 0)])
    lines.append(
        [
            ("Press ", 0),
            ("Y", curses.A_REVERSE),
            (" to approve, ", 0),
            ("N", curses.A_REVERSE),
            (" or ", 0),
            ("Esc", curses.A_REVERSE),
            (" to deny.", 0),
        ]
    )

    def line_len(segments):
        return sum(len(text) for text, _ in segments)

    inner_width = min(
        max_inner_width,
        max(line_len(line) for line in lines) if lines else max_inner_width,
    )
    max_inner_height = max(1, height - 6)
    if len(lines) > max_inner_height:
        lines = lines[: max_inner_height - 1] + [[("...", 0)]]

    inner_height = max(1, len(lines))
    box_width = inner_width + 2
    box_height = inner_height + 2

    top = max(0, (height - box_height) // 2)
    left = max(0, (width - box_width) // 2)

    title = " Terminal Command Approval "
    title = title[: inner_width]
    pad = max(0, inner_width - len(title))
    left_pad = pad // 2
    right_pad = pad - left_pad
    top_border = "+" + ("-" * left_pad) + title + ("-" * right_pad) + "+"

    try:
        stdscr.addnstr(top, left, top_border, box_width, curses.A_BOLD)
        for i in range(inner_height):
            stdscr.addnstr(top + 1 + i, left, "|", 1)
            x = left + 1
            remaining = inner_width
            for text, attr in lines[i]:
                if remaining <= 0:
                    break
                chunk = text[:remaining]
                stdscr.addnstr(top + 1 + i, x, chunk, remaining, attr)
                x += len(chunk)
                remaining -= len(chunk)
            if remaining > 0:
                stdscr.addnstr(top + 1 + i, x, " " * remaining, remaining)
            stdscr.addnstr(top + 1 + i, left + box_width - 1, "|", 1)
        stdscr.addnstr(top + box_height - 1, left, "+" + "-" * inner_width + "+", box_width)
        stdscr.refresh()
    except curses.error:
        return False
    return True


def make_chat(chat_id, title=None):
    if not title:
        title = f"Chat {chat_id}"
    return {"id": chat_id, "title": title, "history": [], "messages": [], "system": ""}


def ensure_data_dir():
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)


def ensure_export_dir():
    ensure_data_dir()
    os.makedirs(DEFAULT_EXPORT_DIR, exist_ok=True)


def sanitize_filename(value, fallback="chat"):
    safe = []
    for ch in value.strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        elif ch.isspace():
            safe.append("-")
    name = "".join(safe).strip("-._")
    if not name:
        return fallback
    return name[:80]


def default_export_path(chat_title):
    ensure_export_dir()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = sanitize_filename(chat_title or "chat")
    filename = f"{base}-{stamp}.txt"
    path = os.path.join(DEFAULT_EXPORT_DIR, filename)
    if not os.path.exists(path):
        return path
    for idx in range(2, 1000):
        candidate = os.path.join(DEFAULT_EXPORT_DIR, f"{base}-{stamp}-{idx}.txt")
        if not os.path.exists(candidate):
            return candidate
    return path


def serialize_session(chats, active_idx, next_chat_id, model, host):
    return {
        "version": 1,
        "active_chat_id": chats[active_idx]["id"] if chats else None,
        "next_chat_id": next_chat_id,
        "model": model,
        "host": host,
        "chats": chats,
    }


def load_session(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    chats = data.get("chats")
    if not isinstance(chats, list) or not chats:
        return None
    for chat in chats:
        if "history" not in chat or "messages" not in chat:
            return None
        if "system" not in chat:
            chat["system"] = ""
    return data


def save_session(path, chats, active_idx, next_chat_id, model, host):
    ensure_data_dir()
    data = serialize_session(chats, active_idx, next_chat_id, model, host)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def find_chat_index(chats, chat_id):
    for idx, chat in enumerate(chats):
        if chat["id"] == chat_id:
            return idx
    return None


def format_chat_list(chats, active_idx):
    lines = ["Chats:"]
    for idx, chat in enumerate(chats):
        marker = "*" if idx == active_idx else " "
        lines.append(f"{marker} {chat['id']:>2}  {chat['title']}")
    return "\n".join(lines)


def handle_command(cmd, chats, active_idx, models, model, host, next_chat_id, pending_delete_id):
    raw = cmd.strip()
    parts = raw.split()
    if not parts:
        return model, host, models, "", False, active_idx, next_chat_id, False
    name = parts[0].lower()
    current = chats[active_idx]

    if name in ("/quit", "/q"):
        return model, host, models, "quit", True, active_idx, next_chat_id, False, None
    if name in ("/clear", "/c"):
        current["history"].clear()
        current["messages"].clear()
        return model, host, models, "cleared", False, active_idx, next_chat_id, True, None
    if name in ("/help", "/h"):
        help_text = build_help_text()
        current["history"].append(("system", help_text))
        return model, host, models, "", False, active_idx, next_chat_id, True, None
    if name in ("/models",):
        models = list_models(host)
        if models:
            current["history"].append(("system", "Available models: " + ", ".join(models)))
            return model, host, models, "", False, active_idx, next_chat_id, True, None
        current["history"].append(("system", "No models found (is ollama running?)"))
        return model, host, models, "", False, active_idx, next_chat_id, True, None
    if name in ("/model",):
        if len(parts) < 2:
            return model, host, models, "usage: /model <name>", False, active_idx, next_chat_id, False, None
        candidate = parts[1]
        models = list_models(host)
        if candidate in models or not models:
            return candidate, host, models, f"model set to {candidate}", False, active_idx, next_chat_id, True, None
        return model, host, models, f"model not found: {candidate}", False, active_idx, next_chat_id, False, None
    if name in ("/chats",):
        current["history"].append(("system", format_chat_list(chats, active_idx)))
        return model, host, models, "", False, active_idx, next_chat_id, True, None
    if name in ("/chat",):
        args = raw[len(parts[0]) :].strip()
        if not args:
            return (
                model,
                host,
                models,
                "usage: /chat <id|new|title|delete|cancel> [args]",
                False,
                active_idx,
                next_chat_id,
                False,
                None,
            )
        tokens = args.split()
        action = tokens[0].lower()
        if action in ("new", "n"):
            title = args[len(tokens[0]) :].strip()
            chat = make_chat(next_chat_id, title if title else None)
            chats.append(chat)
            active_idx = len(chats) - 1
            next_chat_id += 1
            return model, host, models, f"switched to chat {chat['id']}", False, active_idx, next_chat_id, True, None
        if action in ("cancel", "can"):
            return model, host, models, "action cancelled", False, active_idx, next_chat_id, False, None
        if action in ("delete", "del", "rm"):
            if len(tokens) >= 2 and tokens[1].isdigit():
                chat_id = int(tokens[1])
            else:
                chat_id = chats[active_idx]["id"]

            idx = find_chat_index(chats, chat_id)
            if idx is None:
                return model, host, models, f"chat not found: {chat_id}", False, active_idx, next_chat_id, False, None

            if pending_delete_id == chat_id:
                # Confirmed deletion
                chats.pop(idx)
                status = f"deleted chat {chat_id}"
                if not chats:
                    chat = make_chat(next_chat_id)
                    chats.append(chat)
                    active_idx = 0
                    next_chat_id += 1
                    status += f" (created chat {chat['id']})"
                else:
                    if idx == active_idx:
                        active_idx = min(idx, len(chats) - 1)
                    elif idx < active_idx:
                        active_idx -= 1
                return model, host, models, status, False, active_idx, next_chat_id, True, None
            else:
                # Request confirmation
                title = chats[idx]["title"]
                status = f"Delete chat '{title}' (ID {chat_id})? Run command again to confirm or /chat cancel."
                return model, host, models, status, False, active_idx, next_chat_id, False, chat_id
        if action in ("title", "rename"):
            remainder = args[len(tokens[0]) :].strip()
            if not remainder:
                return (
                    model,
                    host,
                    models,
                    "usage: /chat title [id] <new title>",
                    False,
                    active_idx,
                    next_chat_id,
                    False,
                    None,
                )
            remainder_tokens = remainder.split()
            if remainder_tokens and remainder_tokens[0].isdigit():
                chat_id = int(remainder_tokens[0])
                new_title = remainder[len(remainder_tokens[0]) :].strip()
            else:
                chat_id = chats[active_idx]["id"]
                new_title = remainder
            if not new_title:
                return (
                    model,
                    host,
                    models,
                    "usage: /chat title [id] <new title>",
                    False,
                    active_idx,
                    next_chat_id,
                    False,
                    None,
                )
            idx = find_chat_index(chats, chat_id)
            if idx is None:
                return model, host, models, f"chat not found: {chat_id}", False, active_idx, next_chat_id, False, None
            chats[idx]["title"] = new_title
            return (
                model,
                host,
                models,
                f"renamed chat {chat_id}",
                False,
                active_idx,
                next_chat_id,
                True,
                None,
            )
        if action.isdigit():
            chat_id = int(action)
            idx = find_chat_index(chats, chat_id)
            if idx is None:
                return model, host, models, f"chat not found: {chat_id}", False, active_idx, next_chat_id, False, None
            active_idx = idx
            return model, host, models, f"switched to chat {chat_id}", False, active_idx, next_chat_id, True, None
        return (
            model,
            host,
            models,
            "usage: /chat <id|new|title|delete|cancel> [args]",
            False,
            active_idx,
            next_chat_id,
            False,
            None,
        )
    if name in ("/system",):
        if len(parts) < 2:
            return model, host, models, "usage: /system <text|clear>", False, active_idx, next_chat_id, False, None
        text = raw[len(parts[0]) :].strip()
        if text.lower() in ("clear", "off", "none"):
            current["system"] = ""
            return model, host, models, "system prompt cleared", False, active_idx, next_chat_id, True, None
        current["system"] = text
        return model, host, models, "system prompt set", False, active_idx, next_chat_id, True, None
    if name in ("/retry",):
        return model, host, models, "retry", False, active_idx, next_chat_id, False, None
    if name in ("/host",):
        if len(parts) < 2:
            return model, host, models, "usage: /host <url>", False, active_idx, next_chat_id, False, None
        host = parts[1]
        return model, host, models, f"host set to {host}", False, active_idx, next_chat_id, True, None
    if name in ("/save",):
        path = None
        if len(parts) >= 2:
            path = parts[1]
        if not path:
            path = default_export_path(current.get("title") or "chat")
        try:
            save_history(path, current["history"])
            return model, host, models, f"saved to {path}", False, active_idx, next_chat_id, False, None
        except OSError as exc:
            return model, host, models, f"save failed: {exc}", False, active_idx, next_chat_id, False, None
    if name in ("/offload", "/stop"):
        target = parts[1] if len(parts) >= 2 else model
        ok, detail = offload_model(target, host)
        if ok:
            return model, host, models, f"offloaded model {target}", False, active_idx, next_chat_id, False, None
        msg = f"offload failed: {detail}" if detail else "offload failed"
        return model, host, models, msg, False, active_idx, next_chat_id, False, None

    return model, host, models, f"unknown command: {name}", False, active_idx, next_chat_id, False, None


def main(stdscr, cli_model=None):
    curses.curs_set(1)
    stdscr.nodelay(False)
    stdscr.keypad(True)
    init_styles()

    session = load_session(DEFAULT_SESSION_PATH)
    host = DEFAULT_HOST
    models = list_models(host)
    model = models[0] if models else "llama3"

    chats = [make_chat(1)]
    active_idx = 0
    next_chat_id = 2
    if session:
        chats = session.get("chats", chats) or chats
        active_id = session.get("active_chat_id")
        if active_id is not None:
            found_idx = find_chat_index(chats, active_id)
            if found_idx is not None:
                active_idx = found_idx
        next_chat_id = session.get("next_chat_id", next_chat_id)
        model = session.get("model", model)
        host = session.get("host", host)
    if cli_model:
        model = cli_model
    input_buf = ""
    cursor_idx = 0
    status_msg = ""
    scroll = 0
    last_status_time = 0
    suggestions = []
    prev_suggestions = []
    suggest_idx = 0
    input_history = []
    input_hist_idx = None
    last_user_text = ""
    pending_delete_id = None

    current = chats[active_idx]
    history = current["history"]
    chat_title = current["title"]
    status_msg = f"loading model {model}..."
    last_status_time = time.time()
    scroll = draw(
        stdscr,
        history,
        input_buf,
        cursor_idx,
        status_msg,
        model,
        chat_title,
        scroll,
        suggestions,
        suggest_idx,
    )
    ok, detail = load_model(model, host)
    if ok:
        status_msg = ""
    else:
        status_msg = f"load failed: {detail}" if detail else "load failed"
        last_status_time = time.time()

    while True:
        suggestions = get_command_suggestions(input_buf)
        if [cmd["name"] for cmd in suggestions] != [cmd["name"] for cmd in prev_suggestions]:
            suggest_idx = 0
        if suggestions:
            suggest_idx = min(suggest_idx, len(suggestions) - 1)
        else:
            suggest_idx = 0
        prev_suggestions = suggestions

        current = chats[active_idx]
        history = current["history"]
        messages = current["messages"]
        chat_title = current["title"]

        scroll = draw(
            stdscr,
            history,
            input_buf,
            cursor_idx,
            status_msg,
            model,
            chat_title,
            scroll,
            suggestions,
            suggest_idx,
        )

        if status_msg and time.time() - last_status_time > 3:
            status_msg = ""

        ch = stdscr.getch()

        if ch == -1:
            continue

        if ch in (curses.KEY_RESIZE,):
            continue

        if ch in (3, 17):  # Ctrl+C or Ctrl+Q
            break
        if ch == 12:  # Ctrl+L
            history.clear()
            messages.clear()
            scroll = 0
            continue

        if ch in (curses.KEY_PPAGE,):
            scroll = min(scroll + 5, 10_000)
            continue
        if ch in (curses.KEY_NPAGE,):
            scroll = max(scroll - 5, 0)
            continue

        if ch in (curses.KEY_UP,):
            if suggestions:
                suggest_idx = (suggest_idx - 1) % len(suggestions)
            else:
                if input_history:
                    if input_hist_idx is None:
                        input_hist_idx = len(input_history) - 1
                    else:
                        input_hist_idx = max(0, input_hist_idx - 1)
                    input_buf = input_history[input_hist_idx]
                    cursor_idx = len(input_buf)
                else:
                    scroll = min(scroll + 1, 10_000)
            continue
        if ch in (curses.KEY_DOWN,):
            if suggestions:
                suggest_idx = (suggest_idx + 1) % len(suggestions)
            else:
                if input_hist_idx is None:
                    scroll = max(scroll - 1, 0)
                else:
                    input_hist_idx += 1
                    if input_hist_idx >= len(input_history):
                        input_hist_idx = None
                        input_buf = ""
                    else:
                        input_buf = input_history[input_hist_idx]
                    cursor_idx = len(input_buf)
            continue

        if ch in (curses.KEY_LEFT,):
            cursor_idx = max(0, cursor_idx - 1)
            continue
        if ch in (curses.KEY_RIGHT,):
            cursor_idx = min(len(input_buf), cursor_idx + 1)
            continue
        if ch in (curses.KEY_HOME,):
            cursor_idx = 0
            continue
        if ch in (curses.KEY_END,):
            cursor_idx = len(input_buf)
            continue

        if ch in (curses.KEY_BACKSPACE, 127, 8):
            if cursor_idx > 0:
                input_buf = input_buf[: cursor_idx - 1] + input_buf[cursor_idx:]
                cursor_idx -= 1
            continue
        if ch == curses.KEY_DC:
            if cursor_idx < len(input_buf):
                input_buf = input_buf[:cursor_idx] + input_buf[cursor_idx + 1 :]
            continue

        if ch == 9:  # Tab
            if suggestions:
                chosen = suggestions[suggest_idx]
                input_buf = chosen["name"]
                if chosen["takes_arg"]:
                    input_buf += " "
                cursor_idx = len(input_buf)
            continue

        if ch in (10, 13):  # Enter
            text = input_buf.strip()
            input_buf = ""
            cursor_idx = 0
            input_hist_idx = None
            if not text:
                continue

            if text.startswith("/"):
                prev_active_idx = active_idx
                prev_model = model
                (
                    model,
                    host,
                    models,
                    status_msg,
                    should_quit,
                    active_idx,
                    next_chat_id,
                    should_save,
                    pending_delete_id,
                ) = handle_command(
                    text,
                    chats,
                    active_idx,
                    models,
                    model,
                    host,
                    next_chat_id,
                    pending_delete_id,
                )
                if status_msg == "retry":
                    if last_user_text:
                        text = last_user_text
                        status_msg = ""
                    else:
                        status_msg = "nothing to retry"
                        last_status_time = time.time()
                        continue
                else:
                    if should_save:
                        save_session(DEFAULT_SESSION_PATH, chats, active_idx, next_chat_id, model, host)
                if status_msg:
                    last_status_time = time.time()
                if active_idx != prev_active_idx:
                    scroll = 0
                if model != prev_model:
                    current = chats[active_idx]
                    history = current["history"]
                    chat_title = current["title"]
                    offload_ok = True
                    offload_detail = ""
                    if prev_model:
                        status_msg = f"offloading model {prev_model}..."
                        last_status_time = time.time()
                        scroll = draw(
                            stdscr,
                            history,
                            input_buf,
                            cursor_idx,
                            status_msg,
                            model,
                            chat_title,
                            scroll,
                            suggestions,
                            suggest_idx,
                        )
                        offload_ok, offload_detail = offload_model(prev_model, host)
                    if not offload_ok:
                        model = prev_model
                        status_msg = (
                            f"offload failed: {offload_detail}" if offload_detail else "offload failed"
                        )
                        last_status_time = time.time()
                        save_session(DEFAULT_SESSION_PATH, chats, active_idx, next_chat_id, model, host)
                    else:
                        status_msg = f"loading model {model}..."
                        last_status_time = time.time()
                        scroll = draw(
                            stdscr,
                            history,
                            input_buf,
                            cursor_idx,
                            status_msg,
                            model,
                            chat_title,
                            scroll,
                            suggestions,
                            suggest_idx,
                        )
                        ok, detail = load_model(model, host)
                        if ok:
                            status_msg = ""
                        else:
                            status_msg = f"load failed: {detail}" if detail else "load failed"
                        last_status_time = time.time()
                if should_quit:
                    break
                continue

            history.append(("user", text))
            messages.append({"role": "user", "content": text})
            last_user_text = text
            input_history.append(text)
            if len(input_history) > 200:
                input_history = input_history[-200:]
            system_prompt = build_system_prompt(current.get("system") or "")
            reply, status_msg, last_status_time, scroll, err = stream_assistant_reply(
                stdscr,
                history,
                messages,
                model,
                host,
                chat_title,
                input_buf,
                cursor_idx,
                scroll,
                suggestions,
                suggest_idx,
                system_prompt,
            )
            if err:
                continue
            save_session(DEFAULT_SESSION_PATH, chats, active_idx, next_chat_id, model, host)

            tool_calls = 0
            tool_requests = parse_tool_requests(reply)
            while tool_requests and tool_calls < TOOL_MAX_CALLS:
                if messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == reply:
                    messages.pop()
                if history and history[-1][0] == "assistant":
                    history.pop()

                for tool_request in tool_requests:
                    if tool_calls >= TOOL_MAX_CALLS:
                        break
                    tool_calls += 1
                    command = tool_request["command"]
                    cwd = tool_request["cwd"]
                    history.append(("system", format_tool_request(command, cwd)))
                    approved = prompt_tool_approval(
                        stdscr,
                        history,
                        input_buf,
                        cursor_idx,
                        model,
                        chat_title,
                        scroll,
                        suggestions,
                        suggest_idx,
                        command,
                        cwd,
                    )
                    if approved:
                        scroll = 0
                        tool_output_idx = len(history)
                        history.append(("system", format_tool_output_status(command, "running...", "", "")))

                        def on_update(stdout_text, stderr_text, status):
                            history[tool_output_idx] = (
                                "system",
                                format_tool_output_status(command, status, stdout_text, stderr_text),
                            )
                            draw(
                                stdscr,
                                history,
                                input_buf,
                                cursor_idx,
                                "running command...",
                                model,
                                chat_title,
                                scroll,
                                suggestions,
                                suggest_idx,
                            )

                        returncode, stdout, stderr, error = run_terminal_command_stream(
                            command,
                            cwd,
                            on_update=on_update,
                        )
                        if error:
                            status = f"error: {error}"
                        else:
                            status = f"exit: {returncode}"
                        tool_msg = format_tool_output_status(command, status, stdout, stderr)
                        history[tool_output_idx] = ("system", tool_msg)
                    else:
                        tool_msg = format_tool_denied(command)
                        history.append(("system", tool_msg))

                    messages.append({"role": "system", "content": tool_msg})

                if tool_calls >= TOOL_MAX_CALLS:
                    break

                reply, status_msg, last_status_time, scroll, err = stream_assistant_reply(
                    stdscr,
                    history,
                    messages,
                    model,
                    host,
                    chat_title,
                    input_buf,
                    cursor_idx,
                    scroll,
                    suggestions,
                    suggest_idx,
                    system_prompt,
                )
                if err:
                    break
                save_session(DEFAULT_SESSION_PATH, chats, active_idx, next_chat_id, model, host)
                tool_requests = parse_tool_requests(reply)
            if tool_requests and tool_calls >= TOOL_MAX_CALLS:
                history.append(("system", f"Tool call limit reached ({TOOL_MAX_CALLS})."))
                status_msg = "tool call limit reached"
                last_status_time = time.time()
            continue

        if 32 <= ch <= 126:
            input_buf = input_buf[:cursor_idx] + chr(ch) + input_buf[cursor_idx:]
            cursor_idx += 1
            continue


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Terminal UI for chatting with Ollama models")
        parser.add_argument("--model", help="Model name to load on startup")
        args = parser.parse_args()
        curses.wrapper(main, args.model)
    except KeyboardInterrupt:
        pass
