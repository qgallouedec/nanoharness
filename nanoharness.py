#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=0.34", "textual>=1.0"]
# ///
"""nanoharness — a coding agent in one file.

Four tools, a tiny system prompt, a terminal UI, and any model on Hugging Face
Inference Providers.

    uv run nanoharness.py

Assembled from what the good open harnesses already proved: the minimal tool
set and lazy skills from pi, the SKILL.md and AGENTS.md conventions shared by
Claude Code and dsh, an approval gate from Codex. README.md credits what came
from where.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from getpass import getpass
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient, get_token, whoami
from huggingface_hub.constants import HF_TOKEN_PATH
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.theme import Theme
from textual.widgets import Input, Static

DEFAULT_MODEL = "moonshotai/Kimi-K3"
HUB_SKILL_TAG = "agent-skill"
MAX_STEPS = 100  # tool rounds per turn; a runaway backstop, not a budget
READ_MAX_LINES, READ_MAX_BYTES = 2000, 100_000
BASH_MAX_LINES, BASH_TIMEOUT = 200, 120
CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md")
SKILL_DIRS = (".agent/skills", ".claude/skills")

# ---------------------------------------------------------------------------
# skills
#
# A skill is a directory with a SKILL.md whose frontmatter carries `name` and
# `description` -- the convention Claude Code and dsh share, and the one ~100
# repos on the Hub already publish under. pi's insight is that skills need no
# tool of their own: list name, description and path in the prompt, and let the
# model `read` the body when a task matches. One line of context per skill,
# full instructions only when they are actually relevant.
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    name: str
    description: str
    location: str


BLOCK_SCALARS = {"", ">", "|", ">-", "|-", ">+", "|+"}


def parse_frontmatter(raw: str) -> dict[str, str]:
    """Pull `---` delimited frontmatter off the top of a document.

    Enough YAML for a SKILL.md and no more: flat `key: value` pairs, plus the
    folded and literal block scalars (`description: >`) that real skills use
    for anything longer than a line. Blocks are folded to a single line, which
    is all a one-line routing description needs.
    """
    if not raw.startswith("---"):
        return {}
    end = raw.find("\n---", 3)
    if end == -1:
        return {}

    meta: dict[str, str] = {}
    key, block = None, []
    for line in raw[3:end].splitlines():
        if key is not None and (not line.strip() or line[:1] in " \t"):
            block.append(line.strip())
            continue
        if key is not None:
            meta[key] = " ".join(filter(None, block))
            key, block = None, []
        name, sep, value = line.partition(":")
        if not sep or name.strip() != name:
            continue
        if (value := value.strip()) in BLOCK_SCALARS:
            key = name.strip()
        else:
            meta[name.strip()] = value.strip("'\"")
    if key is not None:
        meta[key] = " ".join(filter(None, block))
    return meta


def _skill(raw: str, location: str, fallback: str) -> Skill | None:
    meta = parse_frontmatter(raw)
    if not meta.get("description"):
        return None
    return Skill(meta.get("name") or fallback, meta["description"], location)


def discover_skills(root: Path) -> list[Skill]:
    """Collect SKILL.md files from the project's and the user's skill roots."""
    skills = []
    for skill_root in [root / d for d in SKILL_DIRS] + [Path.home() / ".agent/skills"]:
        for path in sorted(skill_root.glob("*/SKILL.md")) if skill_root.is_dir() else []:
            found = _skill(path.read_text("utf-8", "replace"), str(path), path.parent.name)
            if found:
                skills.append(found)
    return skills


def fetch_hub_skills(tag: str = HUB_SKILL_TAG, limit: int = 30) -> list[Skill]:
    """Load skills published on the Hub as dataset repos with a root SKILL.md.

    Only the frontmatter is kept. The location stays a pinned resolve URL that
    the model fetches with `bash curl` if, and only if, it wants the body.
    """

    def get(url: str) -> Any:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.read().decode("utf-8", "replace")

    try:
        repos = json.loads(get(f"https://huggingface.co/api/datasets?filter={tag}&limit={limit}"))
    except Exception:
        return []
    skills = []
    for repo in repos:
        repo_id, sha = repo.get("id"), repo.get("sha", "main")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/{sha}/SKILL.md"
        try:
            found = _skill(get(url), url, str(repo_id).split("/")[-1])
        except Exception:
            continue
        if found:
            skills.append(found)
    return skills


# ---------------------------------------------------------------------------
# system prompt
#
# Deliberately tiny. Frontier models are already trained to be coding agents,
# so a 10k-token briefing mostly crowds out the context that matters. Only
# what actually exists gets appended.
# ---------------------------------------------------------------------------

SYSTEM = """You are a coding agent running in nanoharness. You help by reading files, \
editing code, writing files, and running shell commands.

Available tools:
- read: read a file, optionally a line range
- write: create or overwrite a file
- edit: replace exact snippets in an existing file
- bash: run a shell command in the working directory

Guidelines:
- Use bash for exploration: ls, rg, find, git.
- Read a file before editing it.
- Prefer edit over write for existing files.
- Verify your work by running the tests or the code itself.
- Be concise."""


def build_system_prompt(root: Path, skills: list[Skill]) -> str:
    parts = [SYSTEM]
    for name in CONTEXT_FILES:
        path = root / name
        if path.is_file() and (body := path.read_text("utf-8", "replace").strip()):
            parts.append(f'<project_instructions path="{name}">\n{body}\n</project_instructions>')
    if skills:
        listing = "\n".join(
            f"  <skill><name>{s.name}</name><description>{s.description}</description>"
            f"<location>{s.location}</location></skill>"
            for s in skills
        )
        parts.append(
            "These skills hold specialized instructions. When a task matches one, load its "
            "location first: `read` for a path, `bash curl -sL` for a URL.\n\n"
            f"<available_skills>\n{listing}\n</available_skills>"
        )
    parts.append(f"Current working directory: {root}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# tools
#
# Four is enough. That is pi's finding and it holds up: read/write/edit/bash
# covers what a coding agent does, and bash subsumes grep, find, ls and git
# without spending context on four more schemas.
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """A failure the model is expected to read and route around."""


def resolve(root: Path, path: str) -> Path:
    """Resolve a tool path. An absolute `path` replaces `root`, which is pathlib's rule."""
    return (root / path).resolve()


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'s' * (count != 1)}"


def truncate(text: str, max_lines: int, max_bytes: int, keep: str = "head") -> str:
    lines = text.splitlines()
    head, tail = "", ""
    if len(lines) > max_lines:
        dropped = len(lines) - max_lines
        note = f"... [{dropped} lines truncated]"
        # The note goes where the cut happened, so the surviving text stays contiguous.
        if keep == "tail":
            lines, head = lines[-max_lines:], note + "\n"
        else:
            lines, tail = lines[:max_lines], "\n" + note
    text = "\n".join(lines)
    if len(text.encode()) > max_bytes:
        text = text.encode()[:max_bytes].decode("utf-8", "ignore")
        tail = f"\n... [truncated at {max_bytes} bytes]"
    return head + text + tail


def tool_read(root: Path, path: str, offset: int = 1, limit: int | None = None) -> str:
    """Read a file as 1-indexed numbered lines, so edits have something to cite."""
    target = resolve(root, path)
    if not target.is_file():
        raise ToolError(f"not a file: {path}")
    lines = target.read_text("utf-8", "replace").splitlines()
    if not lines:
        return "(empty file)"
    start = max(1, offset)
    if start > len(lines):
        raise ToolError(f"offset {start} is past the end of {path} ({len(lines)} lines)")
    end = len(lines) if limit is None else min(len(lines), start + limit - 1)
    body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(start, end + 1))
    return truncate(body, READ_MAX_LINES, READ_MAX_BYTES)


def tool_write(root: Path, path: str, content: str) -> str:
    target = resolve(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    verb = "overwrote" if target.is_file() else "created"
    target.write_text(content, encoding="utf-8")
    return f"{verb} {path} ({plural(len(content.splitlines()), 'line')})"


def tool_edit(root: Path, path: str, edits: list[dict[str, str]]) -> str:
    """Apply exact-match replacements and return a diff.

    Two rules, both borrowed from pi, both there because the loose version
    silently corrupts files: every `old` must be unique in the file, and every
    match is located against the *original* content rather than the result of
    the previous edit in the same call. Overlapping edits are refused instead
    of being applied in whatever order they happened to arrive.
    """
    target = resolve(root, path)
    if not target.is_file():
        raise ToolError(f"not a file: {path}")
    # Read bytes, not text: text mode collapses CRLF on the way in, which would
    # silently rewrite every line ending in the file we are about to save.
    raw = target.read_bytes().decode("utf-8", "replace")
    newline = "\r\n" if "\r\n" in raw else "\n"
    original = raw.replace("\r\n", "\n")

    spans: list[tuple[int, int, str]] = []
    for edit in edits:
        old = (edit.get("old") or "").replace("\r\n", "\n")
        if not old:
            raise ToolError("every edit needs a non-empty 'old'")
        count = original.count(old)
        if count == 0:
            raise ToolError(f"'old' not found in {path}; read the file and match it exactly")
        if count > 1:
            raise ToolError(f"'old' appears {count}x in {path}; include more surrounding context")
        start = original.index(old)
        spans.append((start, start + len(old), (edit.get("new") or "").replace("\r\n", "\n")))

    spans.sort()
    pairs = zip(spans, spans[1:], strict=False)  # consecutive pairs; the last span has no successor
    if any(start < prev_end for (_, prev_end, _), (start, _, _) in pairs):
        raise ToolError("edits overlap; merge them into a single edit")

    updated = original
    for start, end, new in reversed(spans):
        updated = updated[:start] + new + updated[end:]
    target.write_bytes(updated.replace("\n", newline).encode())

    name = os.path.relpath(target, root)
    diff = difflib.unified_diff(
        original.splitlines(), updated.splitlines(), f"a/{name}", f"b/{name}", lineterm="", n=2
    )
    return f"applied {plural(len(spans), 'edit')} to {name}\n" + truncate("\n".join(diff), 120, 8000)


def tool_bash(root: Path, command: str, timeout: int = BASH_TIMEOUT) -> str:
    """Run a command, keeping the tail of the output -- where errors live."""
    try:
        done = subprocess.run(command, shell=True, cwd=root, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ToolError(f"timed out after {timeout}s") from None
    out = truncate(
        ((done.stdout or "") + (done.stderr or "")).strip(), BASH_MAX_LINES, READ_MAX_BYTES, "tail"
    )
    prefix = "" if done.returncode == 0 else f"[exit {done.returncode}]\n"
    return prefix + (out or "(no output)")


TOOL_FUNCS: dict[str, Callable[..., str]] = {}


def tool(func: Callable[..., str], description: str, required: str, **properties: dict[str, Any]) -> dict:
    """Describe one tool for the model and register its implementation.

    The name comes from the function, so the schema and the dispatch table can
    never drift apart. `required` is a space separated list of property names.
    """
    name = func.__name__.removeprefix("tool_")
    TOOL_FUNCS[name] = func
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required.split(),
            },
        },
    }


STR, INT = {"type": "string"}, {"type": "integer"}
TOOLS = [
    tool(
        tool_read,
        "Read a file. Returns 1-indexed numbered lines. Use offset/limit for large files.",
        "path",
        path=STR | {"description": "File path, relative or absolute."},
        offset=INT | {"description": "First line to read, 1-indexed."},
        limit=INT | {"description": "Maximum number of lines."},
    ),
    tool(
        tool_write,
        "Create a file, or overwrite one entirely. Prefer edit for files that exist.",
        "path content",
        path=STR,
        content=STR,
    ),
    tool(
        tool_edit,
        "Replace exact snippets in a file. Each 'old' must occur exactly once and match byte for "
        "byte including indentation. Edits are matched against the original file, so they must not overlap.",
        "path edits",
        path=STR,
        edits={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"old": STR, "new": STR},
                "required": ["old", "new"],
            },
        },
    ),
    tool(
        tool_bash,
        "Run a shell command in the working directory. Use it for ls, rg, find, git and tests.",
        "command",
        command=STR,
        timeout=INT | {"description": f"Seconds, default {BASH_TIMEOUT}."},
    ),
]
MUTATING = {"write", "edit", "bash"}  # the calls worth confirming


def summarize(name: str, args: dict[str, Any]) -> str:
    """One line describing a pending call, for the log and the approval prompt."""
    if name == "bash":
        return args.get("command", "")
    if name == "edit":
        return f"{args.get('path', '?')}  ({plural(len(args.get('edits', [])), 'edit')})"
    if name == "write":
        return f"{args.get('path', '?')}  ({plural(len(args.get('content', '').splitlines()), 'line')})"
    span = f"  [{args.get('offset', 1)}:+{args.get('limit')}]" if args.get("limit") else ""
    return f"{args.get('path', '?')}{span}"


# ---------------------------------------------------------------------------
# agent loop
#
# reason -> act -> observe, until the model stops asking for tools. No UI
# imports here, so the loop stays readable and testable on its own.
# ---------------------------------------------------------------------------


@dataclass
class Callbacks:
    """How the loop reports to whatever is driving it."""

    on_text: Callable[[str], None] = lambda chunk: None
    on_tool_start: Callable[[str, dict[str, Any]], None] = lambda name, args: None
    on_tool_end: Callable[[str, str, bool], None] = lambda name, result, ok: None
    on_turn_end: Callable[[], None] = lambda: None
    approve: Callable[[str, dict[str, Any]], bool] = lambda name, args: True


@dataclass
class Agent:
    root: Path
    model: str
    client: InferenceClient
    system: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0
    cb: Callbacks = field(default_factory=Callbacks)

    def step(self) -> tuple[str, list[dict[str, Any]]]:
        """One model call. Streams text out through the callbacks as it arrives."""
        text, pending = "", {}
        for chunk in self.client.chat_completion(
            messages=[{"role": "system", "content": self.system}, *self.messages],
            model=self.model,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
            max_tokens=8192,
            stream_options={"include_usage": True},
        ):
            if getattr(chunk, "usage", None):
                self.tokens = chunk.usage.total_tokens or self.tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text += delta.content
                self.cb.on_text(delta.content)
            for call in delta.tool_calls or []:
                slot = pending.setdefault(call.index or 0, {"id": "", "name": "", "arguments": ""})
                slot["id"] = call.id or slot["id"]
                if call.function:
                    slot["name"] = call.function.name or slot["name"]
                    slot["arguments"] += call.function.arguments or ""
        return text, [pending[i] for i in sorted(pending)]

    def run(self, prompt: str, cb: Callbacks) -> None:
        """Drive one user turn to completion."""
        self.cb = cb
        self.messages.append({"role": "user", "content": prompt})

        for _ in range(MAX_STEPS):
            text, calls = self.step()
            if not calls:
                self.messages.append({"role": "assistant", "content": text})
                break
            self.messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in calls
                    ],
                }
            )
            for call in calls:
                result, ok = self.invoke(call)
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                cb.on_tool_end(call["name"], result, ok)
        else:
            # Falling out of the loop means the model never stopped asking for
            # tools. Say so rather than ending the turn as if it had finished.
            cb.on_text(f"\n[stopped after {MAX_STEPS} tool rounds]")

        cb.on_turn_end()

    def invoke(self, call: dict[str, Any]) -> tuple[str, bool]:
        name, args, error = call["name"], {}, None
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError as exc:
            error = f"error: arguments were not valid JSON: {exc}"

        # Announce the call before doing anything with it, so a failure is always
        # attributed to a visible tool rather than appearing out of nowhere.
        self.cb.on_tool_start(name, args)
        if error:
            return error, False
        func = TOOL_FUNCS.get(name)
        if func is None:
            return f"error: unknown tool {name!r}", False
        if name in MUTATING and not self.cb.approve(name, args):
            return "error: the user rejected this call", False
        try:
            return func(self.root, **args), True
        except ToolError as exc:
            return f"error: {exc}", False
        except TypeError as exc:
            return f"error: bad arguments for {name}: {exc}", False
        except Exception as exc:  # a crashing tool is context, not a fatal error
            return f"error: {type(exc).__name__}: {exc}", False


# ---------------------------------------------------------------------------
# terminal ui
# ---------------------------------------------------------------------------

ADD, DEL, DIM = "#3fb950", "#f85149", "#6e7681"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

THEME = Theme(
    name="nano",
    primary="#ffd21e",  # the Hugging Face yellow, since that is where the models come from
    secondary="#ff9d00",
    accent="#ffd21e",
    foreground="#e6edf3",
    background="#0e1116",
    surface="#0e1116",
    panel="#1b212a",
    success=ADD,
    error=DEL,
    warning="#d29922",
    dark=True,
    variables={"text-muted": DIM, "block-cursor-background": "#ffd21e"},
)

CSS = """
Screen { background: $background; }

#log {
    padding: 1 3 0 3;
    scrollbar-size: 0 1;
    scrollbar-background: $background;
    scrollbar-color: $panel;
    scrollbar-color-hover: $primary;
    scrollbar-color-active: $primary;
}

/* A turn reads top to bottom: your line is marked, the answer is plain prose,
   and the machinery underneath it is dim and indented. */
.user {
    margin: 2 0 1 0;
    padding: 0 1;
    background: $panel;
    border-left: thick $primary;
    text-style: bold;
    color: $foreground;
}
.assistant {
    margin: 1 0 0 0;
    padding: 0 1;
    border-left: thick #3d444d;
    color: $foreground;
}
.tool { margin: 1 0 0 0; }
.result { margin: 0 0 0 2; }
.gate { margin: 0 0 0 2; }
.note { margin: 1 0 0 0; color: $text-muted; }

#status { height: 1; padding: 0 3; color: $text-muted; background: $background; }
#prompt {
    margin: 0 2;
    padding: 0 1;
    border: round $panel;
    background: $background;
}
#prompt:focus { border: round $primary; }
#hint { height: 1; padding: 0 3; color: $text-muted; background: $background; }

"""


def result_style(name: str, line: str, ok: bool) -> str:
    """Colour one line of a tool result. Diffs get gutters; everything else stays quiet."""
    if not ok:
        return DEL
    if name == "edit" and not line.startswith(("+++", "---")):
        if line.startswith("+"):
            return ADD
        if line.startswith("-"):
            return DEL
    return DIM


def render_result(name: str, body: str, ok: bool, max_lines: int = 12) -> Text:
    """Fold a tool result into a glanceable block hung off the call above it."""
    lines = body.splitlines() or ["(no output)"]
    if ok and name == "edit":
        # The a/ and b/ header names repeat the path already shown on the call line.
        lines = [line for line in lines if not line.startswith(("--- a/", "+++ b/"))]
    extra = len(lines) - max_lines
    shown = lines[:max_lines]
    if extra > 0:
        shown.append(f"… +{plural(extra, 'more line')}")
    out = Text()
    for i, line in enumerate(shown):
        out.append("└ " if i == 0 else "\n  ", style=DIM)
        out.append(line, style=result_style(name, line, ok))
    return out


class NanoHarness(App[None]):
    TITLE = "nanoharness"
    CSS = CSS
    BINDINGS = [
        Binding("ctrl+c", "bail", "quit"),
        Binding("ctrl+l", "clear", "clear"),
    ]

    def __init__(self, agent: Agent, skills: list[Skill], yolo: bool) -> None:
        super().__init__()
        self.agent, self.skills, self.yolo = agent, skills, yolo
        self.busy = False
        self.live: Static | None = None
        self.buffer = ""
        self.started = 0.0
        self.frame = 0
        self.gate: threading.Event | None = None
        self.row: Static | None = None
        self.answer = ""

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="log")
        yield Static(id="status")
        yield Input(placeholder="ask for a change…", id="prompt")
        yield Static(id="hint")

    def on_mount(self) -> None:
        self.register_theme(THEME)
        self.theme = "nano"
        self.query_one("#prompt", Input).focus()

        skills = f"  ·  {plural(len(self.skills), 'skill')}" if self.skills else ""
        self.say(
            Text.assemble(
                ("◆ ", f"bold {THEME.primary}"),
                ("nanoharness", "bold"),
                (f"\n  {self.agent.model}  ·  {self.agent.root}{skills}", DIM),
            ),
            "note",
        )
        self.query_one("#hint", Static).update(
            Text("/help for commands   ctrl+l clear   ctrl+c quit", style=DIM)
        )
        # One timer drives the spinner and the elapsed clock; it is cheap and
        # only repaints a single line.
        self.set_interval(1 / 12, self.sync_status)
        self.sync_status()

    # --- rendering ---

    def status_line(self) -> Text:
        """Only says something while there is something to say."""
        if not self.busy:
            return Text("")
        self.frame = (self.frame + 1) % len(SPINNER)
        return Text.assemble(
            (SPINNER[self.frame], f"bold {THEME.primary}"),
            (f"  {time.monotonic() - self.started:.0f}s", DIM),
        )

    def say(self, renderable: Any, css_class: str = "note") -> Static:
        widget = Static(renderable, classes=css_class)
        log = self.query_one("#log", VerticalScroll)
        log.mount(widget)
        log.scroll_end(animate=False)
        return widget

    def sync_status(self) -> None:
        self.query_one("#status", Static).update(self.status_line())

    # --- input ---

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text, event.input.value = event.value.strip(), ""
        if not text or self.busy:
            return
        if text.startswith("/"):
            return self.command(text)
        self.say(Text(text), "user")
        self.busy, self.started = True, time.monotonic()
        self.sync_status()
        self.turn(text)

    def command(self, text: str) -> None:
        name, *args = shlex.split(text)
        match name[1:]:
            case "q" | "quit" | "exit":
                self.exit()
            case "clear":
                self.action_clear()
            case "model":
                if args:
                    self.agent.model = args[0]
                self.say(Text(f"model: {self.agent.model}", style="dim"))
                self.sync_status()
            case "yolo":
                self.yolo = not self.yolo
                self.say(Text(f"approvals: {'off' if self.yolo else 'on'}", style="dim"))
                self.sync_status()
            case "skills":
                body = "\n".join(f"{s.name}  —  {s.description[:70]}" for s in self.skills)
                self.say(Text(body or "no skills found", style="dim"))
            case _:
                self.say(Text("/model <id>   /skills   /yolo   /clear   /quit", style="dim"))

    def action_clear(self) -> None:
        self.agent.messages.clear()
        self.query_one("#log", VerticalScroll).remove_children()
        self.say(Text("context cleared", style="dim"))

    # --- the turn runs on a worker thread; every ui touch hops back ---

    @work(thread=True, exclusive=True)
    def turn(self, prompt: str) -> None:
        hop = self.call_from_thread

        try:
            self.agent.run(
                prompt,
                Callbacks(
                    on_text=lambda chunk: hop(self.stream, chunk),
                    on_tool_start=lambda name, args: hop(self.show_call, name, args),
                    on_tool_end=lambda name, result, ok: hop(self.show_result, name, result, ok),
                    on_turn_end=lambda: hop(self.finish),
                    approve=self.approve,
                ),
            )
        except Exception as exc:
            hop(self.fail, f"{type(exc).__name__}: {exc}")

    def stream(self, chunk: str) -> None:
        if self.live is None:
            self.buffer, self.live = "", self.say("", "assistant")
        self.buffer += chunk
        self.live.update(self.buffer)
        self.query_one("#log", VerticalScroll).scroll_end(animate=False)

    def seal(self) -> None:
        """Re-render finished prose as markdown; it stays plain while streaming."""
        if self.live is not None and self.buffer.strip():
            self.live.update(Markdown(self.buffer))
        self.live, self.buffer = None, ""

    def show_call(self, name: str, args: dict[str, Any]) -> None:
        self.seal()
        self.say(
            Text.assemble(
                ("● ", f"bold {THEME.primary}"),
                (name, "bold"),
                ("  " + summarize(name, args), DIM),
            ),
            "tool",
        )

    def show_result(self, name: str, result: str, ok: bool) -> None:
        self.say(render_result(name, result, ok), "result")

    # --- approval, inline rather than a screen ---

    def approve(self, _name: str, _args: dict[str, Any]) -> bool:
        """Called on the worker thread. Blocks it until a keypress resolves the gate."""
        if self.yolo:
            return True
        done = threading.Event()
        self.call_from_thread(self.open_gate, done)
        done.wait()
        self.yolo = self.yolo or self.answer == "always"
        return self.answer in ("yes", "always")

    def open_gate(self, done: threading.Event) -> None:
        """Ask in the transcript, directly under the call being asked about.

        The gate becomes answerable only here, on the UI thread, once the row is
        up and the input is off. Publishing it from the worker instead would let
        a keypress resolve a gate that has not been drawn yet -- and this method
        would then disable the input with nothing left to switch it back on.
        """
        self.answer, self.gate = "", done
        self.row = self.say(
            Text.assemble(
                ("? ", f"bold {THEME.warning}"),
                ("run this?", "bold"),
                ("   y", f"bold {ADD}"),
                (" yes", DIM),
                ("   a", f"bold {ADD}"),
                (" always", DIM),
                ("   n", f"bold {DEL}"),
                (" no", DIM),
            ),
            "gate",
        )
        # The input would otherwise swallow y/a/n before the app sees them.
        self.query_one("#prompt", Input).disabled = True
        self.set_focus(None)

    def on_key(self, event: events.Key) -> None:
        if self.gate is None:
            return
        answer = {"y": "yes", "a": "always", "n": "no", "escape": "no"}.get(event.key)
        if answer is None:
            return
        event.stop()
        self.close_gate(answer)

    def close_gate(self, answer: str) -> None:
        """Take the row down, give the keyboard back, and release the worker."""
        self.answer = answer
        if self.row is not None:
            self.row.remove()
            self.row = None
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        gate, self.gate = self.gate, None
        if gate is not None:
            gate.set()

    def action_bail(self) -> None:
        """Quit. An open gate is refused first, or the blocked worker hangs the exit."""
        if self.gate is not None:
            self.close_gate("no")
        self.exit()

    def finish(self) -> None:
        self.seal()
        self.busy = False
        self.sync_status()

    def fail(self, message: str) -> None:
        self.seal()
        self.say(Text(message, style="red"))
        self.busy = False
        self.sync_status()


# ---------------------------------------------------------------------------
# sign-in
#
# A first run should hand you a token rather than an error. Inference Providers
# authenticate with an ordinary read token, and huggingface_hub caches it where
# every other HF tool will find it, so this happens exactly once per machine.
# ---------------------------------------------------------------------------

TOKEN_URL = "https://huggingface.co/settings/tokens/new?tokenType=read"


def sign_in(token_file: str | None = None) -> str | None:
    """Return a Hub token, walking the user through creating one when there is none.

    Precedence: an explicit --token-file, then $NANOHARNESS_TOKEN, then whatever
    huggingface_hub already has ($HF_TOKEN or the login cache). The first two let
    you point this harness at one token without disturbing your global HF login.
    """
    console = Console()
    if token_file:
        path = Path(token_file).expanduser()
        try:
            token = path.read_text().strip()
        except OSError as exc:
            console.print(f"[#f85149]Cannot read {path}:[/] {exc}")
            return None
        if not token:
            console.print(f"[#f85149]{path} is empty.[/]")
            return None
        return token

    if token := (os.environ.get("NANOHARNESS_TOKEN") or get_token()):
        return token

    if not sys.stdin.isatty():
        console.print("[#f85149]No Hugging Face token.[/] Set HF_TOKEN, or run `hf auth login`.")
        return None

    with suppress(Exception):
        webbrowser.open(TOKEN_URL)
    console.print(
        f"\n  [bold {THEME.primary}]◆[/] [bold]nanoharness[/] needs a Hugging Face token.\n\n"
        f"  Create one with [bold]read[/] access:\n"
        f"  [{THEME.primary} underline]{TOKEN_URL}[/]\n"
        f"  [dim]If a browser did not open, copy that link.[/]\n\n"
        f"  [dim]Paste it below — input stays hidden, and it is saved for next time.[/]\n"
    )

    for _ in range(3):
        try:
            token = getpass("  token: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not token:
            continue
        try:
            name = whoami(token=token)["name"]
        except Exception as exc:
            console.print(f"  [#f85149]That token did not work[/] [dim]({type(exc).__name__})[/]\n")
            continue
        # Written straight to the shared cache rather than through login(),
        # which does extra account bookkeeping this flow does not need.
        path = Path(HF_TOKEN_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token)
        path.chmod(0o600)
        console.print(f"\n  [#3fb950]✓[/] signed in as [bold]{name}[/]  [dim]{path}[/]\n")
        return token

    console.print("  [dim]Giving up for now — run nanoharness again when you have a token.[/]")
    return None


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def headless(agent: Agent, prompt: str) -> int:
    """One turn on plain stdout. Handy for scripts, and for reading the loop bare."""
    agent.run(
        prompt,
        Callbacks(
            on_text=lambda chunk: (sys.stdout.write(chunk), sys.stdout.flush()),
            on_tool_start=lambda name, args: print(f"\n▸ {name}  {summarize(name, args)}", flush=True),
            on_tool_end=lambda name, result, ok: print(truncate(result, 20, 2000, "tail"), flush=True),
        ),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser("nanoharness", description="a coding agent in one file")
    parser.add_argument("prompt", nargs="*", help="run one turn without the ui and exit")
    parser.add_argument(
        "--model",
        default=os.environ.get("NANOHARNESS_MODEL", DEFAULT_MODEL),
        help=f"model on Inference Providers (default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("NANOHARNESS_PROVIDER", "auto"),
        help="pin one provider instead of routing automatically",
    )
    parser.add_argument("--cwd", default=".", help="working directory")
    parser.add_argument(
        "--token-file",
        default=os.environ.get("NANOHARNESS_TOKEN_FILE"),
        help="read the Hub token from this file instead of the usual HF login",
    )
    parser.add_argument(
        "--bill-to",
        default=os.environ.get("NANOHARNESS_BILL_TO"),
        help="charge inference to this org, for tokens scoped to one",
    )
    parser.add_argument("--yolo", action="store_true", help="skip approval prompts")
    parser.add_argument("--hub-skills", action="store_true", help="also load skills from the Hub")
    args = parser.parse_args()

    token = sign_in(args.token_file)
    if token is None:
        return 1

    root = Path(args.cwd).resolve()
    skills = discover_skills(root) + (fetch_hub_skills() if args.hub_skills else [])
    agent = Agent(
        root=root,
        model=args.model,
        client=InferenceClient(provider=args.provider, api_key=token, bill_to=args.bill_to),
        system=build_system_prompt(root, skills),
    )

    if args.prompt:
        return headless(agent, " ".join(args.prompt))
    NanoHarness(agent, skills, yolo=args.yolo).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
