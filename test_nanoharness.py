"""Tests for nanoharness. No network: the model is faked.

pytest test_nanoharness.py
"""

from __future__ import annotations

import asyncio
import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import nanoharness
from nanoharness import (
    Agent,
    Callbacks,
    NanoHarness,
    Skill,
    ToolError,
    build_system_prompt,
    discover_skills,
    git_autocommit,
    parse_frontmatter,
    summarize,
    tool_bash,
    tool_edit,
    tool_read,
    tool_write,
    truncate,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    return tmp_path


# --- read -----------------------------------------------------------------


def test_read_numbers_lines(root: Path) -> None:
    assert tool_read(root, "calc.py").splitlines()[0] == "    1  def add(a, b):"


def test_read_offset_and_limit(root: Path) -> None:
    out = tool_read(root, "calc.py", offset=2, limit=1)
    assert out.strip().startswith("2  ") and "return" in out


def test_read_rejects_missing_file(root: Path) -> None:
    with pytest.raises(ToolError, match="not a file"):
        tool_read(root, "nope.py")


def test_read_rejects_offset_past_end(root: Path) -> None:
    with pytest.raises(ToolError, match="past the end"):
        tool_read(root, "calc.py", offset=99)


def test_read_reports_empty_file(root: Path) -> None:
    (root / "empty.py").write_text("")
    assert tool_read(root, "empty.py") == "(empty file)"


# --- write ----------------------------------------------------------------


def test_write_creates_then_overwrites(root: Path) -> None:
    assert "created" in tool_write(root, "new.py", "x = 1\n")
    assert "overwrote" in tool_write(root, "new.py", "x = 2\n")
    assert (root / "new.py").read_text() == "x = 2\n"


def test_write_makes_parent_directories(root: Path) -> None:
    tool_write(root, "pkg/mod/thing.py", "y = 1\n")
    assert (root / "pkg/mod/thing.py").is_file()


# --- edit -----------------------------------------------------------------


def test_edit_replaces_and_returns_a_diff(root: Path) -> None:
    out = tool_edit(root, "calc.py", [{"old": "a - b", "new": "a + b"}])
    assert (root / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert "-    return a - b" in out and "+    return a + b" in out


def test_edit_requires_a_unique_match(root: Path) -> None:
    (root / "dup.py").write_text("x = 1\nx = 1\n")
    with pytest.raises(ToolError, match="appears 2x"):
        tool_edit(root, "dup.py", [{"old": "x = 1", "new": "x = 2"}])


def test_edit_rejects_a_missing_match(root: Path) -> None:
    with pytest.raises(ToolError, match="not found"):
        tool_edit(root, "calc.py", [{"old": "nonexistent", "new": "x"}])


def test_edit_matches_against_the_original_not_the_running_result(root: Path) -> None:
    """Both `old`s are located in the original file, so edit two cannot see edit one."""
    (root / "seq.py").write_text("first\nsecond\n")
    tool_edit(root, "seq.py", [{"old": "first", "new": "second"}, {"old": "second", "new": "third"}])
    assert (root / "seq.py").read_text() == "second\nthird\n"


def test_edit_refuses_overlapping_edits(root: Path) -> None:
    (root / "over.py").write_text("abcdef\n")
    with pytest.raises(ToolError, match="overlap"):
        tool_edit(root, "over.py", [{"old": "abcd", "new": "x"}, {"old": "cdef", "new": "y"}])


def test_edit_preserves_crlf_line_endings(root: Path) -> None:
    (root / "win.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    tool_edit(root, "win.py", [{"old": "a = 1", "new": "a = 9"}])
    assert (root / "win.py").read_bytes() == b"a = 9\r\nb = 2\r\n"


def test_edit_rejects_empty_old(root: Path) -> None:
    with pytest.raises(ToolError, match="non-empty"):
        tool_edit(root, "calc.py", [{"old": "", "new": "x"}])


# --- bash -----------------------------------------------------------------


def test_bash_returns_output(root: Path) -> None:
    assert tool_bash(root, "echo hello") == "hello"


def test_bash_reports_the_exit_code(root: Path) -> None:
    assert tool_bash(root, "exit 3").startswith("[exit 3]")


def test_bash_runs_in_the_working_directory(root: Path) -> None:
    assert "calc.py" in tool_bash(root, "ls")


def test_bash_times_out(root: Path) -> None:
    with pytest.raises(ToolError, match="timed out"):
        tool_bash(root, "sleep 5", timeout=1)


def test_bash_keeps_the_tail_of_long_output(root: Path) -> None:
    out = tool_bash(root, "seq 1 500")
    assert out.rstrip().endswith("500") and "300 lines truncated" in out


# --- helpers --------------------------------------------------------------


def test_truncate_keeps_head_or_tail() -> None:
    body = "\n".join(str(i) for i in range(10))
    assert truncate(body, 3, 1000).startswith("0\n1\n2")
    assert truncate(body, 3, 1000).rstrip().endswith("[7 lines truncated]")
    assert truncate(body, 3, 1000, "tail").rstrip().endswith("7\n8\n9")
    assert truncate(body, 3, 1000, "tail").startswith("... [7 lines truncated]")


def test_parse_frontmatter() -> None:
    meta = parse_frontmatter("---\nname: demo\ndescription: does a thing\n---\nbody\n")
    assert meta == {"name": "demo", "description": "does a thing"}


def test_parse_frontmatter_ignores_a_document_without_one() -> None:
    assert parse_frontmatter("# just markdown\n") == {}


@pytest.mark.parametrize("marker", [">", "|", ">-", "|-", ""])
def test_parse_frontmatter_folds_block_scalars(marker: str) -> None:
    """Real skills on the Hub write long descriptions as YAML blocks."""
    meta = parse_frontmatter(f"---\nname: demo\ndescription: {marker}\n  one\n  two\n---\nbody\n")
    assert meta == {"name": "demo", "description": "one two"}


def test_parse_frontmatter_ends_a_block_at_the_next_key() -> None:
    meta = parse_frontmatter("---\ndescription: >\n  long\n\n  text\nname: demo\n---\nbody\n")
    assert meta == {"description": "long text", "name": "demo"}


def test_summarize_describes_each_call() -> None:
    assert summarize("bash", {"command": "ls -la"}) == "ls -la"
    assert "2 edits" in summarize("edit", {"path": "a.py", "edits": [{}, {}]})
    assert "1 line" in summarize("write", {"path": "a.py", "content": "x\n"})


# --- skills and prompt ----------------------------------------------------


def test_discover_skills_reads_frontmatter(root: Path) -> None:
    skill_dir = root / ".agent/skills/tidy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: tidy\ndescription: tidies up\n---\nbody\n")
    assert discover_skills(root)[0] == Skill("tidy", "tidies up", str(skill_dir / "SKILL.md"))


def test_discover_skills_skips_a_file_without_a_description(root: Path) -> None:
    skill_dir = root / ".agent/skills/broken"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: broken\n---\nbody\n")
    assert discover_skills(root) == []


def test_system_prompt_stays_small() -> None:
    """The whole point. Guard against it creeping back up to 10k tokens."""
    assert len(build_system_prompt(Path("/tmp"), [])) < 1000


def test_system_prompt_includes_context_and_skills(root: Path) -> None:
    (root / "AGENTS.md").write_text("always use tabs")
    prompt = build_system_prompt(root, [Skill("tidy", "tidies up", "/s/SKILL.md")])
    assert "always use tabs" in prompt
    assert "<name>tidy</name>" in prompt and "/s/SKILL.md" in prompt


# --- git ------------------------------------------------------------------


def test_git_autocommit_commits_changes(root: Path) -> None:
    for args in (["init", "-q"], ["config", "user.email", "t@t.co"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=root, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=root, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, capture_output=True)

    (root / "calc.py").write_text("changed\n")
    assert git_autocommit(root, "fix the thing") is not None
    log = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=root, capture_output=True, text=True)
    assert log.stdout.strip() == "fix the thing"


def test_git_autocommit_is_a_no_op_outside_a_repo(root: Path) -> None:
    assert git_autocommit(root, "nothing to see") is None


# --- the loop, with a faked model ----------------------------------------


def chunk(content: str = "", calls: list[dict] | None = None) -> SimpleNamespace:
    tool_calls = [
        SimpleNamespace(
            index=i,
            id=c["id"],
            function=SimpleNamespace(name=c["name"], arguments=c["arguments"]),
        )
        for i, c in enumerate(calls or [])
    ]
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


class FakeClient:
    """Replays scripted rounds in place of Inference Providers."""

    def __init__(self, rounds: list[list[SimpleNamespace]]) -> None:
        self.rounds, self.calls = rounds, 0

    def chat_completion(self, **_: object) -> list[SimpleNamespace]:
        self.calls += 1
        return self.rounds[min(self.calls - 1, len(self.rounds) - 1)]


def make_agent(root: Path, rounds: list[list[SimpleNamespace]]) -> Agent:
    return Agent(
        root=root,
        model="fake",
        client=FakeClient(rounds),  # type: ignore[arg-type]
        system="test",
        autocommit=False,
    )


def test_loop_runs_a_tool_then_answers(root: Path) -> None:
    call = {"id": "1", "name": "read", "arguments": '{"path": "calc.py"}'}
    agent = make_agent(root, [[chunk(calls=[call])], [chunk("all done")]])
    seen: list[tuple[str, bool]] = []
    agent.run("look at calc", Callbacks(on_tool_end=lambda n, r, ok: seen.append((n, ok))))

    assert seen == [("read", True)]
    assert agent.messages[-1] == {"role": "assistant", "content": "all done"}
    assert agent.messages[-2]["role"] == "tool"
    assert "def add" in agent.messages[-2]["content"]


def test_loop_streams_text_through_the_callback(root: Path) -> None:
    agent = make_agent(root, [[chunk("hel"), chunk("lo")]])
    out: list[str] = []
    agent.run("hi", Callbacks(on_text=out.append))
    assert "".join(out) == "hello"


def test_loop_accumulates_split_tool_arguments(root: Path) -> None:
    """Providers stream arguments in fragments; they must be concatenated."""
    parts = [
        chunk(calls=[{"id": "1", "name": "read", "arguments": '{"path": '}]),
        chunk(calls=[{"id": "", "name": "", "arguments": '"calc.py"}'}]),
    ]
    agent = make_agent(root, [parts, [chunk("done")]])
    agent.run("read it", Callbacks())
    assert "def add" in agent.messages[-2]["content"]


def test_loop_reports_a_rejected_call_back_to_the_model(root: Path) -> None:
    call = {"id": "1", "name": "write", "arguments": '{"path": "x.py", "content": "boom"}'}
    agent = make_agent(root, [[chunk(calls=[call])], [chunk("understood")]])
    agent.run("write it", Callbacks(approve=lambda name, args: False))

    assert not (root / "x.py").exists()
    assert agent.messages[-2]["content"] == "error: the user rejected this call"


def test_loop_turns_a_tool_error_into_an_observation(root: Path) -> None:
    call = {"id": "1", "name": "read", "arguments": '{"path": "missing.py"}'}
    agent = make_agent(root, [[chunk(calls=[call])], [chunk("ok")]])
    agent.run("read", Callbacks())
    assert agent.messages[-2]["content"].startswith("error: not a file")


def test_loop_survives_malformed_tool_arguments(root: Path) -> None:
    call = {"id": "1", "name": "read", "arguments": "{not json"}
    agent = make_agent(root, [[chunk(calls=[call])], [chunk("ok")]])
    agent.run("read", Callbacks())
    assert "not valid JSON" in agent.messages[-2]["content"]


def test_loop_reports_an_exhausted_step_budget(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that never stops asking for tools must not look like a finished turn."""
    monkeypatch.setattr(nanoharness, "MAX_STEPS", 2)
    call = {"id": "1", "name": "read", "arguments": '{"path": "calc.py"}'}
    agent = make_agent(root, [[chunk(calls=[call])]])
    out: list[str] = []
    agent.run("spin", Callbacks(on_text=out.append))
    assert "stopped after 2 tool rounds" in "".join(out)


def test_read_only_calls_skip_approval(root: Path) -> None:
    call = {"id": "1", "name": "read", "arguments": '{"path": "calc.py"}'}
    agent = make_agent(root, [[chunk(calls=[call])], [chunk("ok")]])
    asked: list[str] = []

    def approve(name: str, args: dict) -> bool:
        asked.append(name)
        return True

    agent.run("read", Callbacks(approve=approve))
    assert asked == []


# --- sign-in --------------------------------------------------------------


def test_sign_in_uses_an_existing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nanoharness, "get_token", lambda: "hf_already_here")
    assert nanoharness.sign_in() == "hf_already_here"


def test_sign_in_reads_a_token_file(root: Path) -> None:
    (root / ".inf_token").write_text("hf_from_a_file\n")
    assert nanoharness.sign_in(str(root / ".inf_token")) == "hf_from_a_file"


def test_sign_in_prefers_the_token_file_over_the_hf_login(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointing the harness at one token must not depend on the global HF login."""
    monkeypatch.setattr(nanoharness, "get_token", lambda: "hf_global")
    (root / ".inf_token").write_text("hf_specific")
    assert nanoharness.sign_in(str(root / ".inf_token")) == "hf_specific"


@pytest.mark.parametrize("contents", ["", "   \n"])
def test_sign_in_rejects_an_empty_token_file(root: Path, contents: str) -> None:
    (root / ".inf_token").write_text(contents)
    assert nanoharness.sign_in(str(root / ".inf_token")) is None


def test_sign_in_reports_an_unreadable_token_file(root: Path) -> None:
    assert nanoharness.sign_in(str(root / "nope")) is None


def test_sign_in_honours_the_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nanoharness, "get_token", lambda: "hf_global")
    monkeypatch.setenv("NANOHARNESS_TOKEN", "hf_from_env")
    assert nanoharness.sign_in() == "hf_from_env"


def test_sign_in_does_not_prompt_without_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A piped or scripted run should say what to do, not block on a hidden prompt."""
    monkeypatch.setattr(nanoharness, "get_token", lambda: None)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    assert nanoharness.sign_in() is None


# --- ui -------------------------------------------------------------------


@pytest.mark.parametrize(("key", "written"), [("y", True), ("n", False)])
def test_the_approval_gate_blocks_a_write_until_a_key(root: Path, key: str, written: bool) -> None:
    """Covers the worker-thread to UI hop: the loop blocks until a key is pressed."""
    call = {"id": "1", "name": "write", "arguments": '{"path": "gated.py", "content": "x = 1\\n"}'}
    agent = make_agent(root, [[chunk(calls=[call])], [chunk("ok")]])
    app = NanoHarness(agent, [], yolo=False)

    async def drive() -> None:
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "write it"
            await pilot.press("enter")
            for _ in range(50):
                await pilot.pause()
                if app.row is not None:  # the inline gate is up
                    break
            assert app.query_one("#prompt").disabled, "input must not eat the answer"
            await pilot.press(key)
            for _ in range(50):
                await pilot.pause()
                if not app.busy:
                    break

    asyncio.run(drive())
    assert (root / "gated.py").exists() is written


def test_the_app_boots_and_takes_a_command(root: Path) -> None:
    app = NanoHarness(make_agent(root, [[chunk("hi")]]), [Skill("tidy", "tidies", "/s")], yolo=True)

    async def drive() -> str:
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/skills"
            await pilot.press("enter")
            await pilot.pause()
            return str(app.query_one("#log").children[-1].content)

    assert "tidy" in asyncio.run(drive())
