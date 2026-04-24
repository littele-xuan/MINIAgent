from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from miniagent import MINIAgent  # noqa: E402
from miniagent.runtime.config import AgentConfig  # noqa: E402
from miniagent.runtime.diagnostics import collect_diagnostics, format_diagnostics  # noqa: E402
from miniagent.runtime.env import load_dotenv_if_present  # noqa: E402


def print_event(event: dict) -> None:
    kind = event.get("event")
    if kind == "llm_start":
        print(f"\n[llm] turn {event.get('turn')} ...", flush=True)
    elif kind == "llm_empty_retry":
        print(f"[llm] empty response; retrying finalization ({event.get('attempt')})", flush=True)
    elif kind == "llm_end":
        text = (event.get("text") or "").strip()
        calls = event.get("tool_calls") or []
        if text:
            preview = text.replace("\n", " ")[:500]
            print(f"[llm:text] {preview}", flush=True)
        if calls:
            print(f"[llm:tools] {', '.join(calls)}", flush=True)
    elif kind == "tool_start":
        print(f"[tool] {event.get('name')} args={event.get('arguments')}", flush=True)
    elif kind == "tool_end":
        status = "ok" if event.get("ok") else "fail"
        preview = str(event.get("content") or "").replace("\n", " ")[:500]
        print(f"[tool:{status}] {event.get('name')} -> {preview}", flush=True)


def _config_summary(config_path: str) -> str:
    config = AgentConfig.load(config_path)
    base = config.llm_base_url() or "<OpenAI SDK default>"
    return f"model={config.llm_model()} | api_mode={config.llm.api_mode} | base={base}"


def run_once(prompt: str, *, config: str, verbose: bool = False, show_session: bool = False) -> int:
    callback = print_event if verbose else None
    agent = MINIAgent.create_default(config, event_callback=callback)
    result = agent.run(prompt)
    final = (result.final_text or "").strip()
    if not final:
        final = "[empty final answer] The LLM endpoint returned no text after retries. Check the run log in workspace/logs."
    if verbose:
        print("\n=== FINAL ===")
    print(final)
    if show_session or verbose:
        print(f"\n[session] {result.session_id} | turns={result.turns} | exit={result.exit_reason} | usage={result.usage}")
    return 0 if result.exit_reason in {"final_answer", "blocked_for_user"} else 1


def run_repl(*, config: str, verbose: bool = False, show_session: bool = False) -> int:
    print("MINIAgent interactive mode. 输入任务后按 Enter 运行；输入 :q / :quit / exit 退出。")
    print(f"Runtime: {_config_summary(config)}")
    print("Run tests: python run_all_tests.py --require-real  或  python run_real_tests.py")
    while True:
        try:
            prompt = input("\nminiagent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not prompt:
            continue
        if prompt.lower() in {":q", ":quit", "quit", "exit"}:
            print("bye")
            return 0
        try:
            code = run_once(prompt, config=config, verbose=verbose, show_session=show_session)
            if code != 0:
                print(f"[warn] run finished with non-success exit code: {code}", file=sys.stderr)
        except Exception as exc:
            print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
            print("可运行 `python agent.py --doctor` 检查 MCP_API_BASE、MCP_API_KEY、MCP_MODEL、Langfuse 和依赖。", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MINIAgent GenericAgentCore.")
    parser.add_argument("prompt", nargs="*", help="Task prompt. If omitted in a terminal, starts interactive mode. If stdin is piped, reads stdin.")
    parser.add_argument("--config", default=str(ROOT / "config" / "agent.yaml"))
    parser.add_argument("--verbose", "-v", action="store_true", help="Show LLM/tool event logs.")
    parser.add_argument("--show-session", action="store_true", help="Print session id, turn count, exit reason and usage after each run.")
    parser.add_argument("--quiet", action="store_true", help="Deprecated alias; output is concise by default.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Force interactive REPL mode.")
    parser.add_argument("--doctor", action="store_true", help="Print local configuration diagnostics without calling the LLM API.")
    args = parser.parse_args()

    load_dotenv_if_present(ROOT)
    if args.doctor:
        print(format_diagnostics(collect_diagnostics(args.config)))
        return 0

    prompt = " ".join(args.prompt).strip()
    if prompt:
        return run_once(prompt, config=args.config, verbose=args.verbose, show_session=args.show_session)

    if args.interactive or sys.stdin.isatty():
        return run_repl(config=args.config, verbose=args.verbose, show_session=args.show_session)

    prompt = sys.stdin.read().strip()
    if not prompt:
        parser.error("Provide a prompt, pipe stdin, or use --interactive.")
    return run_once(prompt, config=args.config, verbose=args.verbose, show_session=args.show_session)


if __name__ == "__main__":
    raise SystemExit(main())
