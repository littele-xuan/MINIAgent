from __future__ import annotations

import argparse
import compileall
import os
import runpy
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from miniagent.runtime.diagnostics import collect_diagnostics, format_diagnostics  # noqa: E402
from miniagent.runtime.env import load_dotenv_if_present  # noqa: E402


def _has_real_llm_env() -> bool:
    key_ok = bool(os.getenv("MCP_API_KEY") or os.getenv("OPENAI_API_KEY"))
    model_ok = bool(os.getenv("MCP_MODEL") or os.getenv("OPENAI_MODEL"))
    return key_ok and model_ok


def _missing_real_llm_env() -> list[str]:
    missing: list[str] = []
    if not (os.getenv("MCP_API_KEY") or os.getenv("OPENAI_API_KEY")):
        missing.append("MCP_API_KEY")
    if not (os.getenv("MCP_MODEL") or os.getenv("OPENAI_MODEL")):
        missing.append("MCP_MODEL")
    return missing


@contextmanager
def project_runtime(extra_path: Path | None = None):
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    old_modules = {name: sys.modules.get(name) for name in ("_bootstrap",)}
    os.chdir(ROOT)
    try:
        # `_bootstrap.py` exists in both tests_offline and tests_real.  runpy
        # executes scripts in one interpreter, so clear the cached module before
        # every script to avoid importing the offline bootstrap in real tests.
        for name in old_modules:
            sys.modules.pop(name, None)
        for p in [SRC, extra_path]:
            if p is not None and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def run_case(name: str, func: Callable[[], None]) -> tuple[str, bool, float]:
    print(f"\n=== RUN: {name} ===", flush=True)
    started = time.time()
    try:
        func()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        ok = code == 0
        if not ok:
            print(f"[SystemExit] {exc}", flush=True)
    except Exception as exc:
        ok = False
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
    else:
        ok = True
    elapsed = time.time() - started
    print(f"=== {'PASS' if ok else 'FAIL'}: {name} ({elapsed:.2f}s) ===", flush=True)
    return name, ok, elapsed


def compile_sources() -> None:
    targets = ["src", "agent.py", "run_all_tests.py", "run_real_tests.py", "tests_offline", "tests_real", "examples"]
    for target in targets:
        path = ROOT / target
        if not path.exists():
            continue
        if path.is_dir():
            ok = compileall.compile_dir(str(path), quiet=1, force=False)
        else:
            ok = compileall.compile_file(str(path), quiet=1, force=False)
        if not ok:
            raise RuntimeError(f"compile failed: {target}")


def run_script(path: Path) -> None:
    with project_runtime(path.parent):
        runpy.run_path(str(path), run_name="__main__")


def run_real_scripts(results: list[tuple[str, bool, float]]) -> bool:
    print("\n=== REAL LLM AGENT TESTS ENABLED ===")
    print("Using MCP_* first, with OPENAI_* as compatibility fallback.")
    for script in sorted((ROOT / "tests_real").glob("[0-9][0-9]_*.py")):
        results.append(run_case(f"real:{script.name}", lambda p=script: run_script(p)))
        if not results[-1][1]:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MINIAgent checks from the project root.")
    parser.add_argument("--skip-real", action="store_true", help="Skip tests_real even if MCP_API_KEY/MCP_MODEL are present.")
    parser.add_argument("--require-real", action="store_true", help="Fail if MCP_API_KEY/MCP_MODEL are missing instead of skipping real API tests.")
    parser.add_argument("--real-only", action="store_true", help="Run only tests_real. This still requires MCP_API_KEY and MCP_MODEL.")
    args = parser.parse_args()

    load_dotenv_if_present(ROOT)
    results: list[tuple[str, bool, float]] = []

    if not args.real_only:
        results.append(run_case("compile", compile_sources))
        if not results[-1][1]:
            return summarize(results, skipped_real=False)

        def doctor() -> None:
            print(format_diagnostics(collect_diagnostics(ROOT / "config" / "agent.yaml")))

        results.append(run_case("doctor", doctor))
        if not results[-1][1]:
            return summarize(results, skipped_real=False)

        for script in sorted((ROOT / "tests_offline").glob("[0-9][0-9]_*.py")):
            results.append(run_case(f"offline:{script.name}", lambda p=script: run_script(p)))
            if not results[-1][1]:
                return summarize(results, skipped_real=False)

    skipped_real = False
    if args.skip_real and not args.real_only:
        skipped_real = True
        print("\n=== SKIP: tests_real (--skip-real) ===")
    elif not _has_real_llm_env():
        missing = _missing_real_llm_env()
        if args.require_real or args.real_only:
            print(f"\nMissing real LLM environment variables: {', '.join(missing)}", file=sys.stderr)
            print("Set MCP_API_BASE, MCP_API_KEY, MCP_MODEL in the environment or project .env.", file=sys.stderr)
            results.append(("tests_real", False, 0.0))
            return summarize(results, skipped_real=False)
        skipped_real = True
        print("\n=== SKIP: tests_real because MCP_API_KEY/MCP_MODEL are not fully set ===")
        print("Set MCP_API_BASE, MCP_API_KEY, MCP_MODEL in the environment or project .env, then run: python run_all_tests.py --require-real")
    else:
        if not run_real_scripts(results):
            return summarize(results, skipped_real=False)

    return summarize(results, skipped_real=skipped_real)


def summarize(results: list[tuple[str, bool, float]], *, skipped_real: bool) -> int:
    print("\n=== SUMMARY ===")
    for name, ok, elapsed in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<36} {elapsed:>7.2f}s")
    if skipped_real:
        print("SKIP  tests_real                           no MCP_API_KEY/MCP_MODEL or --skip-real")
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nALL REQUESTED TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
