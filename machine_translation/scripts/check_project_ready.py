from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


FORBIDDEN_TRACKED_RE = re.compile(
    r"(^|/)(model/|data/raw/|data/processed/|wandb/|outputs/)|"
    r"data/results\.csv$|"
    r"\.(safetensors|pt|bin)$"
)


@dataclass(frozen=True)
class CheckResult:
    status: str
    name: str
    detail: str = ""


def _path_exists(root: Path, relative_path: str) -> CheckResult:
    path = root / relative_path
    if path.exists():
        return CheckResult("PASS", relative_path)
    return CheckResult("FAIL", relative_path, "missing")


def _warn_if_missing(root: Path, relative_path: str, detail: str) -> CheckResult:
    path = root / relative_path
    if path.exists():
        return CheckResult("PASS", relative_path)
    return CheckResult("WARN", relative_path, detail)


def _tracked_files(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _check_forbidden_tracked(root: Path) -> CheckResult:
    forbidden = [path for path in _tracked_files(root) if FORBIDDEN_TRACKED_RE.search(path)]
    if forbidden:
        return CheckResult("FAIL", "git tracked artifacts", ", ".join(forbidden[:10]))
    return CheckResult("PASS", "git tracked artifacts", "no raw/model/heavy artifacts tracked")


def _check_readme_identity(root: Path) -> CheckResult:
    readme = root / "README.md"
    if not readme.exists():
        return CheckResult("FAIL", "README identity", "README.md missing")
    text = readme.read_text(encoding="utf-8")
    if "Full name: TODO" in text or "Group: TODO" in text:
        return CheckResult("WARN", "README identity", "fill full name and group")
    return CheckResult("PASS", "README identity")


def _check_docker_cli() -> CheckResult:
    if shutil.which("docker"):
        return CheckResult("PASS", "Docker CLI")
    return CheckResult("WARN", "Docker CLI", "not installed here; verify compose on Docker host")


def run_checks(root: Path) -> list[CheckResult]:
    required_paths = [
        "README.md",
        "model.py",
        "pyproject.toml",
        "Dockerfile",
        "docker-compose.yaml",
        ".dockerignore",
        "api/server.py",
        "web/index.html",
        "scripts/evaluate.py",
        "scripts/validate_submission.py",
        "docs/streaming-demo.mp4",
        "dist/akkadian_streaming_translator-0.1.0-py3-none-any.whl",
    ]
    results = [_path_exists(root, path) for path in required_paths]
    results.extend(
        [
            _warn_if_missing(
                root,
                "data/raw/sample_submission.csv",
                "download Kaggle CSV before final local validation",
            ),
            _warn_if_missing(root, "data/results.csv", "generate Kaggle predictions before upload"),
            _warn_if_missing(root, "docs/kaggle-leaderboard.png", "add leaderboard screenshot"),
            _check_forbidden_tracked(root),
            _check_readme_identity(root),
            _check_docker_cli(),
        ]
    )
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = run_checks(root)
    for result in results:
        suffix = f" - {result.detail}" if result.detail else ""
        print(f"{result.status}: {result.name}{suffix}")

    if any(result.status == "FAIL" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

