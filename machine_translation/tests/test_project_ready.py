from scripts.check_project_ready import FORBIDDEN_TRACKED_RE, _check_readme_identity


def test_forbidden_tracked_regex_matches_heavy_artifacts() -> None:
    forbidden = [
        "machine_translation/model/baseline/model.safetensors",
        "machine_translation/data/raw/train.csv",
        "machine_translation/data/results.csv",
        "machine_translation/wandb/run/run.wandb",
    ]

    assert all(FORBIDDEN_TRACKED_RE.search(path) for path in forbidden)


def test_forbidden_tracked_regex_allows_project_files() -> None:
    allowed = [
        "machine_translation/model.py",
        "machine_translation/data/.gitkeep",
        "machine_translation/docs/streaming-demo.mp4",
    ]

    assert not any(FORBIDDEN_TRACKED_RE.search(path) for path in allowed)


def test_readme_identity_warns_on_todo(tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("- Full name: TODO\n- Group: TODO\n", encoding="utf-8")

    result = _check_readme_identity(tmp_path)

    assert result.status == "WARN"

