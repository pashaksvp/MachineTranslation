import pandas as pd

from scripts.validate_submission import validate_submission


def test_validate_submission_accepts_matching_file(tmp_path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "results.csv"
    sample = pd.DataFrame({"id": [0, 1], "translation": ["a", "b"]})
    sample.to_csv(sample_path, index=False)
    sample.to_csv(submission_path, index=False)

    assert validate_submission(sample_path, submission_path) == []


def test_validate_submission_rejects_wrong_id_order(tmp_path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "results.csv"
    pd.DataFrame({"id": [0, 1], "translation": ["a", "b"]}).to_csv(sample_path, index=False)
    pd.DataFrame({"id": [1, 0], "translation": ["b", "a"]}).to_csv(submission_path, index=False)

    assert validate_submission(sample_path, submission_path) == [
        "id values/order do not match sample_submission"
    ]

