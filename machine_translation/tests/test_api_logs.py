from pathlib import Path

from fastapi.testclient import TestClient


class _FakeResponse:
    def __enter__(self):
        return iter(
            [
                b'data: {"token": {"text": "To ", "special": false}}\n',
                b'data: {"token": {"text": "</s>", "special": true}}\n',
                b'data: {"token": {"text": "king", "special": false}}\n',
                b"data: [DONE]\n",
            ]
        )

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_logs_are_viewable_and_downloadable(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    from api import server

    server.log_path = Path("data/log_file.log")
    server.log_path.parent.mkdir(parents=True, exist_ok=True)
    server.log_path.write_text("hello log\n", encoding="utf-8")

    client = TestClient(server.app)

    view_response = client.get("/logs")
    assert view_response.status_code == 200
    assert view_response.text == "hello log\n"

    download_response = client.get("/logs/download")
    assert download_response.status_code == 200
    assert download_response.text == "hello log\n"
    assert "log_file.log" in download_response.headers["content-disposition"]


def test_tgi_stream_yields_non_special_tokens(monkeypatch) -> None:
    from api import server

    monkeypatch.setattr(server, "tgi_url", "http://tgi:80")
    monkeypatch.setattr(server, "urlopen", lambda request, timeout: _FakeResponse())

    assert list(server._tgi_stream("šarrum")) == ["To ", "king"]
