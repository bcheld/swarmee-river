from __future__ import annotations

import io

from tools.s3_browser import s3_browser


def _text(result: dict[str, object]) -> str:
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            value = first.get("text")
            if isinstance(value, str):
                return value
    return ""


def test_s3_browser_list_objects_requires_bucket() -> None:
    result = s3_browser(action="list_objects")

    assert result.get("status") == "error"
    assert "bucket is required" in _text(result).lower()


def test_s3_browser_read_binary_returns_notice(monkeypatch) -> None:
    class FakeS3:
        def head_object(self, **kwargs):
            return {"ContentLength": 4, "ContentType": "application/octet-stream"}

        def get_object(self, **kwargs):
            return {"Body": io.BytesIO(b"\x00\x01\x02\x03")}

    monkeypatch.setattr("tools.s3_browser._s3_client", lambda: FakeS3())

    result = s3_browser(action="read", bucket="bucket", key="file.bin")

    assert result.get("status") == "success"
    assert "[Binary file: 4 bytes" in _text(result)


def test_s3_browser_read_csv_renders_markdown_table(monkeypatch) -> None:
    class FakeS3:
        def head_object(self, **kwargs):
            return {"ContentLength": 30, "ContentType": "text/csv"}

        def get_object(self, **kwargs):
            payload = b"name,age\nalice,30\nbob,31\n"
            return {"Body": io.BytesIO(payload)}

    monkeypatch.setattr("tools.s3_browser._s3_client", lambda: FakeS3())

    result = s3_browser(action="read", bucket="bucket", key="people.csv")
    text = _text(result)

    assert result.get("status") == "success"
    assert "| name | age |" in text
    assert "| alice | 30 |" in text


def test_s3_browser_search_lists_matching_keys(monkeypatch) -> None:
    class FakeS3:
        def list_objects_v2(self, **kwargs):
            return {
                "Contents": [
                    {"Key": "logs/2026-01-01.log", "Size": 1000},
                    {"Key": "logs/2026-01-02.log", "Size": 2000},
                ],
                "IsTruncated": False,
            }

    monkeypatch.setattr("tools.s3_browser._s3_client", lambda: FakeS3())

    result = s3_browser(action="search", bucket="bucket", prefix="logs/", max_results=10)

    assert result.get("status") == "success"
    text = _text(result)
    assert "logs/2026-01-01.log" in text
    assert "logs/2026-01-02.log" in text
