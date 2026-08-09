from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.fetch_nasdaq_itch_public import _download, _record_sha256


class FakeResponse:
    def __init__(self, payload: bytes, status: int, headers: dict[str, str]):
        self._stream = BytesIO(payload)
        self.status = status
        self.headers = headers

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_download_resumes_existing_partial_file(tmp_path: Path):
    output = tmp_path / "sample.gz"
    partial = tmp_path / "sample.gz.part"
    partial.write_bytes(b"abc")
    response = FakeResponse(
        b"def",
        status=206,
        headers={"content-length": "3", "content-range": "bytes 3-5/6"},
    )

    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        _download("https://example.test/sample.gz", output, expected_size=6)

    assert output.read_bytes() == b"abcdef"
    assert not partial.exists()
    assert urlopen.call_args.args[0].headers["Range"] == "bytes=3-"


def test_download_restarts_when_server_ignores_range(tmp_path: Path):
    output = tmp_path / "sample.gz"
    partial = tmp_path / "sample.gz.part"
    partial.write_bytes(b"stale")
    response = FakeResponse(b"abcdef", status=200, headers={"content-length": "6"})

    with patch("urllib.request.urlopen", return_value=response):
        _download("https://example.test/sample.gz", output, expected_size=6)

    assert output.read_bytes() == b"abcdef"


def test_download_reconnects_when_range_response_is_segmented(tmp_path: Path):
    output = tmp_path / "sample.gz"
    partial = tmp_path / "sample.gz.part"
    partial.write_bytes(b"ab")
    responses = [
        FakeResponse(
            b"cd",
            status=206,
            headers={"content-length": "2", "content-range": "bytes 2-3/6"},
        ),
        FakeResponse(
            b"ef",
            status=206,
            headers={"content-length": "2", "content-range": "bytes 4-5/6"},
        ),
    ]

    with patch("urllib.request.urlopen", side_effect=responses) as urlopen:
        _download("https://example.test/sample.gz", output, expected_size=6)

    assert output.read_bytes() == b"abcdef"
    assert urlopen.call_count == 2
    assert urlopen.call_args_list[0].args[0].headers["Range"] == "bytes=2-"
    assert urlopen.call_args_list[1].args[0].headers["Range"] == "bytes=4-"


def test_incomplete_download_keeps_partial_file(tmp_path: Path):
    output = tmp_path / "sample.gz"
    partial = tmp_path / "sample.gz.part"
    response = FakeResponse(b"abc", status=200, headers={"content-length": "3"})

    with patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(RuntimeError, match="Incomplete download retained"):
            _download("https://example.test/sample.gz", output, expected_size=6)

    assert not output.exists()
    assert partial.read_bytes() == b"abc"


def test_record_sha256_writes_standard_sidecar(tmp_path: Path):
    path = tmp_path / "sample.gz"
    path.write_bytes(b"abc")

    digest = _record_sha256(path)

    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert (tmp_path / "sample.gz.sha256").read_text() == f"{digest}  sample.gz\n"
