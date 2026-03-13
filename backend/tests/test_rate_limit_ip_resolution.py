import sys
import unittest
from pathlib import Path

from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import _client_ip


def _build_request(*, client_host: str, headers: dict[str, str] | None = None) -> Request:
    raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": raw_headers,
        "client": (client_host, 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


class RateLimitClientIpResolutionTests(unittest.TestCase):
    def test_public_clients_cannot_spoof_forwarded_ip_headers(self) -> None:
        request = _build_request(
            client_host="8.8.4.4",
            headers={"x-forwarded-for": "9.9.9.9"},
        )

        self.assertEqual(_client_ip(request), "8.8.4.4")

    def test_private_proxy_prefers_leftmost_public_forwarded_ip(self) -> None:
        request = _build_request(
            client_host="10.0.0.8",
            headers={"x-forwarded-for": "9.9.9.9, 1.1.1.1"},
        )

        self.assertEqual(_client_ip(request), "9.9.9.9")

    def test_private_proxy_falls_back_to_leftmost_parseable_forwarded_ip(self) -> None:
        request = _build_request(
            client_host="10.0.0.8",
            headers={"x-forwarded-for": "10.1.2.3, 172.16.0.9"},
        )

        self.assertEqual(_client_ip(request), "10.1.2.3")

    def test_private_proxy_uses_single_proxy_headers_when_available(self) -> None:
        request = _build_request(
            client_host="10.0.0.8",
            headers={"cf-connecting-ip": "1.0.0.1"},
        )

        self.assertEqual(_client_ip(request), "1.0.0.1")


if __name__ == "__main__":
    unittest.main()
