from __future__ import annotations

import unittest

from starlette.requests import Request

import api.support as support_module


def make_request(*, client_host: str, headers: dict[str, str] | None = None) -> Request:
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in (headers or {}).items()],
        "client": (client_host, 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    })


class RequestClientIpTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_data = dict(support_module.config.data)

    def tearDown(self) -> None:
        support_module.config.data = self._old_data

    def test_forwarded_header_is_ignored_by_default(self) -> None:
        support_module.config.data.update({
            "trust_proxy_headers": False,
            "trusted_proxy_ips": [],
        })
        request = make_request(
            client_host="198.51.100.20",
            headers={"x-forwarded-for": "203.0.113.10"},
        )

        self.assertEqual(support_module.request_client_ip(request), "198.51.100.20")

    def test_forwarded_header_is_used_for_trusted_proxy(self) -> None:
        support_module.config.data.update({
            "trust_proxy_headers": True,
            "trusted_proxy_ips": ["198.51.100.0/24"],
        })
        request = make_request(
            client_host="198.51.100.20",
            headers={"x-forwarded-for": "203.0.113.10, 198.51.100.20"},
        )

        self.assertEqual(support_module.request_client_ip(request), "203.0.113.10")

    def test_forwarded_header_uses_rightmost_untrusted_ip_from_trusted_proxy(self) -> None:
        support_module.config.data.update({
            "trust_proxy_headers": True,
            "trusted_proxy_ips": ["198.51.100.0/24"],
        })
        request = make_request(
            client_host="198.51.100.20",
            headers={"x-forwarded-for": "203.0.113.200, 203.0.113.10"},
        )

        self.assertEqual(support_module.request_client_ip(request), "203.0.113.10")

    def test_forwarded_header_skips_trusted_proxy_chain(self) -> None:
        support_module.config.data.update({
            "trust_proxy_headers": True,
            "trusted_proxy_ips": ["198.51.100.0/24"],
        })
        request = make_request(
            client_host="198.51.100.20",
            headers={"x-forwarded-for": "203.0.113.10, 198.51.100.30"},
        )

        self.assertEqual(support_module.request_client_ip(request), "203.0.113.10")

    def test_forwarded_header_is_ignored_for_untrusted_proxy(self) -> None:
        support_module.config.data.update({
            "trust_proxy_headers": True,
            "trusted_proxy_ips": ["192.0.2.0/24"],
        })
        request = make_request(
            client_host="198.51.100.20",
            headers={"x-forwarded-for": "203.0.113.10"},
        )

        self.assertEqual(support_module.request_client_ip(request), "198.51.100.20")

    def test_forwarded_header_is_ignored_when_trusted_proxy_list_is_empty(self) -> None:
        support_module.config.data.update({
            "trust_proxy_headers": True,
            "trusted_proxy_ips": [],
        })
        request = make_request(
            client_host="198.51.100.20",
            headers={"x-forwarded-for": "203.0.113.10"},
        )

        self.assertEqual(support_module.request_client_ip(request), "198.51.100.20")

    def test_forwarded_header_is_ignored_when_trusted_proxy_config_is_invalid(self) -> None:
        support_module.config.data.update({
            "trust_proxy_headers": True,
            "trusted_proxy_ips": ["not-an-ip"],
        })
        request = make_request(
            client_host="198.51.100.20",
            headers={"x-forwarded-for": "203.0.113.10"},
        )

        self.assertEqual(support_module.request_client_ip(request), "198.51.100.20")


if __name__ == "__main__":
    unittest.main()
