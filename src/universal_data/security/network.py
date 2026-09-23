"""URL validation for the API ingestion layer.

Any URL the toolkit fetches may come from a configuration file, so the request
layer is a server-side request forgery (SSRF) surface.  Every URL - including
every redirect hop - is checked here before a socket is opened.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

from universal_data.core.exceptions import SecurityError

_ALLOWED_SCHEMES = ("http", "https")
_BLOCKED_HOSTNAMES = frozenset({"metadata.google.internal", "metadata", "instance-data"})


@dataclass(frozen=True)
class URLPolicy:
    """Constraints applied to outbound HTTP requests."""

    allow_http: bool = False
    allow_private_networks: bool = False
    allowed_hosts: frozenset[str] = field(default_factory=frozenset)
    max_redirects: int = 5

    def _check_scheme(self, scheme: str, url: str) -> None:
        if scheme not in _ALLOWED_SCHEMES:
            raise SecurityError("Only http and https URLs are supported", url=url, scheme=scheme)
        if scheme == "http" and not self.allow_http:
            raise SecurityError(
                "Plain HTTP is disabled; use https or enable allow_http explicitly", url=url
            )

    def _check_host_allowlist(self, host: str, url: str) -> None:
        if not self.allowed_hosts:
            return
        host = host.lower()
        for allowed in self.allowed_hosts:
            allowed = allowed.lower()
            if host == allowed or host.endswith(f".{allowed}"):
                return
        raise SecurityError("Host is not in the allow-list", url=url, host=host)

    def _resolve_addresses(
        self, host: str, port: int, url: str
    ) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        try:
            literal = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            pass
        else:
            return [literal]
        try:
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise SecurityError("Hostname could not be resolved", url=url, host=host) from exc
        addresses = []
        for info in infos:
            sockaddr = info[4]
            try:
                addresses.append(ipaddress.ip_address(sockaddr[0]))
            except ValueError:  # pragma: no cover - defensive
                continue
        if not addresses:
            raise SecurityError("Hostname resolved to no usable address", url=url, host=host)
        return addresses

    def validate(self, url: str) -> str:
        """Return the normalised URL or raise :class:`SecurityError`."""
        if not url or not url.strip():
            raise SecurityError("Empty URL")
        parsed = urlparse(url.strip())
        self._check_scheme(parsed.scheme.lower(), url)
        host = parsed.hostname
        if not host:
            raise SecurityError("URL has no host component", url=url)
        if host.lower() in _BLOCKED_HOSTNAMES:
            raise SecurityError("Access to cloud metadata endpoints is blocked", url=url)
        self._check_host_allowlist(host, url)

        if not self.allow_private_networks:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            for address in self._resolve_addresses(host, port, url):
                if (
                    address.is_private
                    or address.is_loopback
                    or address.is_link_local
                    or address.is_reserved
                    or address.is_multicast
                    or address.is_unspecified
                ):
                    raise SecurityError(
                        "URL resolves to a non-public address; enable allow_private_networks "
                        "if this is intended",
                        url=url,
                        address=str(address),
                    )
        return urlunparse(parsed)


def validate_url(url: str, policy: URLPolicy | None = None) -> str:
    return (policy or URLPolicy()).validate(url)
