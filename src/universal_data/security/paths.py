"""Filesystem safety helpers.

The toolkit reads and writes paths that often come from YAML configuration or
from CLI arguments, so every path goes through :class:`PathPolicy` before it is
touched.  The policy handles the three problems we actually care about:

* path traversal out of a workspace directory (``../../etc/passwd``),
* unexpected file types (a pipeline configured to write ``.parquet`` should not
  suddenly overwrite ``.bashrc``),
* symlinks pointing outside the workspace.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from universal_data.core.exceptions import SecurityError
from universal_data.core.types import PathLike

# Device names that Windows still resolves even inside an ordinary directory.
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_UNSAFE_NAME_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]')


def sanitize_filename(name: str, *, fallback: str = "file") -> str:
    """Return *name* reduced to a safe single path component.

    Used for filenames derived from user input (report names, table names,
    downloaded resources).  Directory separators are removed rather than
    escaped, so the result can never point at a parent directory.
    """
    basename = re.split(r"[\\/]", str(name))[-1]
    cleaned = _UNSAFE_NAME_CHARS.sub("_", basename).strip().strip(".")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_{2,}", "_", cleaned)
    if not cleaned:
        return fallback
    stem, dot, suffix = cleaned.partition(".")
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"{stem}_"
    return f"{stem}{dot}{suffix}"[:255]


@dataclass(frozen=True)
class PathPolicy:
    """Rules applied to every path the toolkit opens.

    ``roots`` empty means "no confinement" which is the right default for an
    interactive CLI: the user already chose the file.  Config-driven pipelines
    build a confined policy rooted at the configuration directory instead.
    """

    roots: tuple[Path, ...] = ()
    allowed_suffixes: frozenset[str] | None = None
    allow_symlinks: bool = False
    max_bytes: int | None = None

    @classmethod
    def confined_to(cls, root: PathLike, **kwargs: object) -> PathPolicy:
        resolved = Path(root).expanduser().resolve()
        return cls(roots=(resolved,), **kwargs)  # type: ignore[arg-type]

    def _normalize(self, path: PathLike) -> Path:
        raw = os.fspath(path)
        if not raw or not raw.strip():
            raise SecurityError("Empty path is not allowed")
        if "\x00" in raw:
            raise SecurityError("Path contains a NUL byte", path=raw.replace("\x00", "\\0"))
        candidate = Path(raw).expanduser()
        if self.roots and not candidate.is_absolute():
            candidate = self.roots[0] / candidate
        return candidate

    def _check_confinement(self, resolved: Path, original: Path) -> None:
        if not self.roots:
            return
        for root in self.roots:
            if resolved == root or resolved.is_relative_to(root):
                return
        raise SecurityError(
            "Path escapes the allowed workspace",
            path=str(original),
            roots=[str(r) for r in self.roots],
        )

    @staticmethod
    def _check_reserved_name(resolved: Path) -> None:
        """Reject Windows device names.

        ``NUL``, ``CON`` and friends resolve to a device even inside an ordinary
        directory, so an output path of ``<workspace>/NUL`` silently discards
        everything written to it instead of failing.
        """
        stem = resolved.name.split(".")[0].upper()
        if stem in _WINDOWS_RESERVED:
            raise SecurityError(
                "Path refers to a reserved device name", path=str(resolved), name=stem
            )

    def _check_suffix(self, resolved: Path) -> None:
        if self.allowed_suffixes is None:
            return
        suffix = resolved.suffix.lower()
        if suffix not in self.allowed_suffixes:
            raise SecurityError(
                "File extension is not permitted",
                path=str(resolved),
                suffix=suffix or "<none>",
                allowed=sorted(self.allowed_suffixes),
            )

    def _check_symlink(self, candidate: Path) -> None:
        if self.allow_symlinks:
            return
        node = candidate
        seen: set[Path] = set()
        while node not in seen:
            seen.add(node)
            if node.is_symlink():
                raise SecurityError("Symlinked paths are not permitted", path=str(candidate))
            if node.parent == node:
                break
            node = node.parent

    def resolve_input(self, path: PathLike, *, must_exist: bool = True) -> Path:
        """Validate a path we intend to read from."""
        candidate = self._normalize(path)
        self._check_symlink(candidate)
        resolved = candidate.resolve()
        self._check_confinement(resolved, candidate)
        self._check_reserved_name(resolved)
        self._check_suffix(resolved)
        if must_exist:
            if not resolved.exists():
                raise SecurityError("Input path does not exist", path=str(resolved))
            if not resolved.is_file():
                raise SecurityError("Input path is not a regular file", path=str(resolved))
            if self.max_bytes is not None:
                size = resolved.stat().st_size
                if size > self.max_bytes:
                    raise SecurityError(
                        "Input file exceeds the configured size limit",
                        path=str(resolved),
                        size=size,
                        limit=self.max_bytes,
                    )
        return resolved

    def resolve_output(self, path: PathLike, *, create_parents: bool = True) -> Path:
        """Validate a path we intend to write to, creating parent folders."""
        candidate = self._normalize(path)
        self._check_symlink(candidate)
        resolved = candidate.resolve()
        self._check_confinement(resolved, candidate)
        self._check_reserved_name(resolved)
        self._check_suffix(resolved)
        if resolved.exists() and resolved.is_dir():
            raise SecurityError("Output path is a directory", path=str(resolved))
        if create_parents:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        elif not resolved.parent.exists():
            raise SecurityError("Output directory does not exist", path=str(resolved.parent))
        return resolved


_DEFAULT_POLICY = PathPolicy()


def default_policy() -> PathPolicy:
    """Policy used when a caller does not supply one."""
    return _DEFAULT_POLICY


def resolve_input(path: PathLike, policy: PathPolicy | None = None, **kwargs: bool) -> Path:
    return (policy or _DEFAULT_POLICY).resolve_input(path, **kwargs)


def resolve_output(path: PathLike, policy: PathPolicy | None = None, **kwargs: bool) -> Path:
    return (policy or _DEFAULT_POLICY).resolve_output(path, **kwargs)


@dataclass
class TempWorkspace:
    """Small helper used by tests and the demo to build a confined policy."""

    root: Path
    policy: PathPolicy = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = PathPolicy.confined_to(self.root)
