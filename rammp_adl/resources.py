"""Atomic exclusive resource ownership for the event-driven DAG executor."""
from __future__ import annotations

import asyncio
from collections.abc import Iterable


class ResourceError(RuntimeError):
    pass


class ResourceManager:
    """Single event-loop owner. No partial acquisition and no hidden release."""

    def __init__(self, allowed: Iterable[str]):
        self.allowed = frozenset(allowed)
        self._owners: dict[str, str] = {}
        self.changed = asyncio.Event()

    @property
    def owners(self) -> dict[str, str]:
        return dict(self._owners)

    def try_acquire(self, owner: str, claims: Iterable[str]) -> bool:
        claims = frozenset(claims)
        if not owner or not claims <= self.allowed:
            raise ResourceError("Unknown resource or empty owner")
        if owner in self._owners.values():
            raise ResourceError("Owner already holds a reservation")
        if any(resource in self._owners for resource in claims):
            return False
        self._owners.update({resource: owner for resource in claims})
        return True

    def release(self, owner: str) -> None:
        held = [resource for resource, holder in self._owners.items() if holder == owner]
        for resource in held:
            del self._owners[resource]
        self.changed.set()

    def assert_owned(self, owner: str, claims: Iterable[str]) -> None:
        if any(self._owners.get(resource) != owner for resource in claims):
            raise ResourceError("Command lacks its complete reservation")

