# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import Polygon, shape


@dataclass
class AlertText:
    """The human-readable fields of one CAP ``<info>`` block in one language."""
    event: str
    description: str
    area_desc: str


@dataclass
class AlertEntry:
    """Parsed CAP alert with optional polygon geometry.

    ``event``/``description``/``area_desc`` are in ``language`` (the first
    language the CAP message lists).  ``translations`` holds the issuer's own
    parallel ``<info>`` blocks in other languages, keyed by lower-cased CAP
    language tag (e.g. ``"en-gb"``).
    """
    source_id: str
    event: str
    description: str
    severity: str
    effective: str
    expires: str
    area_desc: str
    url: str
    polygon: Optional[Polygon] = None
    language: str = ""
    translations: dict[str, AlertText] = field(default_factory=dict)

    def localized(self, lang: Optional[str]) -> tuple[AlertText, str]:
        """Best text for a requested language, plus the tag actually used."""
        primary = AlertText(self.event, self.description, self.area_desc)
        if not lang:
            return primary, self.language
        texts = {self.language: primary, **self.translations}
        tag = pick_language(list(texts), lang)
        if tag is None:
            return primary, self.language
        return texts[tag], tag


def _base(tag: str) -> str:
    return tag.lower().replace("_", "-").split("-")[0]


def pick_language(available: list[str], wanted: str) -> Optional[str]:
    """Choose among ``available`` CAP language tags for a requested one.

    Exact tag, then same base language (``de`` ↔ ``de-DE``), then English.
    ``None`` means nothing fits and the caller should use the original.
    The first entry in ``available`` wins ties, so callers list the
    original language first.
    """
    wanted = wanted.lower().replace("_", "-")
    for tag in available:
        if tag.lower() == wanted:
            return tag
    for base in (_base(wanted), "en"):
        for tag in available:
            if _base(tag) == base:
                return tag
    return None


class AlertsStore:
    """Thread-safe in-memory store of active weather alerts."""

    def __init__(self):
        self._lock = threading.RLock()
        self._alerts: list[AlertEntry] = []
        self.last_updated: float = 0.0
        self._fetch_success: bool = False

    @property
    def alerts(self) -> list[AlertEntry]:
        with self._lock:
            return list(self._alerts)

    def replace_all(self, alerts: list[AlertEntry]) -> None:
        with self._lock:
            self._alerts = alerts
            self.last_updated = time.time()
            self._fetch_success = True

    def mark_failed(self) -> None:
        with self._lock:
            self._fetch_success = False

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._alerts)

    @property
    def fetch_success(self) -> bool:
        with self._lock:
            return self._fetch_success

    def __getstate__(self) -> dict:
        """Serialize state for the multi-worker snapshot.

        Polygons are emitted as GeoJSON via shapely's ``__geo_interface__``
        so the whole payload is JSON-friendly.  Render-only workers
        rebuild Polygon objects in ``__setstate__``.
        """
        with self._lock:
            return {
                "last_updated": self.last_updated,
                "fetch_success": self._fetch_success,
                "alerts": [
                    {
                        "source_id": a.source_id,
                        "event": a.event,
                        "description": a.description,
                        "severity": a.severity,
                        "effective": a.effective,
                        "expires": a.expires,
                        "area_desc": a.area_desc,
                        "url": a.url,
                        "language": a.language,
                        "translations": {
                            tag: {
                                "event": t.event,
                                "description": t.description,
                                "area_desc": t.area_desc,
                            }
                            for tag, t in a.translations.items()
                        },
                        "polygon": (
                            a.polygon.__geo_interface__ if a.polygon is not None else None
                        ),
                    }
                    for a in self._alerts
                ],
            }

    def __setstate__(self, state: dict) -> None:
        """Restore from a snapshot written by ``__getstate__``."""
        if not hasattr(self, "_lock"):
            # __setstate__ may run before __init__ (pickle compat).
            self._lock = threading.RLock()
        with self._lock:
            self._alerts = [
                AlertEntry(
                    source_id=a["source_id"],
                    event=a["event"],
                    description=a["description"],
                    severity=a["severity"],
                    effective=a["effective"],
                    expires=a["expires"],
                    area_desc=a["area_desc"],
                    url=a["url"],
                    polygon=shape(a["polygon"]) if a.get("polygon") else None,
                    language=a.get("language", ""),
                    translations={
                        tag: AlertText(**t)
                        for tag, t in (a.get("translations") or {}).items()
                    },
                )
                for a in state.get("alerts", [])
            ]
            self.last_updated = state.get("last_updated", 0.0)
            self._fetch_success = state.get("fetch_success", False)
