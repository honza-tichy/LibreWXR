# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import json
import logging
import os
from pathlib import Path

import numpy as np

from librewxr.data.regions import RegionDef
from librewxr.data.store import RadarFrame

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DTYPE = np.uint8


class RadarFrameCache:
    """Persistent disk cache for radar frame region arrays.

    Each (timestamp, region) is stored as its own raw uint8 file written
    atomically (write-to-tmp, then os.replace). A ``metadata.json`` file
    records the schema version and per-region (height, width) — on load
    any region whose shape no longer matches its current ``RegionDef``
    is silently dropped, so a code-side region resize can't restore
    broken data.

    Cleanup is driven by the in-memory ``FrameStore`` ring buffer:
    anything that's been evicted from memory is also evicted from disk.
    """

    def __init__(self, cache_dir: Path):
        self._dir = cache_dir / "radar"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self._dir / "metadata.json"

    def _file_path(self, unix_ts: int, region_name: str) -> Path:
        return self._dir / f"radar_{unix_ts}_{region_name}.dat"

    def has(self, unix_ts: int, region_name: str) -> bool:
        return self._file_path(unix_ts, region_name).exists()

    def write_frame(self, frame: RadarFrame) -> None:
        """Write every region in a frame to disk atomically."""
        for region_name, data in frame.regions.items():
            self._write_region(frame.timestamp, region_name, np.asarray(data))

    def _write_region(
        self, unix_ts: int, region_name: str, data: np.ndarray
    ) -> None:
        if data.dtype != DTYPE:
            data = data.astype(DTYPE, copy=False)
        final = self._file_path(unix_ts, region_name)
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=DTYPE, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        os.replace(tmp, final)

    def load_frames(
        self, regions: dict[str, RegionDef]
    ) -> list[RadarFrame]:
        """Load all cached frames, validated against current RegionDef shapes.

        Only regions listed in ``regions`` are restored, and only if the
        cached file's shape matches the current ``RegionDef``. Any
        timestamp with at least one valid region is returned as a frame
        with whatever subset of regions survived validation.

        Self-healing: timestamps discovered by scanning ``radar_*_*.dat``
        files are unioned with whatever ``metadata.json`` declares, so a
        crash mid-backfill that leaves the metadata stale doesn't orphan
        valid frame files. If a per-region shape isn't recorded in
        metadata, ``np.memmap`` validates by file size at read time.
        """
        meta = self._load_metadata() or {}

        # An explicit schema version mismatch is fatal; missing metadata
        # entirely is fine — disk scan + per-file size check still works.
        declared_version = meta.get("schema_version")
        if declared_version is not None and declared_version != SCHEMA_VERSION:
            logger.info(
                "Radar cache schema_version mismatch (have=%s, expect=%d); "
                "ignoring cache",
                declared_version, SCHEMA_VERSION,
            )
            return []

        cached_shapes = meta.get("regions", {})
        metadata_timestamps = set(meta.get("timestamps", []))
        disk_timestamps = self._scan_timestamps()
        orphans = disk_timestamps - metadata_timestamps
        if orphans:
            logger.info(
                "Radar cache: %d timestamp(s) on disk missing from metadata; "
                "including them",
                len(orphans),
            )
        timestamps = sorted(metadata_timestamps | disk_timestamps)

        # Absent in metadata written before this key existed; empty then
        # means "nothing known to be carried", which is the right reading.
        carried_meta = meta.get("carried_from", {}) or {}

        frames: list[RadarFrame] = []
        for ts in timestamps:
            regions_data: dict[str, np.ndarray] = {}
            for name, region in regions.items():
                expected_shape = (region.height, region.width)
                cached_meta = cached_shapes.get(name)
                if cached_meta is not None:
                    cached_shape = tuple(cached_meta.get("shape", []))
                    if cached_shape != expected_shape:
                        # Region was reshaped in code — drop the stale file.
                        self._file_path(ts, name).unlink(missing_ok=True)
                        continue
                # Either the region's shape is recorded and matches, or
                # metadata doesn't know about it — fall through and let
                # _read_region's memmap fail by file size if it's wrong.
                arr = self._read_region(ts, name, expected_shape)
                if arr is not None:
                    regions_data[name] = arr
            if regions_data:
                # Filter to regions that actually survived shape
                # validation above — provenance for a dropped region
                # would describe an array that is not there.
                entry = carried_meta.get(str(ts), {})
                frames.append(RadarFrame(
                    timestamp=ts,
                    regions=regions_data,
                    carried_from={
                        name: int(origin)
                        for name, origin in entry.items()
                        if name in regions_data
                    },
                ))
        return frames

    def _scan_timestamps(self) -> set[int]:
        """Enumerate timestamps from radar_<ts>_<region>.dat filenames."""
        result: set[int] = set()
        for path in self._dir.glob("radar_*.dat"):
            stem_parts = path.stem.split("_")
            if len(stem_parts) >= 3:
                try:
                    result.add(int(stem_parts[1]))
                except ValueError:
                    pass
        return result

    def _read_region(
        self, unix_ts: int, region_name: str, shape: tuple[int, int]
    ) -> np.ndarray | None:
        path = self._file_path(unix_ts, region_name)
        if not path.exists():
            return None
        try:
            return np.memmap(path, dtype=DTYPE, mode="r", shape=shape)
        except Exception:
            logger.warning("Failed to memmap %s, removing", path)
            path.unlink(missing_ok=True)
            return None

    def save_metadata(
        self,
        regions: dict[str, RegionDef],
        timestamps: list[int],
        carried_from: dict[int, dict[str, int]] | None = None,
    ) -> None:
        """Atomically write metadata JSON with current shapes and timestamps.

        ``carried_from`` records which cached regions are carry-forward
        copies and of what, so a restart does not silently promote stale
        data back to "fresh".  Optional, and SCHEMA_VERSION is
        deliberately NOT bumped for it: ``load_frames`` treats a version
        mismatch as fatal and returns nothing, so a bump would throw away
        the whole cached window on upgrade and cold-start the pipeline —
        a real outage in exchange for a version number.  The change is
        additive both ways: new code reading old metadata sees no key and
        reads it as all-fresh, old code ignores the key it doesn't know.
        """
        payload = {
            "schema_version": SCHEMA_VERSION,
            "regions": {
                name: {"shape": [r.height, r.width], "dtype": "uint8"}
                for name, r in regions.items()
            },
            "timestamps": sorted(timestamps),
            # Sparse — only regions actually carried appear.  JSON object
            # keys must be strings, hence str(ts).
            "carried_from": {
                str(ts): entry
                for ts, entry in (carried_from or {}).items()
                if entry
            },
        }
        tmp = self._metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self._metadata_path)

    def _load_metadata(self) -> dict | None:
        if not self._metadata_path.exists():
            return None
        try:
            return json.loads(self._metadata_path.read_text())
        except Exception:
            logger.warning("Corrupt radar metadata.json, ignoring")
            return None

    def stats(self) -> dict:
        """Return a snapshot of disk-cache state for the /health endpoint."""
        files = list(self._dir.glob("radar_*.dat"))
        total_bytes = 0
        timestamps: set[int] = set()
        for path in files:
            try:
                total_bytes += path.stat().st_size
            except OSError:
                continue
            stem_parts = path.stem.split("_")
            if len(stem_parts) >= 3:
                try:
                    timestamps.add(int(stem_parts[1]))
                except ValueError:
                    pass
        return {
            "files": len(files),
            "used_mb": round(total_bytes / (1024 * 1024), 1),
            "oldest_ts": min(timestamps) if timestamps else None,
            "newest_ts": max(timestamps) if timestamps else None,
        }

    def cleanup(self, active_timestamps: list[int]) -> None:
        """Remove .dat files for timestamps no longer in the active set."""
        active = set(active_timestamps)
        removed = 0
        for path in self._dir.glob("radar_*.dat"):
            stem_parts = path.stem.split("_")
            if len(stem_parts) < 3:
                continue
            try:
                ts = int(stem_parts[1])
            except ValueError:
                continue
            if ts not in active:
                path.unlink(missing_ok=True)
                removed += 1

        for path in self._dir.glob("*.tmp"):
            path.unlink(missing_ok=True)

        if removed:
            logger.info("Radar cache cleanup: removed %d old files", removed)
