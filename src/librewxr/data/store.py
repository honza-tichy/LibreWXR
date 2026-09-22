# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RadarFrame:
    timestamp: int  # Unix timestamp
    regions: dict[str, np.ndarray] = field(default_factory=dict)
    # region name -> unix timestamp of the observation the array in
    # ``regions`` actually came from.  Present ONLY when that differs
    # from ``timestamp``, i.e. the region was carried forward by the
    # pass in data/fetcher.py.
    #
    # The three states a client cares about are read off this plus
    # ``regions``, and only ONE of them is ever written:
    #
    #   fresh    - name in regions, no carried_from entry
    #   carried  - name in regions, carried_from[name] = origin
    #   absent   - name not in regions at all
    #
    # Deriving "absent" rather than writing it is deliberate.  A region
    # can go missing down paths that never reach a writer at all — most
    # sharply when fetch_cycle_timeout abandons the second priority wave,
    # so that wave's regions are touched by no code that cycle.  Anything
    # stamped by a writer would call those fresh.
    #
    # Empty is also the right reading of every frame written before this
    # field existed, in memory, on disk or in state.json.
    carried_from: dict[str, int] = field(default_factory=dict)


class FrameStore:
    """Frame store backed by memory-mapped files.

    Region arrays are written to disk and accessed via np.memmap,
    allowing the OS to manage physical RAM through the page cache.
    This reduces RSS and lets the kernel reclaim frame memory under
    pressure instead of triggering OOM kills.

    When ``cache_dir`` is given, memmap files are stored under
    ``<cache_dir>/radar/`` and survive process restarts.  Multiple
    processes can map the same files read-only to share the OS page
    cache — the foundation of the multi-worker tile-server split.
    """

    def __init__(self, max_frames: int = 12, cache_dir: Path | None = None):
        self._max_frames = max_frames
        self._frames: list[RadarFrame] = []
        self._lock = asyncio.Lock()
        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "radar"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_frames_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        # Drop stale .tmp files from a crash mid-write so subsequent
        # atomic writes don't trip on existing files.
        for path in self._memmap_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        logger.info(
            "Frame memmap directory: %s (persistent=%s)",
            self._memmap_dir, self._persistent,
        )

    def _to_memmap(self, timestamp: int, region_name: str, data: np.ndarray) -> np.ndarray:
        """Write array to disk atomically and return a read-only memory-mapped view.

        Atomic write (.tmp → os.replace) ensures readers in other processes
        never see a half-written file — required for multi-worker safety.
        """
        final = self._memmap_dir / f"{timestamp}_{region_name}.dat"
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        os.replace(tmp, final)
        return np.memmap(final, dtype=data.dtype, mode="r", shape=data.shape)

    def _cleanup_timestamp(self, timestamp: int) -> None:
        """Delete memmap files for an evicted timestamp."""
        for path in self._memmap_dir.glob(f"{timestamp}_*.dat"):
            try:
                path.unlink()
            except OSError:
                pass

    async def add_frame(self, frame: RadarFrame) -> tuple[int | None, bool]:
        """Add a frame, evicting the oldest if at capacity.

        If a frame with the same timestamp exists, merge the region data.
        Returns (evicted_timestamp | None, was_merged).
        """
        async with self._lock:
            # Convert regions to memory-mapped files
            for name, data in list(frame.regions.items()):
                frame.regions[name] = self._to_memmap(frame.timestamp, name, data)

            # Merge into existing frame if same timestamp
            for existing in self._frames:
                if existing.timestamp == frame.timestamp:
                    existing.regions.update(frame.regions)
                    # Provenance merges here too, and the order matters.
                    # This pass supplied these regions, so an earlier
                    # pass's verdict on them is out of date — drop it
                    # first, so a region that arrives FRESH now loses the
                    # carried mark it picked up before.  Without this a
                    # successful re-fetch keeps reporting stale forever.
                    for name in frame.regions:
                        existing.carried_from.pop(name, None)
                    # Then this pass's own marks, so a region carried in
                    # THIS batch keeps its origin.  Scoping the pop to
                    # frame.regions is what keeps the second priority
                    # wave from disturbing the first wave's entries.
                    existing.carried_from.update(frame.carried_from)
                    return None, True

            evicted_ts = None
            if len(self._frames) >= self._max_frames:
                evicted = self._frames.pop(0)
                evicted_ts = evicted.timestamp
                self._cleanup_timestamp(evicted_ts)

            self._frames.append(frame)
            self._frames.sort(key=lambda f: f.timestamp)
            return evicted_ts, False

    async def get_frame(self, timestamp: int) -> RadarFrame | None:
        async with self._lock:
            for f in self._frames:
                if f.timestamp == timestamp:
                    return f
        return None

    async def get_latest_frame(self) -> RadarFrame | None:
        async with self._lock:
            return self._frames[-1] if self._frames else None

    async def get_timestamps(self) -> list[int]:
        async with self._lock:
            return [f.timestamp for f in self._frames]

    async def get_region_keys(self) -> dict[int, set[str]]:
        """Return a mapping of timestamp -> set of region names present."""
        async with self._lock:
            return {f.timestamp: set(f.regions.keys()) for f in self._frames}

    async def get_region_status(self) -> dict[int, dict[str, int | None]]:
        """Return timestamp -> {region present in that frame: origin | None}.

        ``None`` means the region was observed at the frame's own
        timestamp (fresh); an int is the timestamp of the observation the
        data really came from (carried).  A region missing from the inner
        dict was absent from that frame entirely — callers derive that by
        comparing against the enabled-region list, which is the only way
        to catch regions no code path touched at all.
        """
        async with self._lock:
            return {
                f.timestamp: {
                    name: f.carried_from.get(name) for name in f.regions
                }
                for f in self._frames
            }

    async def frame_count(self) -> int:
        async with self._lock:
            return len(self._frames)

    @property
    def data_bytes(self) -> int:
        """Total bytes across all region arrays in all frames."""
        total = 0
        for frame in self._frames:
            for arr in frame.regions.values():
                total += arr.nbytes
        return total

    def __getstate__(self) -> dict:
        """Serialize state for cross-process reload.

        Returns a JSON-serializable dict describing the on-disk layout.
        Only meaningful in persistent mode (``cache_dir`` configured) —
        a tile-server worker re-opens the same memmaps via __setstate__.
        File paths are stored as basenames (relative to ``memmap_dir``)
        so the snapshot is portable across processes that share the
        cache volume even if it's mounted at different absolute paths.
        """
        return {
            "max_frames": self._max_frames,
            "memmap_dir": str(self._memmap_dir),
            "frames": [
                {
                    "timestamp": f.timestamp,
                    "regions": {
                        name: [
                            os.path.basename(str(arr.filename)),
                            arr.dtype.str,
                            list(arr.shape),
                        ]
                        for name, arr in f.regions.items()
                    },
                    # The only channel per-region provenance has to the
                    # render workers: they answer weather-maps.json but
                    # never run a fetcher, so anything kept as fetcher
                    # state is invisible to them.
                    "carried_from": dict(f.carried_from),
                }
                for f in self._frames
            ],
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state from the dict produced by ``__getstate__``.

        Re-opens memmaps read-only from the recorded basenames under
        ``memmap_dir``.  Used by the tile-server worker on startup and
        on every state.json refresh — replaces the in-memory frame list
        in place so existing references to ``FrameStore`` stay valid for
        ongoing renders (Linux holds the old memmap inodes alive until
        all readers release them).
        """
        max_frames = state["max_frames"]
        memmap_dir = Path(state["memmap_dir"])
        new_frames: list[RadarFrame] = []
        for f_info in state["frames"]:
            frame = RadarFrame(
                timestamp=f_info["timestamp"],
                # Absent in snapshots written before this field existed;
                # empty is correct there — nothing was known to be stale.
                carried_from={
                    k: int(v) for k, v in (f_info.get("carried_from") or {}).items()
                },
            )
            for name, (basename, dtype_str, shape) in f_info["regions"].items():
                frame.regions[name] = np.memmap(
                    memmap_dir / basename,
                    dtype=np.dtype(dtype_str), mode="r",
                    shape=tuple(shape),
                )
            new_frames.append(frame)
        new_frames.sort(key=lambda f: f.timestamp)

        # Apply atomically — if this object is being updated in place,
        # readers see either the old list or the new list, never partial.
        self._max_frames = max_frames
        self._memmap_dir = memmap_dir
        self._frames = new_frames
        self._persistent = True
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()

    def cleanup(self) -> None:
        """Release in-memory frame references; remove temp dir if non-persistent.

        In persistent mode, the on-disk memmap files are intentionally
        kept so a fresh process can pick them up via the constructor's
        warm-restart logic or via __setstate__.
        """
        self._frames = []
        if self._persistent:
            logger.info("Frame memmaps retained on disk at %s", self._memmap_dir)
        else:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("Frame memmap directory cleaned up")
