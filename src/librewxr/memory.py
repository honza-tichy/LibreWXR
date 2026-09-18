# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Memory pressure monitor — safety net to prevent OOM kills.

Periodically checks unreclaimable memory against the container/system
limit and proactively evicts caches before the OOM killer intervenes.

Reclaimable page cache is excluded on purpose: the radar and NWP stores
are file-backed memmaps, so counting their cache would have the monitor
evicting caches to relieve pressure created by its own caches.
"""
import asyncio
import ctypes
import gc
import logging
from pathlib import Path

import psutil

from librewxr.tiles.cache import TileCache

logger = logging.getLogger(__name__)


def release_memory() -> None:
    """Force Python garbage collection and return freed pages to the OS.

    Python's garbage collector doesn't run eagerly for non-cyclic objects,
    and glibc's malloc never returns freed heap pages to the OS on its own.
    Calling gc.collect() + malloc_trim(0) after heavy operations (ECMWF
    regridding, nowcast optical flow) reclaims hundreds of MB that would
    otherwise show up as "other" in the memory breakdown.
    """
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass  # Non-glibc platform (musl, macOS) — gc.collect() is enough

# Eviction thresholds (fraction of memory limit)
_WARN_THRESHOLD = 0.80
_EVICT_TILES_THRESHOLD = 0.85
_EVICT_ALL_THRESHOLD = 0.90


def detect_memory_limit_mb(override_mb: int = 0) -> int:
    """Detect container memory limit in MB.

    Priority: explicit override > cgroup v2 > cgroup v1 > system RAM.
    """
    if override_mb > 0:
        return override_mb

    # cgroup v2
    try:
        cg2 = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if cg2 != "max":
            return int(cg2) // (1024 * 1024)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    # cgroup v1
    try:
        cg1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip()
        limit = int(cg1)
        # cgroup v1 reports a huge number when unlimited
        if limit < psutil.virtual_memory().total * 2:
            return limit // (1024 * 1024)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    # Fallback: system RAM
    return psutil.virtual_memory().total // (1024 * 1024)


_CGROUP_ROOT = Path("/sys/fs/cgroup")

# cgroup v2 ``memory.stat`` keys that the kernel CANNOT reclaim under
# pressure.  Everything omitted — above all the page cache behind
# file-backed memmaps — is dropped on demand and never causes an OOM
# kill, so counting it as pressure is counting our own cache against us.
#
# ``shmem`` is included even though it is accounted under ``file``: it is
# tmpfs-backed and cannot be reclaimed without swap, which is where the
# non-persistent frame/nowcast memmaps land if /tmp is a tmpfs mount.
_V2_UNRECLAIMABLE_KEYS = (
    "anon",
    "shmem",
    "slab_unreclaimable",
    "kernel_stack",
    "pagetables",
    "percpu",
    "sock",
)


def _parse_memory_stat(path: Path) -> dict[str, int] | None:
    """Parse a cgroup ``memory.stat`` into a key→bytes dict, or None."""
    try:
        text = path.read_text()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, raw = line.partition(" ")
        try:
            values[key] = int(raw)
        except ValueError:
            continue  # nested/percpu lines that aren't a plain integer
    return values


def _read_cgroup_unreclaimable_bytes(cgroup_root: Path = _CGROUP_ROOT) -> int | None:
    """Return the cgroup's UNRECLAIMABLE memory in bytes, or None.

    Captures every process in the container — important in multi-worker
    mode where each render worker's own RSS is only a fraction of the
    container's total.  Returns ``None`` outside containers so callers
    can use per-process RSS instead.

    Deliberately NOT ``memory.current``: that includes page cache, and
    this project memmaps multi-GB radar/NWP stores to ``cache_dir`` by
    design.  Reading it raw made the monitor fire on its own caches — in
    production, ``anon`` 1.6 GiB + ``slab`` 0.2 GiB against a 12 GiB cap
    was reported as "12103 MB / 11000 MB (110%)", because 6.7 GiB of
    reclaimable page cache was counted as pressure.  The monitor then
    cleared the tile and coordinate caches (~10 s to rebuild) several
    times per fetch cycle to relieve pressure that did not exist, while
    ``memory.events`` showed ``oom_kill 0`` over 20 hours.

    Summing the unreclaimable keys instead makes the limit mean what the
    thresholds assume: how close we are to an actual OOM kill.
    """
    # cgroup v2 — ``anon`` is the discriminator; v1 has ``rss`` instead.
    v2 = _parse_memory_stat(cgroup_root / "memory.stat")
    if v2 is not None and "anon" in v2:
        return sum(v2.get(key, 0) for key in _V2_UNRECLAIMABLE_KEYS)

    # cgroup v1 — ``total_*`` are the hierarchy-inclusive variants.
    # ``rss`` here already excludes page cache but also excludes shmem,
    # which lives in ``cache``, so add it back.
    v1 = _parse_memory_stat(cgroup_root / "memory" / "memory.stat")
    if v1 is not None:
        rss = v1.get("total_rss", v1.get("rss"))
        if rss is not None:
            return rss + v1.get("total_shmem", v1.get("shmem", 0))

    return None


class MemoryMonitor:
    """Background task that monitors memory and evicts caches under pressure."""

    def __init__(
        self,
        tile_cache: TileCache,
        coord_cache_clear_fn,
        memory_limit_mb: int,
        check_interval: int = 30,
    ):
        self._tile_cache = tile_cache
        self._clear_coord_caches = coord_cache_clear_fn
        self._limit_bytes = memory_limit_mb * 1024 * 1024
        self._limit_mb = memory_limit_mb
        self._check_interval = check_interval
        self._task: asyncio.Task | None = None
        self._process = psutil.Process()

    async def start(self) -> None:
        scope = (
            "container (cgroup)"
            if _read_cgroup_unreclaimable_bytes() is not None
            else "process"
        )
        logger.info(
            "Memory monitor started (scope=%s, limit=%d MB, check every %ds, "
            "warn=%.0f%%, evict_tiles=%.0f%%, evict_all=%.0f%%)",
            scope, self._limit_mb, self._check_interval,
            _WARN_THRESHOLD * 100, _EVICT_TILES_THRESHOLD * 100,
            _EVICT_ALL_THRESHOLD * 100,
        )
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._check_interval)
            try:
                self._check()
            except Exception:
                logger.exception("Memory monitor check failed")

    def _check(self) -> None:
        # In multi-worker deployments the container holds N render
        # workers, each with its own ``psutil.Process``.  Comparing one
        # worker's RSS to the container-wide cgroup limit never trips
        # the thresholds because no single worker holds more than ~1/N
        # of the limit.  Read the cgroup's own usage when available so
        # every worker sees the same shared pressure and they all evict
        # their local caches in concert.  Falls back to per-process RSS
        # outside containers (local dev, single-process deployments).
        cgroup_usage = _read_cgroup_unreclaimable_bytes()
        if cgroup_usage is not None:
            rss = cgroup_usage
        else:
            rss = self._process.memory_info().rss
        usage = rss / self._limit_bytes

        if usage >= _EVICT_ALL_THRESHOLD:
            logger.warning(
                "Memory critical: %d MB / %d MB (%.0f%%) — clearing tile + coord caches",
                rss // (1024 * 1024), self._limit_mb, usage * 100,
            )
            self._tile_cache.clear()
            self._clear_coord_caches()
            release_memory()

        elif usage >= _EVICT_TILES_THRESHOLD:
            freed = self._tile_cache.evict_half()
            release_memory()
            logger.warning(
                "Memory pressure: %d MB / %d MB (%.0f%%) — evicted %.1f MB of tiles",
                rss // (1024 * 1024), self._limit_mb, usage * 100,
                freed / (1024 * 1024),
            )

        elif usage >= _WARN_THRESHOLD:
            logger.info(
                "Memory usage elevated: %d MB / %d MB (%.0f%%)",
                rss // (1024 * 1024), self._limit_mb, usage * 100,
            )
