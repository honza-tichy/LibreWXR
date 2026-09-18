# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Memory monitor accounting.

The monitor exists to pre-empt an OOM kill, so the number it compares
against the limit must be memory the kernel cannot reclaim.  Page cache
is reclaimable, and this project memmaps its radar/NWP stores to disk by
design — counting that cache made the monitor evict its own caches to
relieve imaginary pressure.
"""
import pytest

pytestmark = pytest.mark.store

from librewxr.memory import (
    _read_cgroup_unreclaimable_bytes,
    _V2_UNRECLAIMABLE_KEYS,
)

# Verbatim from a production container (2026-09-18) whose monitor was
# reporting "12103 MB / 11000 MB (110%)" while memory.events showed
# oom_kill 0.  anon 1.63 GiB + slab_unreclaimable, against 6.70 GiB of
# reclaimable page cache.
_PROD_V2_STAT = """\
anon 1751228416
file 7190949888
kernel_stack 2064384
pagetables 23068672
percpu 1441792
sock 274432
shmem 0
file_mapped 3005120512
file_dirty 1048576
file_writeback 0
inactive_anon 1048576
active_anon 1750179840
inactive_file 4185829376
active_file 3005120512
unevictable 0
slab_reclaimable 158924800
slab_unreclaimable 22961856
slab 181886656
"""


def _write_v2(tmp_path, text=_PROD_V2_STAT):
    (tmp_path / "memory.stat").write_text(text)
    return tmp_path


class TestUnreclaimableAccounting:
    def test_page_cache_is_excluded(self, tmp_path):
        """The 6.7 GiB of file cache must not count toward the limit."""
        got = _read_cgroup_unreclaimable_bytes(_write_v2(tmp_path))

        # anon + kernel_stack + pagetables + percpu + sock + shmem
        # + slab_unreclaimable — and nothing file-backed.
        assert got == (
            1751228416 + 2064384 + 23068672 + 1441792 + 274432 + 0 + 22961856
        )
        assert got < 1.9 * 1024**3  # ~1.68 GiB, not the 8.68 GiB of memory.current

    def test_the_regression_thresholds_no_longer_trip(self, tmp_path):
        """Against the real 11000 MB limit this container is quiet.

        The whole point: these exact numbers used to read as 110% and
        clear the tile + coordinate caches several times per fetch cycle.
        """
        from librewxr.memory import _WARN_THRESHOLD

        got = _read_cgroup_unreclaimable_bytes(_write_v2(tmp_path))
        usage = got / (11000 * 1024 * 1024)

        assert usage < _WARN_THRESHOLD
        assert usage == pytest.approx(0.156, abs=0.01)

    def test_shmem_counts_as_unreclaimable(self, tmp_path):
        """tmpfs pages sit under `file` but cannot be reclaimed w/o swap.

        Non-persistent frame/nowcast memmaps land here when /tmp is a
        tmpfs mount, so they must be counted despite that accounting.
        """
        base = _read_cgroup_unreclaimable_bytes(_write_v2(tmp_path))
        with_shmem = _read_cgroup_unreclaimable_bytes(
            _write_v2(tmp_path, _PROD_V2_STAT.replace("shmem 0", "shmem 2147483648"))
        )
        assert with_shmem - base == 2147483648

    def test_missing_keys_are_tolerated(self, tmp_path):
        """Older kernels omit some keys; absent must mean zero, not crash."""
        got = _read_cgroup_unreclaimable_bytes(
            _write_v2(tmp_path, "anon 1000\nfile 9999999\n")
        )
        assert got == 1000

    def test_cgroup_v1_falls_back_to_rss_plus_shmem(self, tmp_path):
        """v1 has no `anon` key; `rss` excludes page cache but also shmem."""
        v1 = tmp_path / "memory"
        v1.mkdir()
        (v1 / "memory.stat").write_text(
            "cache 7000000000\nrss 1500000000\nshmem 500000000\n"
            "total_cache 7000000000\ntotal_rss 1500000000\ntotal_shmem 500000000\n"
        )
        assert _read_cgroup_unreclaimable_bytes(tmp_path) == 2000000000

    def test_outside_a_container_returns_none(self, tmp_path):
        """No cgroup files at all — caller falls back to per-process RSS."""
        assert _read_cgroup_unreclaimable_bytes(tmp_path) is None

    def test_unparsable_lines_are_skipped(self, tmp_path):
        got = _read_cgroup_unreclaimable_bytes(
            _write_v2(tmp_path, "anon 1000\nbogus not_a_number\nsock 7\n")
        )
        assert got == 1007

    def test_every_declared_key_is_summed(self, tmp_path):
        """Guards the key tuple against a typo silently dropping a term."""
        stat = "".join(f"{key} 1\n" for key in _V2_UNRECLAIMABLE_KEYS)
        assert _read_cgroup_unreclaimable_bytes(_write_v2(tmp_path, stat)) == len(
            _V2_UNRECLAIMABLE_KEYS
        )
