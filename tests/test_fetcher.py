# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio

import numpy as np
import pytest

pytestmark = pytest.mark.store

from librewxr.data.fetcher import RadarFetcher
from librewxr.data.radar_cache import RadarFrameCache
from librewxr.data.regions import RegionDef
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH


class TestFrameStore:
    @pytest.mark.asyncio
    async def test_add_and_get(self):
        store = FrameStore(max_frames=3)
        data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
        frame = RadarFrame(timestamp=100, regions={"USCOMP": data})
        await store.add_frame(frame)

        result = await store.get_frame(100)
        assert result is not None
        assert result.timestamp == 100

    @pytest.mark.asyncio
    async def test_eviction(self):
        store = FrameStore(max_frames=2)
        data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)

        await store.add_frame(RadarFrame(timestamp=100, regions={"USCOMP": data}))
        await store.add_frame(RadarFrame(timestamp=200, regions={"USCOMP": data}))
        evicted_ts, merged = await store.add_frame(RadarFrame(timestamp=300, regions={"USCOMP": data}))

        assert evicted_ts == 100
        assert merged is False
        assert await store.get_frame(100) is None
        assert await store.get_frame(200) is not None
        assert await store.get_frame(300) is not None

    @pytest.mark.asyncio
    async def test_duplicate_timestamp_merges_regions(self):
        store = FrameStore(max_frames=3)
        data1 = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
        data2 = np.ones((100, 100), dtype=np.uint8)

        _, merged1 = await store.add_frame(RadarFrame(timestamp=100, regions={"USCOMP": data1}))
        _, merged2 = await store.add_frame(RadarFrame(timestamp=100, regions={"AKCOMP": data2}))

        assert merged1 is False
        assert merged2 is True
        assert await store.frame_count() == 1
        frame = await store.get_frame(100)
        assert "USCOMP" in frame.regions
        assert "AKCOMP" in frame.regions

    @pytest.mark.asyncio
    async def test_sorted_order(self):
        store = FrameStore(max_frames=5)
        data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)

        await store.add_frame(RadarFrame(timestamp=300, regions={"USCOMP": data}))
        await store.add_frame(RadarFrame(timestamp=100, regions={"USCOMP": data}))
        await store.add_frame(RadarFrame(timestamp=200, regions={"USCOMP": data}))

        timestamps = await store.get_timestamps()
        assert timestamps == [100, 200, 300]

    @pytest.mark.asyncio
    async def test_get_latest(self):
        store = FrameStore(max_frames=5)
        data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)

        await store.add_frame(RadarFrame(timestamp=100, regions={"USCOMP": data}))
        await store.add_frame(RadarFrame(timestamp=300, regions={"USCOMP": data}))
        await store.add_frame(RadarFrame(timestamp=200, regions={"USCOMP": data}))

        latest = await store.get_latest_frame()
        assert latest.timestamp == 300


class TestTileCache:
    def test_put_and_get(self):
        cache = TileCache(max_mb=10)
        key = (100, 4, 3, 5, 256, 2, False, False, "png")
        cache.put(key, b"tile_data")
        assert cache.get(key) == b"tile_data"

    def test_byte_eviction(self):
        # Create a cache with a 10-byte limit
        cache = TileCache.__new__(TileCache)
        cache._max_bytes = 10
        cache._cache = __import__("collections").OrderedDict()
        cache._total_bytes = 0
        cache._lock = __import__("threading").Lock()

        k1 = (1,)
        k2 = (2,)
        k3 = (3,)
        cache.put(k1, b"12345")  # 5 bytes, total=5
        cache.put(k2, b"12345")  # 5 bytes, total=10
        cache.put(k3, b"12345")  # 5 bytes, would be 15 -> evicts k1, total=10

        assert cache.get(k1) is None  # evicted
        assert cache.get(k2) == b"12345"
        assert cache.get(k3) == b"12345"
        assert cache.total_bytes == 10

    def test_tracks_bytes(self):
        cache = TileCache(max_mb=10)
        cache.put((1,), b"hello")
        cache.put((2,), b"world!")
        assert cache.total_bytes == 11
        assert cache.size == 2

    def test_invalidate_timestamp(self):
        cache = TileCache(max_mb=10)
        cache.put((100, 4, 3, 5), b"a")
        cache.put((100, 4, 3, 6), b"b")
        cache.put((200, 4, 3, 5), b"c")

        cache.invalidate_timestamp(100)
        assert cache.get((100, 4, 3, 5)) is None
        assert cache.get((100, 4, 3, 6)) is None
        assert cache.get((200, 4, 3, 5)) == b"c"
        assert cache.total_bytes == 1

    def test_evict_half(self):
        cache = TileCache(max_mb=10)
        cache.put((1,), b"aaa")
        cache.put((2,), b"bbb")
        cache.put((3,), b"ccc")
        cache.put((4,), b"ddd")

        freed = cache.evict_half()
        assert freed == 6  # evicted 2 oldest entries (3 bytes each)
        assert cache.size == 2
        assert cache.total_bytes == 6
        assert cache.get((1,)) is None
        assert cache.get((2,)) is None
        assert cache.get((3,)) == b"ccc"
        assert cache.get((4,)) == b"ddd"


class _FakeSource:
    """Returns a deterministic uint8 array sized to the region grid.

    ``fill_value`` is configurable so tests can produce different data
    on different calls (proving carry-forward really copies prior data
    rather than fetching fresh).  Set ``next_return = None`` to
    simulate a silent drop on the next call.
    """

    def __init__(self, fill_value: int = 50):
        self.live_calls: list[tuple[str, int]] = []
        self.archive_calls: list[tuple[str, int]] = []
        self.fill_value = fill_value
        self.next_return: object = ...  # ... = use default array

    def _build_array(self, region):
        return np.full(
            (region.height, region.width), self.fill_value, dtype=np.uint8,
        )

    async def fetch_frame(self, region, minutes_ago):
        self.live_calls.append((region.name, minutes_ago))
        if self.next_return is not ...:
            val = self.next_return
            self.next_return = ...
            return val
        return self._build_array(region)

    async def fetch_archive_frame(self, region, dt):
        self.archive_calls.append((region.name, int(dt.timestamp())))
        if self.next_return is not ...:
            val = self.next_return
            self.next_return = ...
            return val
        return self._build_array(region)


def _build_fetcher(store, tile_cache, radar_cache, region):
    """Bypass __init__ so we don't drag in real source dispatch / settings."""
    fetcher = RadarFetcher.__new__(RadarFetcher)
    fetcher._store = store
    fetcher._cache = tile_cache
    fetcher._nwp_contributions = []
    fetcher._nowcast_generator = None
    fetcher._warmer = None
    fetcher._radar_cache = radar_cache
    fetcher._task = None
    fetcher._warm_task = None
    fetcher._enabled_regions = [region]
    fetcher._na_source = "iem"
    fetcher._ca_source = "msc"
    source = _FakeSource()
    fetcher._sources = {region.name: source}
    fetcher._cacomp_msc_source = None
    fetcher._iem_fallback = None
    fetcher._cacomp_msc_available = None
    fetcher._on_cycle_complete = None
    return fetcher, source


class TestFetcherRadarCacheWiring:
    @pytest.fixture
    def small_region(self):
        # Explicit grid_width/height keeps arrays tiny so despeckle's
        # neighbor scan stays cheap even with the default min_neighbors=3.
        return RegionDef(
            name="TESTREG",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="US",
            grid_width=32, grid_height=32,
        )

    @pytest.mark.asyncio
    async def test_fetcher_persists_frames_to_radar_cache(
        self, tmp_path, small_region
    ):
        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        radar_cache = RadarFrameCache(tmp_path)
        fetcher, source = _build_fetcher(store, tile_cache, radar_cache, small_region)

        await fetcher._fetch_timestamps([
            (1000, "live", 0),
            (2000, "live", 10),
        ])

        # .dat files should exist for both timestamps.
        assert (tmp_path / "radar" / "radar_1000_TESTREG.dat").exists()
        assert (tmp_path / "radar" / "radar_2000_TESTREG.dat").exists()
        # metadata.json should record both timestamps and the region shape.
        meta_path = tmp_path / "radar" / "metadata.json"
        assert meta_path.exists()
        import json
        meta = json.loads(meta_path.read_text())
        assert sorted(meta["timestamps"]) == [1000, 2000]
        assert meta["regions"]["TESTREG"]["shape"] == [32, 32]

    @pytest.mark.asyncio
    async def test_fetcher_cleanup_removes_evicted_timestamps(
        self, tmp_path, small_region
    ):
        # max_frames=2 forces the oldest timestamp to be evicted on the
        # third write; cache.cleanup should follow the store's lead and
        # delete the corresponding .dat file.
        store = FrameStore(max_frames=2)
        tile_cache = TileCache(max_mb=1)
        radar_cache = RadarFrameCache(tmp_path)
        fetcher, _source = _build_fetcher(store, tile_cache, radar_cache, small_region)

        await fetcher._fetch_timestamps([(1000, "live", 0)])
        await fetcher._fetch_timestamps([(2000, "live", 10)])
        await fetcher._fetch_timestamps([(3000, "live", 20)])

        # Store should hold only the newest two; cache should match.
        assert sorted(await store.get_timestamps()) == [2000, 3000]
        assert not (tmp_path / "radar" / "radar_1000_TESTREG.dat").exists()
        assert (tmp_path / "radar" / "radar_2000_TESTREG.dat").exists()
        assert (tmp_path / "radar" / "radar_3000_TESTREG.dat").exists()

    @pytest.mark.asyncio
    async def test_fetcher_without_radar_cache_does_not_crash(
        self, tmp_path, small_region
    ):
        # When cache_dir is unset in production, _radar_cache is None;
        # _fetch_timestamps should still drive the store cleanly.
        store = FrameStore(max_frames=2)
        tile_cache = TileCache(max_mb=1)
        fetcher, _source = _build_fetcher(store, tile_cache, None, small_region)

        await fetcher._fetch_timestamps([(1000, "live", 0)])
        assert await store.get_timestamps() == [1000]


class TestOnCycleCompleteHook:
    @pytest.fixture
    def small_region(self):
        return RegionDef(
            name="TESTREG",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="US",
            grid_width=32, grid_height=32,
        )

    @pytest.mark.asyncio
    async def test_async_hook_runs_after_each_cycle(self, small_region):
        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, _src = _build_fetcher(store, tile_cache, None, small_region)

        calls = 0

        async def hook():
            nonlocal calls
            calls += 1

        fetcher._on_cycle_complete = hook
        await fetcher._fire_cycle_complete()
        await fetcher._fire_cycle_complete()
        assert calls == 2

    @pytest.mark.asyncio
    async def test_sync_hook_supported(self, small_region):
        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, _src = _build_fetcher(store, tile_cache, None, small_region)

        calls = 0

        def hook():
            nonlocal calls
            calls += 1

        fetcher._on_cycle_complete = hook
        await fetcher._fire_cycle_complete()
        assert calls == 1

    @pytest.mark.asyncio
    async def test_hook_failure_does_not_propagate(self, small_region):
        # A failed snapshot dump must never kill the fetcher loop.
        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, _src = _build_fetcher(store, tile_cache, None, small_region)

        async def hook():
            raise RuntimeError("disk full")

        fetcher._on_cycle_complete = hook
        await fetcher._fire_cycle_complete()  # should not raise

    @pytest.mark.asyncio
    async def test_no_hook_is_silent(self, small_region):
        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, _src = _build_fetcher(store, tile_cache, None, small_region)
        assert fetcher._on_cycle_complete is None
        await fetcher._fire_cycle_complete()  # should not raise

    @pytest.mark.asyncio
    async def test_constructor_accepts_hook_kwarg(self):
        # Smoke check that the public constructor accepts on_cycle_complete.
        # We bypass __init__ for the body of the test, but verify the
        # signature includes the kwarg so future refactors don't drop it.
        import inspect as _inspect
        sig = _inspect.signature(RadarFetcher.__init__)
        assert "on_cycle_complete" in sig.parameters


class TestCarryForward:
    """When a fetch returns no data for an enabled region, the fetcher
    fills the new frame from the most recent prior frame in the store
    (up to ``_CARRY_FORWARD_MAX_INTERVALS`` lookback).  Users see
    continuous radar instead of intermittent NWP-fallback flicker.
    """

    @pytest.fixture
    def small_region(self):
        return RegionDef(
            name="TESTREG",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="US",
            grid_width=32, grid_height=32,
        )

    @pytest.mark.asyncio
    async def test_silent_drop_carries_forward_from_prev_frame(
        self, small_region,
    ):
        # settings.fetch_interval defaults to 600 — match that so the
        # lookback math (ts - N*interval) lines up with our test ts.
        from librewxr.config import settings
        interval = settings.fetch_interval

        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, source = _build_fetcher(store, tile_cache, None, small_region)

        # First fetch lands a real frame at ts=1000.
        source.fill_value = 77
        await fetcher._fetch_timestamps([(1000, "live", 0)])
        assert (await store.get_frame(1000)).regions["TESTREG"][0, 0] == 77

        # Second fetch: source returns None (simulating a silent drop).
        source.fill_value = 99  # any later real value should NOT appear
        source.next_return = None
        await fetcher._fetch_timestamps([(1000 + interval, "live", 10)])

        # The new frame must exist AND contain the carried-forward data
        # from ts=1000 — value 77, not the 99 default.
        new_frame = await store.get_frame(1000 + interval)
        assert new_frame is not None
        assert "TESTREG" in new_frame.regions
        assert new_frame.regions["TESTREG"][0, 0] == 77

    @pytest.mark.asyncio
    async def test_carry_forward_respects_staleness_limit(
        self, small_region,
    ):
        """If the only prior frame is more than _CARRY_FORWARD_MAX_INTERVALS
        old, no carry-forward happens — the region drops cleanly."""
        from librewxr.config import settings
        interval = settings.fetch_interval

        store = FrameStore(max_frames=8)
        tile_cache = TileCache(max_mb=1)
        fetcher, source = _build_fetcher(store, tile_cache, None, small_region)

        # Anchor frame at ts=1000.
        source.fill_value = 77
        await fetcher._fetch_timestamps([(1000, "live", 0)])

        # Silent drop 3 intervals later — past the 2-interval limit.
        source.next_return = None
        far_ts = 1000 + 3 * interval
        await fetcher._fetch_timestamps([(far_ts, "live", 30)])

        # The far frame should NOT contain TESTREG — the stale data is
        # too old to carry forward.  Either the frame doesn't exist or
        # the region key is absent.
        far_frame = await store.get_frame(far_ts)
        if far_frame is not None:
            assert "TESTREG" not in far_frame.regions

    @pytest.mark.asyncio
    async def test_successful_refetch_overrides_carried_data(
        self, small_region,
    ):
        """A later real fetch for the same ts replaces the carry-forward
        copy via the FrameStore's merge-on-duplicate-ts behaviour."""
        from librewxr.config import settings
        interval = settings.fetch_interval

        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, source = _build_fetcher(store, tile_cache, None, small_region)

        # Establish prev frame, then carry-forward into a dropped ts.
        source.fill_value = 77
        await fetcher._fetch_timestamps([(1000, "live", 0)])

        source.next_return = None
        await fetcher._fetch_timestamps([(1000 + interval, "live", 10)])
        assert (await store.get_frame(1000 + interval)).regions["TESTREG"][0, 0] == 77

        # Re-fetch the same ts with real data — should override.
        source.fill_value = 111
        await fetcher._fetch_timestamps([(1000 + interval, "live", 10)])
        assert (await store.get_frame(1000 + interval)).regions["TESTREG"][0, 0] == 111

    @pytest.mark.asyncio
    async def test_carry_forward_is_independent_copy(
        self, small_region,
    ):
        """The carried-forward array must be detached from the source
        frame's memmap, so eviction of the source can't invalidate it."""
        from librewxr.config import settings
        interval = settings.fetch_interval

        # max_frames=2 forces the original ts=1000 frame to be evicted
        # once a third timestamp is written.
        store = FrameStore(max_frames=2)
        tile_cache = TileCache(max_mb=1)
        fetcher, source = _build_fetcher(store, tile_cache, None, small_region)

        source.fill_value = 77
        await fetcher._fetch_timestamps([(1000, "live", 0)])

        # Carry-forward into ts=1000+interval.
        source.next_return = None
        await fetcher._fetch_timestamps([(1000 + interval, "live", 10)])

        # Push a third timestamp — evicts ts=1000.  The carried-forward
        # data at ts=1000+interval should survive because it was copied,
        # not memmap-shared.
        source.fill_value = 200
        await fetcher._fetch_timestamps([(1000 + 2 * interval, "live", 20)])

        carried = (await store.get_frame(1000 + interval)).regions["TESTREG"]
        assert carried[0, 0] == 77  # still readable, original value


class _StubSatelliteInstance:
    """Satellite source stand-in exposing only what the background task uses."""

    def __init__(self, behavior):
        self._behavior = behavior

    async def fetch(self) -> bool:
        return await self._behavior()


class _StubSatelliteContribution:
    def __init__(self, behavior, name="GMGSI LW"):
        self.name = name
        self.instance = _StubSatelliteInstance(behavior)


def _build_satellite_fetcher():
    """Bare fetcher with just the state _fetch_satellite_background touches."""
    fetcher = RadarFetcher.__new__(RadarFetcher)
    fetcher._on_cycle_complete = None
    fetcher._satellite_tasks = {}
    return fetcher


class TestSatelliteBackgroundFetch:
    async def test_hung_fetch_times_out_and_frees_the_skip_gate(
        self, monkeypatch, caplog
    ):
        """A fetch that hangs must finish (via deadline) so later cycles retry.

        The scheduler skips a channel while its previous task is pending;
        before the deadline existed, one hung S3 call froze the channel
        until restart with nothing logged above DEBUG.
        """
        from librewxr.config import settings

        monkeypatch.setattr(settings, "satellite_fetch_timeout", 0.05)
        fetcher = _build_satellite_fetcher()

        async def hang() -> bool:
            await asyncio.sleep(30)
            return True

        contrib = _StubSatelliteContribution(hang)
        task = asyncio.create_task(fetcher._fetch_satellite_background(contrib))
        with caplog.at_level("WARNING"):
            await asyncio.wait_for(task, timeout=5)  # must not take ~30s

        assert task.done()
        assert any("timed out" in r.message for r in caplog.records)

    async def test_successful_fetch_fires_cycle_complete(self):
        fetcher = _build_satellite_fetcher()
        fired = asyncio.Event()

        async def on_complete() -> None:
            fired.set()

        fetcher._on_cycle_complete = on_complete

        async def ok() -> bool:
            return True

        await fetcher._fetch_satellite_background(_StubSatelliteContribution(ok))
        assert fired.is_set()

    async def test_failed_fetch_is_dropped_with_a_warning(self, caplog):
        fetcher = _build_satellite_fetcher()

        async def boom() -> bool:
            raise RuntimeError("s3 exploded")

        with caplog.at_level("WARNING"):
            await fetcher._fetch_satellite_background(_StubSatelliteContribution(boom))
        assert any("fetch failed" in r.message for r in caplog.records)


class _StubGrid:
    """NWP grid stand-in.  ``fetch`` takes no kwargs, so the signature
    inspection in _fetch_auxiliary_grids passes an empty kwargs dict."""

    def __init__(self, behavior):
        self._behavior = behavior

    async def fetch(self) -> None:
        await self._behavior()


class _StubNWPContribution:
    def __init__(self, behavior, name="WRF-SMN"):
        self.name = name
        self.instance = _StubGrid(behavior)


def _build_nwp_fetcher(contributions):
    """Bare fetcher with just the state _fetch_auxiliary_grids touches."""
    fetcher = RadarFetcher.__new__(RadarFetcher)
    fetcher._nwp_contributions = contributions
    fetcher._satellite_contributions = []
    fetcher._satellite_tasks = {}
    return fetcher


class TestNWPFetchDeadline:
    async def test_slow_grid_times_out_so_the_cycle_can_reach_radar(
        self, monkeypatch, caplog
    ):
        """A slow NWP source must not hold the cycle open indefinitely.

        _fetch_all_frames awaits _fetch_auxiliary_grids BEFORE fetching
        radar, so an un-deadlined grid stalls radar as well as its own
        layer.  Observed 2026-09-18: WRF-SMN took 23-25 min per fetch for
        ~100 min, radar aged past the staleness limit, and the watchdog
        restarted into the same wait four times over.
        """
        from librewxr.config import settings

        monkeypatch.setattr(settings, "nwp_fetch_timeout", 0.05)

        async def hang() -> None:
            await asyncio.sleep(30)

        fetcher = _build_nwp_fetcher([_StubNWPContribution(hang)])
        with caplog.at_level("WARNING"):
            # Must not take ~30s: the deadline, not the source, ends this.
            await asyncio.wait_for(fetcher._fetch_auxiliary_grids(), timeout=5)

        assert any("timed out" in r.message for r in caplog.records)
        assert any("WRF-SMN" in r.message for r in caplog.records)

    async def test_deadline_is_per_source_not_per_cycle(self, monkeypatch):
        """One timing-out grid must not abort its healthy neighbours."""
        from librewxr.config import settings

        monkeypatch.setattr(settings, "nwp_fetch_timeout", 0.05)
        monkeypatch.setattr(settings, "nwp_fetch_concurrency", 4)
        finished: list[str] = []

        async def hang() -> None:
            await asyncio.sleep(30)

        async def quick() -> None:
            finished.append("quick")

        fetcher = _build_nwp_fetcher([
            _StubNWPContribution(hang, name="WRF-SMN"),
            _StubNWPContribution(quick, name="ICON-EU"),
        ])
        await asyncio.wait_for(fetcher._fetch_auxiliary_grids(), timeout=5)

        assert finished == ["quick"]

    async def test_failed_grid_still_warns_without_the_timeout_path(
        self, monkeypatch, caplog
    ):
        """A grid that raises keeps the pre-existing failure message."""
        from librewxr.config import settings

        monkeypatch.setattr(settings, "nwp_fetch_timeout", 30.0)

        async def boom() -> None:
            raise RuntimeError("grib exploded")

        fetcher = _build_nwp_fetcher([_StubNWPContribution(boom)])
        with caplog.at_level("WARNING"):
            await fetcher._fetch_auxiliary_grids()

        assert any("fetch failed" in r.message for r in caplog.records)
        assert not any("timed out" in r.message for r in caplog.records)


class _RecordingSource(_FakeSource):
    """_FakeSource that records the raw arg it was handed, untouched.

    The base class coerces the archive arg via ``int(dt.timestamp())``,
    which would itself blow up on a wrongly-typed value — this keeps the
    assertion about *which* arg arrived, not about its type.
    """

    def __init__(self):
        super().__init__()
        self.raw_live: list = []
        self.raw_archive: list = []

    async def fetch_frame(self, region, minutes_ago):
        self.raw_live.append(minutes_ago)
        return self._build_array(region)

    async def fetch_archive_frame(self, region, dt):
        self.raw_archive.append(dt)
        return self._build_array(region)


class _NeverSource(_FakeSource):
    """Primary that always returns None, forcing the fallback path."""

    async def fetch_frame(self, region, minutes_ago):
        return None

    async def fetch_archive_frame(self, region, dt):
        return None


class TestSourceArgIsPerFrame:
    """_fetch_timestamps must hand each frame its OWN source_arg.

    Regression: source_arg was read in the results loop but bound only by
    the task-building loop, so every frame received the last entry's arg.
    ts_and_sources is ordered newest-first with live entries for frames
    under 55 min and archive entries beyond, so the leaked value was
    always the oldest frame's datetime.
    """

    @pytest.fixture
    def cacomp_region(self):
        return RegionDef(
            name="CACOMP",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="CA",
            grid_width=32, grid_height=32,
        )

    @pytest.mark.asyncio
    async def test_fallback_gets_each_frames_own_arg(self, cacomp_region):
        from datetime import datetime, timezone

        store = FrameStore(max_frames=8)
        fetcher, _ = _build_fetcher(store, TileCache(max_mb=1), None, cacomp_region)
        fetcher._sources = {cacomp_region.name: _NeverSource()}
        fetcher._ca_source = "mrms_with_msc_blend"
        msc = _RecordingSource()
        fetcher._cacomp_msc_source = msc

        dt2000 = datetime.fromtimestamp(2000, tz=timezone.utc)
        dt3000 = datetime.fromtimestamp(3000, tz=timezone.utc)
        await fetcher._fetch_timestamps([
            (1000, "live", 0),
            (2000, "archive", dt2000),
            (3000, "archive", dt3000),
        ])

        # With the leak, source_arg was dt3000 throughout: the live frame
        # got a datetime where minutes_ago belongs (TypeError against the
        # real MSC source), and ts=2000 was silently fetched at dt3000.
        assert msc.raw_live == [0]
        assert msc.raw_archive == [dt2000, dt3000]


class _HangingSource(_FakeSource):
    """Never returns — stands in for a degraded far-end endpoint."""

    async def fetch_frame(self, region, minutes_ago):
        await asyncio.sleep(30)

    async def fetch_archive_frame(self, region, dt):
        await asyncio.sleep(30)


class TestRadarFetchDeadline:
    """One slow region must not set the whole cycle's wall time.

    Regression 2026-09-21: the East Asia radar endpoints degraded, the
    per-region fetches ran for minutes under their own generous HTTP
    timeouts (CWA: 90 s read x 2 attempts per file), and the gather in
    _fetch_timestamps held the cycle open for 600-950 s against a
    80-250 s baseline.  Cycles overran their 10-minute boundary, frames
    stopped landing, and the external watchdog restarted a server that
    was working — eight times in five hours, each costing a backfill.
    """

    @pytest.fixture
    def region(self):
        return RegionDef(
            name="TWCOMP",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="TW",
            grid_width=32, grid_height=32,
        )

    @pytest.mark.asyncio
    async def test_hanging_region_is_abandoned_not_waited_on(
        self, region, monkeypatch, caplog
    ):
        from librewxr.config import settings

        monkeypatch.setattr(settings, "radar_fetch_timeout", 0.05)
        store = FrameStore(max_frames=8)
        fetcher, _ = _build_fetcher(store, TileCache(max_mb=1), None, region)
        fetcher._sources = {region.name: _HangingSource()}

        with caplog.at_level("WARNING"):
            # Must not take ~30 s: the deadline ends this, not the source.
            await asyncio.wait_for(
                fetcher._fetch_timestamps([(1000, "live", 0)]), timeout=5,
            )

        assert any("timed out" in r.message for r in caplog.records)
        assert any("TWCOMP" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_deadline_is_per_region_so_healthy_ones_still_land(
        self, region, monkeypatch
    ):
        """A hanging region must not cost its healthy neighbours' frames."""
        from librewxr.config import settings

        monkeypatch.setattr(settings, "radar_fetch_timeout", 0.05)
        healthy = RegionDef(
            name="OPERA",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="EU",
            grid_width=32, grid_height=32,
        )
        store = FrameStore(max_frames=8)
        fetcher, _ = _build_fetcher(store, TileCache(max_mb=1), None, region)
        fetcher._enabled_regions = [region, healthy]
        fetcher._sources = {
            region.name: _HangingSource(),
            healthy.name: _FakeSource(fill_value=70),
        }

        await asyncio.wait_for(
            fetcher._fetch_timestamps([(1000, "live", 0)]), timeout=5,
        )

        frame = await store.get_frame(1000)
        assert frame is not None
        # The frame exists and carries the healthy region only.
        assert set(frame.regions) == {"OPERA"}

    @pytest.mark.asyncio
    async def test_fast_region_is_untouched_by_the_deadline(
        self, region, monkeypatch, caplog
    ):
        """A region that answers in time keeps its data and logs nothing."""
        from librewxr.config import settings

        monkeypatch.setattr(settings, "radar_fetch_timeout", 30.0)
        store = FrameStore(max_frames=8)
        fetcher, source = _build_fetcher(store, TileCache(max_mb=1), None, region)

        with caplog.at_level("WARNING"):
            await fetcher._fetch_timestamps([(1000, "live", 0)])

        assert source.live_calls == [("TWCOMP", 0)]
        frame = await store.get_frame(1000)
        assert frame is not None and "TWCOMP" in frame.regions
        assert not any("timed out" in r.message for r in caplog.records)


class TestFetchCycleBudget:
    """A cycle that overruns must be abandoned, not allowed to skip a boundary.

    The loop sleeps to the next clock-aligned boundary, so a cycle that
    runs past it does not overlap the next one — it *skips* it, and a
    skipped boundary is a missing frame.  Per-source deadlines cannot
    prevent this: a cycle is a sum, and on 2026-09-22 four batches of
    merely-slow NWP grids reached 6.5 min without one of them going near
    nwp_fetch_timeout.
    """

    @pytest.fixture
    def region(self):
        return RegionDef(
            name="TESTREG",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="TEST",
            grid_width=32, grid_height=32,
        )

    def _fetcher(self, region):
        fetcher, _ = _build_fetcher(
            FrameStore(max_frames=8), TileCache(max_mb=1), None, region,
        )
        return fetcher

    @pytest.mark.asyncio
    async def test_overrunning_fetch_is_abandoned(
        self, region, monkeypatch, caplog
    ):
        from librewxr.config import settings

        monkeypatch.setattr(settings, "fetch_cycle_timeout", 0.05)
        monkeypatch.setattr(settings, "fetch_interval", 1)
        fetcher = self._fetcher(region)

        # The initial backfill is deliberately NOT bounded — it fetches
        # the whole history window and legitimately runs for minutes — so
        # the stub must clear it before the loop's budget can be tested.
        calls = {"n": 0}

        async def slow_fetch():
            calls["n"] += 1
            if calls["n"] == 1:
                return
            await asyncio.sleep(30)

        ran: list[str] = []

        async def note_nowcast():
            ran.append("nowcast")

        monkeypatch.setattr(fetcher, "_fetch_all_frames", slow_fetch)
        monkeypatch.setattr(fetcher, "_run_nowcast", note_nowcast)
        monkeypatch.setattr(fetcher, "_schedule_warm", lambda: None)

        task = asyncio.create_task(fetcher._backfill_then_loop())
        with caplog.at_level("WARNING"):
            await asyncio.sleep(1.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert any("exceeded its" in r.message for r in caplog.records)
        # The point of bounding the fetch stage and not the whole cycle:
        # what landed is still published.
        assert "nowcast" in ran

    @pytest.mark.asyncio
    async def test_fast_cycle_is_untouched(self, region, monkeypatch, caplog):
        from librewxr.config import settings

        monkeypatch.setattr(settings, "fetch_cycle_timeout", 30.0)
        monkeypatch.setattr(settings, "fetch_interval", 1)
        fetcher = self._fetcher(region)

        calls: list[str] = []

        async def quick_fetch():
            calls.append("fetch")

        async def quick_nowcast():
            calls.append("nowcast")

        monkeypatch.setattr(fetcher, "_fetch_all_frames", quick_fetch)
        monkeypatch.setattr(fetcher, "_run_nowcast", quick_nowcast)
        monkeypatch.setattr(fetcher, "_schedule_warm", lambda: None)

        task = asyncio.create_task(fetcher._backfill_then_loop())
        with caplog.at_level("WARNING"):
            await asyncio.sleep(1.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert calls.count("fetch") >= 1
        assert not any("exceeded its" in r.message for r in caplog.records)


class TestRegionPriorityWaves:
    """High-priority regions are fetched and stored before the rest.

    asyncio.gather hands back results only once every task in it has
    settled, so a single all-regions gather loses the regions that
    already succeeded whenever the enclosing cycle budget abandons the
    stage.  Splitting into waves writes each wave to the store before the
    next starts, so an abandoned cycle costs the periphery rather than
    the regions most clients look at.
    """

    def _region(self, name, group):
        return RegionDef(
            name=name,
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group=group,
            grid_width=32, grid_height=32,
        )

    @pytest.mark.asyncio
    async def test_priority_regions_are_fetched_first(self, monkeypatch):
        from librewxr.config import settings

        monkeypatch.setattr(settings, "radar_priority_groups", "US,EUROPE")
        monkeypatch.setattr(settings, "radar_fetch_timeout", 30.0)

        uscomp = self._region("USCOMP", "US")
        opera = self._region("OPERA", "EUROPE")
        twcomp = self._region("TWCOMP", "TAIWAN")
        svcomp = self._region("SVCOMP", "CENTRAL_AMERICA")

        order: list[str] = []

        class _OrderingSource(_FakeSource):
            async def fetch_frame(self, region, minutes_ago):
                order.append(region.name)
                return self._build_array(region)

        store = FrameStore(max_frames=8)
        fetcher, _ = _build_fetcher(store, TileCache(max_mb=1), None, uscomp)
        # Deliberately listed periphery-first, the order the discovery
        # walker actually produces (SVCOMP, JPCOMP, TWCOMP, ... USCOMP).
        fetcher._enabled_regions = [svcomp, twcomp, opera, uscomp]
        fetcher._sources = {
            r.name: _OrderingSource() for r in fetcher._enabled_regions
        }

        await fetcher._fetch_timestamps([(1000, "live", 0)])

        assert set(order[:2]) == {"OPERA", "USCOMP"}
        assert set(order[2:]) == {"SVCOMP", "TWCOMP"}

    @pytest.mark.asyncio
    async def test_priority_wave_lands_even_when_the_rest_hangs(
        self, monkeypatch
    ):
        """The point of the split: abandoning the stage keeps wave one."""
        from librewxr.config import settings

        monkeypatch.setattr(settings, "radar_priority_groups", "US")
        monkeypatch.setattr(settings, "radar_fetch_timeout", 30.0)

        uscomp = self._region("USCOMP", "US")
        twcomp = self._region("TWCOMP", "TAIWAN")

        store = FrameStore(max_frames=8)
        fetcher, _ = _build_fetcher(store, TileCache(max_mb=1), None, uscomp)
        fetcher._enabled_regions = [twcomp, uscomp]
        fetcher._sources = {
            uscomp.name: _FakeSource(fill_value=60),
            twcomp.name: _HangingSource(),
        }

        # Abandon the whole stage the way fetch_cycle_timeout does.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                fetcher._fetch_timestamps([(1000, "live", 0)]), timeout=0.3,
            )

        frame = await store.get_frame(1000)
        assert frame is not None
        assert "USCOMP" in frame.regions

    @pytest.mark.asyncio
    async def test_empty_priority_list_keeps_one_wave(self, monkeypatch):
        from librewxr.config import settings

        monkeypatch.setattr(settings, "radar_priority_groups", "")
        uscomp = self._region("USCOMP", "US")
        twcomp = self._region("TWCOMP", "TAIWAN")

        store = FrameStore(max_frames=8)
        fetcher, _ = _build_fetcher(store, TileCache(max_mb=1), None, uscomp)
        fetcher._enabled_regions = [twcomp, uscomp]

        assert len(fetcher._region_waves()) == 1

    @pytest.mark.asyncio
    async def test_startup_log_prints_the_waves(self, monkeypatch, caplog):
        """The line everyone checks must match the order actually used."""
        from librewxr.config import settings

        monkeypatch.setattr(settings, "radar_priority_groups", "US,EUROPE")

        uscomp = self._region("USCOMP", "US")
        opera = self._region("OPERA", "EUROPE")
        twcomp = self._region("TWCOMP", "TAIWAN")

        store = FrameStore(max_frames=8)
        fetcher, _ = _build_fetcher(store, TileCache(max_mb=1), None, uscomp)
        fetcher._enabled_regions = [twcomp, opera, uscomp]
        fetcher._sources = {
            r.name: _FakeSource() for r in fetcher._enabled_regions
        }

        async def noop():
            return None

        monkeypatch.setattr(fetcher, "_fetch_initial", noop)
        monkeypatch.setattr(fetcher, "_backfill_then_loop", noop)

        with caplog.at_level("INFO"):
            await fetcher.start()
        # Not stop(): _FakeSource has no close(), and the loop task is
        # the only thing start() leaves behind that needs cleaning up.
        fetcher._task.cancel()

        line = next(
            r.getMessage() for r in caplog.records
            if "Fetching regions" in r.getMessage()
        )
        assert line == "Fetching regions: [OPERA, USCOMP] then [TWCOMP]"

    @pytest.mark.asyncio
    async def test_startup_log_is_a_flat_list_with_one_wave(
        self, monkeypatch, caplog
    ):
        from librewxr.config import settings

        monkeypatch.setattr(settings, "radar_priority_groups", "")

        uscomp = self._region("USCOMP", "US")
        twcomp = self._region("TWCOMP", "TAIWAN")

        store = FrameStore(max_frames=8)
        fetcher, _ = _build_fetcher(store, TileCache(max_mb=1), None, uscomp)
        fetcher._enabled_regions = [twcomp, uscomp]

        async def noop():
            return None

        monkeypatch.setattr(fetcher, "_fetch_initial", noop)
        monkeypatch.setattr(fetcher, "_backfill_then_loop", noop)

        with caplog.at_level("INFO"):
            await fetcher.start()
        # Not stop(): _FakeSource has no close(), and the loop task is
        # the only thing start() leaves behind that needs cleaning up.
        fetcher._task.cancel()

        line = next(
            r.getMessage() for r in caplog.records
            if "Fetching regions" in r.getMessage()
        )
        assert line == "Fetching regions: TWCOMP, USCOMP"


class TestCarryForwardProvenance:
    """Carried regions must say which observation they really came from.

    A carried region is present in its frame, so get_region_keys reports
    that timestamp as complete and it is never re-fetched, and the next
    cycle carries the already-carried copy forward again.
    _CARRY_FORWARD_MAX_INTERVALS bounds the lookback, not the chain — so
    the lookback distance understates the real age, by days during a
    sustained outage (Taiwan/CWA, 2026-09-21 onward). Recording the
    origin is what lets clients report an honest age.
    """

    @pytest.fixture
    def small_region(self):
        return RegionDef(
            name="TESTREG",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="US",
            grid_width=32, grid_height=32,
        )

    @pytest.mark.asyncio
    async def test_carry_forward_records_the_origin(self, small_region):
        from librewxr.config import settings
        interval = settings.fetch_interval

        store = FrameStore(max_frames=8)
        fetcher, source = _build_fetcher(
            store, TileCache(max_mb=1), None, small_region,
        )

        await fetcher._fetch_timestamps([(1000, "live", 0)])
        source.next_return = None
        await fetcher._fetch_timestamps([(1000 + interval, "live", 10)])

        frame = await store.get_frame(1000 + interval)
        assert frame.carried_from == {"TESTREG": 1000}

    @pytest.mark.asyncio
    async def test_chain_reports_the_original_not_the_previous_frame(
        self, small_region,
    ):
        """The regression that matters: a carried copy carried again.

        Without chasing the chain, each frame would claim its data came
        from one interval back and look 10 minutes old forever, however
        long the source has actually been down.
        """
        from librewxr.config import settings
        interval = settings.fetch_interval

        store = FrameStore(max_frames=8)
        fetcher, source = _build_fetcher(
            store, TileCache(max_mb=1), None, small_region,
        )

        await fetcher._fetch_timestamps([(1000, "live", 0)])
        for step in (1, 2, 3):
            source.next_return = None
            await fetcher._fetch_timestamps(
                [(1000 + step * interval, "live", 10 * step)],
            )

        frame = await store.get_frame(1000 + 3 * interval)
        assert frame.carried_from == {"TESTREG": 1000}, (
            "origin must chase back to the real observation, not stop at "
            "the previous (itself carried) frame"
        )

    @pytest.mark.asyncio
    async def test_fresh_region_records_nothing(self, small_region):
        store = FrameStore(max_frames=8)
        fetcher, _ = _build_fetcher(
            store, TileCache(max_mb=1), None, small_region,
        )

        await fetcher._fetch_timestamps([(1000, "live", 0)])

        assert (await store.get_frame(1000)).carried_from == {}

    @pytest.mark.asyncio
    async def test_absent_region_is_derived_not_recorded(self, small_region):
        """Past the lookback limit the region drops, with no entry at all.

        "Absent" is read off the missing key in ``regions``, never
        written — which is what makes it correct for regions no code path
        touched, such as a priority wave the cycle budget abandoned.
        """
        from librewxr.config import settings
        interval = settings.fetch_interval

        store = FrameStore(max_frames=8)
        fetcher, source = _build_fetcher(
            store, TileCache(max_mb=1), None, small_region,
        )

        await fetcher._fetch_timestamps([(1000, "live", 0)])
        far = 1000 + (fetcher._CARRY_FORWARD_MAX_INTERVALS + 1) * interval
        source.next_return = None
        await fetcher._fetch_timestamps([(far, "live", 30)])

        frame = await store.get_frame(far)
        assert frame is None or "TESTREG" not in frame.regions
        if frame is not None:
            assert "TESTREG" not in frame.carried_from

    @pytest.mark.asyncio
    async def test_fresh_refetch_clears_the_carried_mark(self, small_region):
        """A region that comes back must stop being reported stale.

        This is the add_frame merge path: without the pop, a successful
        re-fetch merges into the existing frame and the old mark survives
        forever.
        """
        from librewxr.config import settings
        interval = settings.fetch_interval
        ts = 1000 + interval

        store = FrameStore(max_frames=8)
        fetcher, source = _build_fetcher(
            store, TileCache(max_mb=1), None, small_region,
        )

        await fetcher._fetch_timestamps([(1000, "live", 0)])
        source.next_return = None
        await fetcher._fetch_timestamps([(ts, "live", 10)])
        assert (await store.get_frame(ts)).carried_from == {"TESTREG": 1000}

        # Now the source recovers and the same ts is fetched again.
        source.fill_value = 42
        await fetcher._fetch_timestamps([(ts, "live", 10)])

        frame = await store.get_frame(ts)
        assert frame.regions["TESTREG"][0, 0] == 42
        assert frame.carried_from == {}

    @pytest.mark.asyncio
    async def test_second_wave_does_not_disturb_the_first(self, monkeypatch):
        """Two waves, one timestamp, disjoint regions — marks must not cross.

        _fetch_regions_for_timestamps runs once per priority wave and
        both merge into the same frame, so the merge's pop has to be
        scoped to the regions the wave actually supplied.
        """
        from librewxr.config import settings

        monkeypatch.setattr(settings, "radar_priority_groups", "US")
        monkeypatch.setattr(settings, "radar_fetch_timeout", 30.0)
        interval = settings.fetch_interval

        us = RegionDef(
            name="USCOMP", west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="US", grid_width=32, grid_height=32,
        )
        tw = RegionDef(
            name="TWCOMP", west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="TAIWAN", grid_width=32, grid_height=32,
        )

        store = FrameStore(max_frames=8)
        fetcher, _ = _build_fetcher(store, TileCache(max_mb=1), None, us)
        fetcher._enabled_regions = [us, tw]
        us_src, tw_src = _FakeSource(fill_value=10), _FakeSource(fill_value=20)
        fetcher._sources = {"USCOMP": us_src, "TWCOMP": tw_src}

        await fetcher._fetch_timestamps([(1000, "live", 0)])

        # Taiwan drops; the US stays healthy.
        tw_src.next_return = None
        await fetcher._fetch_timestamps([(1000 + interval, "live", 10)])

        frame = await store.get_frame(1000 + interval)
        assert frame.carried_from == {"TWCOMP": 1000}, (
            "the healthy wave-one region must not pick up a mark, and the "
            "carried wave-two region must keep its own"
        )
