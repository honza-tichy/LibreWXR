# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import json
from dataclasses import dataclass

import numpy as np
import pytest

pytestmark = pytest.mark.store

from librewxr.data.radar_cache import SCHEMA_VERSION, RadarFrameCache
from librewxr.data.store import RadarFrame


@dataclass
class FakeRegion:
    """Minimal RegionDef stand-in for cache shape validation."""
    height: int
    width: int


def _make_frame(ts: int, regions: dict[str, tuple[int, int]]) -> RadarFrame:
    """Build a frame with deterministic uint8 data per region."""
    return RadarFrame(
        timestamp=ts,
        regions={
            name: np.full(shape, ts % 256, dtype=np.uint8)
            for name, shape in regions.items()
        },
    )


class TestRadarFrameCache:
    def test_write_then_load_round_trip(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20), "AKCOMP": FakeRegion(5, 8)}

        frame = _make_frame(1000, {"USCOMP": (10, 20), "AKCOMP": (5, 8)})
        cache.write_frame(frame)
        cache.save_metadata(regions, [1000])

        loaded = cache.load_frames(regions)
        assert len(loaded) == 1
        assert loaded[0].timestamp == 1000
        assert set(loaded[0].regions) == {"USCOMP", "AKCOMP"}
        np.testing.assert_array_equal(loaded[0].regions["USCOMP"],
                                      frame.regions["USCOMP"])
        np.testing.assert_array_equal(loaded[0].regions["AKCOMP"],
                                      frame.regions["AKCOMP"])

    def test_carried_from_survives_the_round_trip(self, tmp_path):
        """A restart must not promote carried data back to "fresh"."""
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20), "AKCOMP": FakeRegion(5, 8)}

        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20), "AKCOMP": (5, 8)}))
        cache.save_metadata(regions, [1000], {1000: {"AKCOMP": 400}})

        loaded = cache.load_frames(regions)
        assert loaded[0].carried_from == {"AKCOMP": 400}

    def test_metadata_without_carried_from_still_loads_frames(self, tmp_path):
        """Upgrade path, and the reason SCHEMA_VERSION is not bumped.

        A bump makes load_frames return nothing on mismatch, which would
        throw away the whole cached window and cold-start the pipeline on
        deploy.  Old metadata must simply read as all-fresh.
        """
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20)}

        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20)}))
        cache.save_metadata(regions, [1000])  # two-arg, as before this change

        meta = json.loads((tmp_path / "radar" / "metadata.json").read_text())
        del meta["carried_from"]
        (tmp_path / "radar" / "metadata.json").write_text(json.dumps(meta))

        loaded = cache.load_frames(regions)
        assert len(loaded) == 1
        assert loaded[0].carried_from == {}

    def test_carried_from_dropped_for_regions_that_fail_validation(
        self, tmp_path,
    ):
        """Provenance must not outlive the array it describes."""
        cache = RadarFrameCache(tmp_path)
        old_regions = {"USCOMP": FakeRegion(10, 20)}
        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20)}))
        cache.save_metadata(old_regions, [1000], {1000: {"USCOMP": 400}})

        # Region reshaped in code — the cached file is dropped on load.
        loaded = cache.load_frames({"USCOMP": FakeRegion(11, 21)})
        assert loaded == [] or "USCOMP" not in loaded[0].carried_from

    def test_schema_version_mismatch_invalidates_cache(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20)}

        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20)}))
        cache.save_metadata(regions, [1000])

        # Stomp the metadata to a different schema version.
        meta = json.loads((tmp_path / "radar" / "metadata.json").read_text())
        meta["schema_version"] = SCHEMA_VERSION + 99
        (tmp_path / "radar" / "metadata.json").write_text(json.dumps(meta))

        assert cache.load_frames(regions) == []

    def test_shape_mismatch_drops_only_that_region(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        old_regions = {"USCOMP": FakeRegion(10, 20), "AKCOMP": FakeRegion(5, 8)}

        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20), "AKCOMP": (5, 8)}))
        cache.save_metadata(old_regions, [1000])

        # Pretend USCOMP got reshaped in code; AKCOMP is unchanged.
        new_regions = {"USCOMP": FakeRegion(15, 25), "AKCOMP": FakeRegion(5, 8)}
        loaded = cache.load_frames(new_regions)
        assert len(loaded) == 1
        assert set(loaded[0].regions) == {"AKCOMP"}
        # The bad file should have been dropped.
        assert not (tmp_path / "radar" / "radar_1000_USCOMP.dat").exists()

    def test_cleanup_removes_inactive_timestamps(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20)}

        for ts in (1000, 2000, 3000):
            cache.write_frame(_make_frame(ts, {"USCOMP": (10, 20)}))

        cache.cleanup(active_timestamps=[2000, 3000])

        assert not (tmp_path / "radar" / "radar_1000_USCOMP.dat").exists()
        assert (tmp_path / "radar" / "radar_2000_USCOMP.dat").exists()
        assert (tmp_path / "radar" / "radar_3000_USCOMP.dat").exists()

    def test_load_with_empty_cache_returns_empty(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        assert cache.load_frames({"USCOMP": FakeRegion(10, 20)}) == []

    def test_load_recovers_orphan_files_when_metadata_missing(self, tmp_path):
        """SIGBUS or crash mid-backfill leaves .dat files but stale metadata —
        load_frames should disk-scan and recover them."""
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20)}

        for ts in (1000, 2000, 3000):
            cache.write_frame(_make_frame(ts, {"USCOMP": (10, 20)}))
        # Note: no save_metadata call — simulates a crash before metadata write.

        loaded = cache.load_frames(regions)
        assert [f.timestamp for f in loaded] == [1000, 2000, 3000]
        for frame in loaded:
            assert set(frame.regions) == {"USCOMP"}

    def test_load_unions_metadata_and_disk_timestamps(self, tmp_path):
        """If metadata is partially stale (older save), disk-scanned orphans
        get added without losing the recorded ones."""
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20)}

        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20)}))
        cache.save_metadata(regions, [1000])
        # New frame written but metadata not refreshed yet.
        cache.write_frame(_make_frame(2000, {"USCOMP": (10, 20)}))

        loaded = cache.load_frames(regions)
        assert [f.timestamp for f in loaded] == [1000, 2000]

    def test_partial_frame_when_only_some_regions_cached(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        # Write only USCOMP, but request both on load.
        regions = {"USCOMP": FakeRegion(10, 20)}
        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20)}))
        cache.save_metadata(regions, [1000])

        load_regions = {"USCOMP": FakeRegion(10, 20), "AKCOMP": FakeRegion(5, 8)}
        loaded = cache.load_frames(load_regions)
        assert len(loaded) == 1
        assert set(loaded[0].regions) == {"USCOMP"}

    def test_atomic_write_no_tmp_left_behind(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20)}))
        cache.save_metadata({"USCOMP": FakeRegion(10, 20)}, [1000])

        tmp_files = list((tmp_path / "radar").glob("*.tmp"))
        assert tmp_files == []
