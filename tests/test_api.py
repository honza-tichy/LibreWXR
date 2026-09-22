# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

from librewxr.api import routes
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH


def _make_test_app() -> tuple[FastAPI, FrameStore, TileCache, int, int]:
    """Create a minimal FastAPI app with just the router — no lifespan."""
    store = FrameStore(max_frames=12)
    cache = TileCache(max_mb=10)
    ts = int(time.time() // 300) * 300
    ts_prev = ts - 600

    data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    data[2500:2700, 6000:6200] = 128

    import asyncio
    frame = RadarFrame(timestamp=ts, regions={"USCOMP": data})
    asyncio.run(store.add_frame(frame))
    prev_frame = RadarFrame(timestamp=ts_prev, regions={"USCOMP": data})
    asyncio.run(store.add_frame(prev_frame))

    # Wire shared state directly — same as main.py does after lifespan init
    routes.frame_store = store
    routes.tile_cache = cache
    routes.ecmwf_grid = None
    routes.tile_warmer = None
    routes.nowcast_store = None
    routes.start_time = time.time()
    routes.enabled_regions = ["USCOMP"]

    test_app = FastAPI()
    test_app.include_router(routes.router)
    return test_app, store, cache, ts, ts_prev


# Module-scoped: built once, shared across all tests in this file
_app, _store, _cache, _ts, _ts_prev = _make_test_app()


@pytest.fixture(scope="module")
def client():
    with TestClient(_app, raise_server_exceptions=False) as c:
        yield c, _ts, _ts_prev


class TestWeatherMapsEndpoint:
    def test_returns_valid_json(self, client):
        c, ts, ts_prev = client
        resp = c.get("/public/weather-maps.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "2.0"
        assert "generated" in data
        assert "host" in data
        assert "radar" in data
        assert "past" in data["radar"]
        assert "nowcast" in data["radar"]
        assert "satellite" in data

    def test_past_contains_timestamps(self, client):
        c, ts, ts_prev = client
        resp = c.get("/public/weather-maps.json")
        data = resp.json()
        past = data["radar"]["past"]
        assert len(past) >= 1
        # past is sorted oldest-first; ts_prev was added first (earlier)
        assert past[0]["time"] == ts_prev
        assert past[0]["path"] == f"/v2/radar/{ts_prev}"


class TestRadarTileEndpoint:
    def test_valid_tile_request(self, client):
        c, ts, ts_prev = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_webp_format(self, client):
        c, ts, ts_prev = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.webp")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"

    def test_missing_timestamp(self, client):
        c, _, _ = client
        resp = c.get("/v2/radar/9999999999/256/4/3/5/2/0_0.png")
        assert resp.status_code == 404

    def test_latest_frame_cache_header(self, client):
        """Latest frame gets short cache lifetime."""
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert "cache-control" in resp.headers
        assert "max-age=300" in resp.headers["cache-control"]

    def test_historical_frame_cache_header(self, client):
        """Historical frames get long cache lifetime since they are immutable."""
        c, _, ts_prev = client
        resp = c.get(f"/v2/radar/{ts_prev}/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        assert "cache-control" in resp.headers
        assert "max-age=7200" in resp.headers["cache-control"]


class TestCoverageTileEndpoint:
    def test_valid_coverage_request(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/4/3/5/0/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"


class TestRadarCoverage:
    """The per-region health block the app reads to flag stale frames.

    Uses its own store and restores the module globals afterwards — the
    shared _make_test_app() state is reused by every other class in this
    file, and mutating it in place produces order-dependent failures.
    """

    @pytest.fixture
    def coverage_client(self):
        import asyncio

        store = FrameStore(max_frames=12)
        ts = int(time.time() // 600) * 600
        arr = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)

        # t0: both regions fresh.  t1: TWCOMP carried from t0.  t2: TWCOMP
        # gone entirely (past the lookback), USCOMP still fine.
        asyncio.run(store.add_frame(RadarFrame(
            timestamp=ts - 1200, regions={"USCOMP": arr, "TWCOMP": arr},
        )))
        asyncio.run(store.add_frame(RadarFrame(
            timestamp=ts - 600,
            regions={"USCOMP": arr, "TWCOMP": arr},
            carried_from={"TWCOMP": ts - 1200},
        )))
        asyncio.run(store.add_frame(RadarFrame(
            timestamp=ts, regions={"USCOMP": arr},
        )))

        saved_store, saved_regions = routes.frame_store, routes.enabled_regions
        routes.frame_store = store
        routes.enabled_regions = ["USCOMP", "TWCOMP"]
        app = FastAPI()
        app.include_router(routes.router)
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, ts
        finally:
            routes.frame_store = saved_store
            routes.enabled_regions = saved_regions

    def test_lists_enabled_region_footprints(self, coverage_client):
        c, _ = coverage_client
        cov = c.get("/public/weather-maps.json").json()["radar"]["coverage"]

        ids = {r["id"] for r in cov["regions"]}
        assert ids == {"USCOMP", "TWCOMP"}
        tw = next(r for r in cov["regions"] if r["id"] == "TWCOMP")
        assert tw["label"] == "Taiwan"
        assert len(tw["bounds"]) == 4 and tw["px"] > 0

    def test_healthy_region_is_omitted_from_degraded(self, coverage_client):
        c, _ = coverage_client
        cov = c.get("/public/weather-maps.json").json()["radar"]["coverage"]
        assert "USCOMP" not in cov["degraded"]

    def test_carried_and_absent_frames_are_reported(self, coverage_client):
        c, ts = coverage_client
        cov = c.get("/public/weather-maps.json").json()["radar"]["coverage"]

        tw = cov["degraded"]["TWCOMP"]
        assert tw["carried"] == [ts - 600]
        assert tw["absent"] == [ts]
        # The age the pill shows comes from here: the last real
        # observation, NOT the frame the copy was taken from.
        assert tw["latestObserved"] == ts - 1200

    def test_rainviewer_shape_is_unchanged(self, coverage_client):
        c, _ = coverage_client
        body = c.get("/public/weather-maps.json").json()

        assert set(body) >= {"version", "generated", "host", "radar", "satellite"}
        assert set(body["radar"]) >= {"past", "nowcast", "colorSchemes"}
        for frame in body["radar"]["past"]:
            # Frames stay exactly {time, path}: coverage is keyed by
            # region, not smeared across every timestamp.
            assert set(frame) == {"time", "path"}

    def test_health_reports_region_freshness(self, coverage_client):
        c, ts = coverage_client
        frames = c.get("/health").json()["frames"]

        tw = frames["per_region_status"]["TWCOMP"]
        assert (tw["fresh"], tw["carried"], tw["absent"]) == (1, 1, 1)
        assert tw["latest_observed"] == ts - 1200
        # Presence alone would have said 2 of 3 and looked healthy.
        assert frames["per_region"]["TWCOMP"] == 2
