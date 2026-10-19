# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
import struct

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.sources

from librewxr.data.regions import REGIONS, resolve_regions
from librewxr.sources.regional.europe.radar.opera import (
    OperaSource,
    _parse_opera_hdf5,
)
from librewxr.tiles.coordinates import tile_overlaps_region


class TestOperaRegion:
    def test_opera_in_regions(self):
        assert "OPERA" in REGIONS

    def test_opera_is_laea(self):
        r = REGIONS["OPERA"]
        assert r.proj == "laea"
        assert r.group == "EUROPE"

    def test_opera_dimensions(self):
        r = REGIONS["OPERA"]
        assert r.width == 3800
        assert r.height == 4400

    def test_europe_group_resolution(self):
        # EUROPE bundles OPERA + ITCOMP (DPC Italian composite).  Group
        # membership is stored alphabetically by the discovery walker.
        assert resolve_regions("EUROPE") == ["ITCOMP", "OPERA"]

    def test_all_includes_opera(self):
        assert "OPERA" in resolve_regions("ALL")

    def test_mixed_regions(self):
        result = resolve_regions("EUROPE,CANADA")
        assert "OPERA" in result
        assert "CACOMP" in result


class TestOperaLAEA:
    def test_tile_over_europe_overlaps(self):
        region = REGIONS["OPERA"]
        # z=4, x=8, y=5 is central Europe
        assert tile_overlaps_region(region, z=4, x=8, y=5)

    def test_tile_over_america_does_not_overlap(self):
        region = REGIONS["OPERA"]
        # z=4, x=3, y=5 is eastern US
        assert not tile_overlaps_region(region, z=4, x=3, y=5)


class TestOperaHDF5Parser:
    def _make_opera_hdf5(
        self,
        data: np.ndarray,
        nodata: float = -9999000.0,
        undetect: float = -8888000.0,
        gain: float = 1.0,
        offset: float = 0.0,
    ) -> bytes:
        """Build a minimal OPERA-style ODIM HDF5 in memory."""
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            ds = f.create_group("dataset1")
            d1 = ds.create_group("data1")
            d1.create_dataset("data", data=data)
            what = d1.create_group("what")
            what.attrs["gain"] = gain
            what.attrs["offset"] = offset
            what.attrs["nodata"] = nodata
            what.attrs["undetect"] = undetect
            what.attrs["quantity"] = b"DBZH"
        return buf.getvalue()

    def test_basic_parsing(self):
        data = np.array([[20.0, 30.0], [-9999000.0, -8888000.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result is not None
        assert result.shape == (2, 2)

    def test_nodata_becomes_zero(self):
        data = np.array([[-9999000.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 0

    def test_undetect_becomes_zero(self):
        data = np.array([[-8888000.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 0

    def test_valid_dbz_encoding(self):
        # 20 dBZ -> (20+32)*2 = 104
        data = np.array([[20.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 104

    def test_high_dbz_encoding(self):
        # 60 dBZ -> (60+32)*2 = 184
        data = np.array([[60.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 184

    def test_negative_dbz_encoding(self):
        # -10 dBZ -> (-10+32)*2 = 44
        data = np.array([[-10.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 44

    def test_bad_data_returns_none(self):
        result = _parse_opera_hdf5(b"not hdf5")
        assert result is None

    def test_coverage_distinction(self):
        """Both nodata and undetect map to 0; only real precip is non-zero."""
        data = np.array([[-9999000.0, -8888000.0, 10.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 0    # nodata → 0
        assert result[0, 1] == 0    # undetect → 0 (gap-filler, let ECMWF fill)
        assert result[0, 2] >= 2    # real dBZ


class TestOperaSourceURL:
    def test_url_construction(self):
        src = OperaSource("https://example.com")
        # 2026-04-10 09:05 UTC → ts=1775991900
        from datetime import datetime, timezone
        dt = datetime(2026, 4, 10, 9, 5, tzinfo=timezone.utc)
        ts = int(dt.timestamp())
        url = src._url_for_timestamp(ts)
        assert "OPERA@20260410T0905@0@DBZH.h5" in url
        assert url.startswith("https://example.com/openradar-24h/2026/04/10/OPERA/COMP/")

    def test_url_rounds_to_5min(self):
        src = OperaSource("https://example.com")
        from datetime import datetime, timezone
        # 09:07 should round to 09:05
        dt = datetime(2026, 4, 10, 9, 7, tzinfo=timezone.utc)
        url = src._url_for_timestamp(int(dt.timestamp()))
        assert "T0905@" in url


class TestOperaSlotSelection:
    """Frames only ever use fetch_interval-aligned (:x0) composites.

    The :x5 composites lack AEMET's 10-minute fleet, so a frame built from
    one shows Spain dry (2026-10-19, Tarragona).
    """

    class _Resp:
        def __init__(self, status_code):
            self.status_code = status_code
            self.content = b""

    def _run(self, monkeypatch, now, available):
        import time as _time
        from datetime import datetime, timezone

        from librewxr.sources.regional.europe.radar.opera import source as mod

        requested: list[str] = []

        async def fake_get(client, url, log_name=""):
            slot = url.split("@")[1]
            requested.append(slot[-4:])
            return self._Resp(200 if slot[-4:] in available else 404)

        monkeypatch.setattr(mod, "retry_get", fake_get)
        monkeypatch.setattr(mod, "_parse_opera_hdf5", lambda data: np.zeros((1, 1), np.uint8))
        monkeypatch.setattr(mod.settings, "fetch_interval", 600)
        ts = datetime(2026, 10, 19, *now, tzinfo=timezone.utc).timestamp()
        monkeypatch.setattr(mod.time, "time", lambda: ts)

        import asyncio
        src = OperaSource("https://example.com")
        result = asyncio.run(src.fetch_frame(REGIONS["OPERA"], 0))
        return requested, result

    def test_live_frame_uses_its_own_x0_slot(self, monkeypatch):
        # Cycle at 10:16 for the frame labelled 10:10: never 10:15.
        requested, result = self._run(monkeypatch, (10, 16), {"1010", "1015"})
        assert requested == ["1010"]
        assert result is not None

    def test_fallback_steps_to_previous_x0_slot(self, monkeypatch):
        # 10:10 not yet published: take 10:00, skipping the partial 10:05.
        requested, result = self._run(monkeypatch, (10, 11), {"1000", "1005"})
        assert requested == ["1010", "1000"]
        assert result is not None

    def test_gives_up_after_one_fallback(self, monkeypatch):
        # Neither :x0 slot available: return None so the fetcher's
        # carry-forward/refetch path takes over, never an :x5 composite.
        requested, result = self._run(monkeypatch, (10, 11), {"1005", "0955"})
        assert requested == ["1010", "1000"]
        assert result is None

    def test_archive_frame_snaps_to_x0(self, monkeypatch):
        from datetime import datetime, timezone

        from librewxr.sources.regional.europe.radar.opera import source as mod

        requested: list[str] = []

        async def fake_get(client, url, log_name=""):
            requested.append(url.split("@")[1][-4:])
            return self._Resp(404)

        monkeypatch.setattr(mod, "retry_get", fake_get)
        monkeypatch.setattr(mod.settings, "fetch_interval", 600)
        import asyncio
        dt = datetime(2026, 10, 19, 9, 50, tzinfo=timezone.utc)
        asyncio.run(OperaSource("https://example.com").fetch_archive_frame(REGIONS["OPERA"], dt))
        assert requested == ["0950", "0940"]
