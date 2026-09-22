# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
from typing import Any

from pydantic import BaseModel


class AlertProperties(BaseModel):
    title: str
    severity: str
    time: int | None
    expires: int | None
    description: str
    regions: list[str]
    uri: str


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    properties: AlertProperties
    geometry: dict[str, Any] | None


class AlertsResponse(BaseModel):
    type: str = "FeatureCollection"
    features: list[GeoJSONFeature]


class RadarTimestamp(BaseModel):
    time: int
    path: str


class RadarRegionInfo(BaseModel):
    """Geographic footprint of one radar composite region.

    ``bounds`` is [west, south, east, north] in degrees and is a
    bounding box, not an exact footprint — the projected regions (OPERA,
    ITCOMP, JPCOMP) cover a curved quadrilateral inside it.  ``px`` is
    the lon-axis pixel size: where regions overlap, the one with the
    smaller ``px`` is the one actually drawn, so a client resolving a
    point to its region has to apply the same rule.
    """

    id: str
    label: str
    bounds: list[float]
    px: float


class RegionCoverage(BaseModel):
    """When one region is not serving current observed radar."""

    # Frame timestamps served from an OLDER observation of this region
    # (carry-forward).  The echo is real but frozen.
    carried: list[int] = []
    # Frame timestamps with no data for this region at all.  The renderer
    # falls through to NWP fill there, so the client is looking at model
    # output rather than radar.
    absent: list[int] = []
    # Newest genuine observation of this region anywhere in the window.
    # May predate the oldest frame: a carry-forward chain outlives the
    # ring buffer.  None = never observed within the window.
    latestObserved: int | None = None


class RadarCoverage(BaseModel):
    """Per-region health, a LibreWXR extension Rain Viewer has no notion of."""

    regions: list[RadarRegionInfo]
    # Sparse: a region in good health is omitted entirely, so a healthy
    # server publishes an empty dict.
    degraded: dict[str, RegionCoverage] = {}


class ColorScheme(BaseModel):
    id: int
    name: str


class RadarData(BaseModel):
    past: list[RadarTimestamp]
    nowcast: list[RadarTimestamp]
    colorSchemes: list[ColorScheme]
    # Additive: Rain Viewer clients ignore unknown keys.  Nowcast frames
    # carry no coverage — they come from a separate store with no region
    # concept — though a client can reasonably infer that a prediction
    # extrapolated from a degraded frame is itself degraded.
    coverage: RadarCoverage | None = None


class SatelliteData(BaseModel):
    infrared: list[RadarTimestamp]


class WeatherMapsResponse(BaseModel):
    version: str
    generated: int
    host: str
    radar: RadarData
    satellite: SatelliteData
