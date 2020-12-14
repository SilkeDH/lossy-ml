"""Constants"""

from collections import namedtuple

Region = namedtuple("Regions", "max_lat, min_lat, min_lon, max_lon")

REGIONS = {
    "globe": Region(-90, 90, -180, 180),
    "europe": Region(34, 74, -16, 33),
    "germany": Region(47, 55, 6, 16),
    "north_atlantic": Region(0, 52, -70, -18),
}
