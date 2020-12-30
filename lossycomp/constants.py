"""Constants"""

from collections import namedtuple

Region = namedtuple("Regions", "max_lat, min_lat, min_lon, max_lon")

REGIONS = {
    "globe": Region(-90, 90, -180, 180),
    "europe": Region(34, 74, -16, 33),
    "germany": Region(47, 55, 6, 16),
    "north_atlantic": Region(0, 52, -70, -18),
    "sahara_desert": Region(15, 30, -15, 38),
    "greenland": Region(60, 80, -60, -20),
    "tibet": Region(28, 40, 78, 100),
    "tropical_zone": Region(-23.5, 23.5, -180, 180),
    "subtropical_zone": Region(23.5, 40, -180, 180),
    "temperate_zone": Region(40,66.5, -180, 180),
    "polar_zone": Region(66.5, 90, -180, 180),
    "pacific_ocean": Region(-70, 20, -180, -90)
}
