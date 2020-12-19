"""Econding methods for extra channels"""
import math

def encode_lon(lon):
    """Encode longitude across the globe from [-1, 1] using two features"""
    assert lon <= 180
    assert lon >= -180

    lon = lon + 180
    lon = lon / 360

    return math.sin(2 * math.pi * lon), math.cos(2 * math.pi * lon)


def encode_lat(lat):
    """Encode latitude across the globe from [-1, 1] using two features"""
    assert lat <= 90
    assert lat >= -90

    lat = lat + 90
    lat = lat / 180

    return math.sin(2 * math.pi * lat), math.cos(2 * math.pi * lat)

def encode_hour(d):
    """Encode hour within a day from [-1, 1] using two features"""
    if isinstance(d, np.datetime64):
        d = datetime.utcfromtimestamp(d.astype(int) * 1e-9)  # 1e-9 is the number of seconds in a nanosecond
    assert isinstance(d, datetime)

    date_midnight = datetime(year=d.year, month=d.month, day=d.day, hour=0, minute=0, second=0, microsecond=0)
    seconds = (d - date_midnight).total_seconds()
    seconds = seconds / (24 * 60 * 60)

    return math.sin(2 * math.pi * seconds), math.cos(2 * math.pi * seconds)


def encode_date(d):
    """Encode date within a year from [-1, 1] using two features"""
    if isinstance(d, np.datetime64):
        d = datetime.utcfromtimestamp(d.astype(int) * 1e-9)  # 1e-9 number of seconds in a nanosecond
    assert isinstance(d, datetime)

    end_of_year = datetime(year=d.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    begin_of_year = datetime(year=d.year, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    seconds_per_year = (end_of_year - begin_of_year).total_seconds()
    second_in_year = (d - begin_of_year).total_seconds()

    c = second_in_year / seconds_per_year

    return math.sin(2 * math.pi * c), math.cos(2 * math.pi * c)


def encode_lev(lev, minimum=10, maximum=100000, mode="linear"):
    """Encode level data either linear or logarithmic between [-1, 1] using one feature"""
    assert mode.lower() in ["linear", "lin", "log", "logarithm"]

    if mode.lower() in ["log", "logarithm"]:
        lev = np.log(lev)
        minimum, maximum = np.log(minimum), np.log(maximum)

    return ((lev - minimum) / (maximum - minimum)) * 2 - 1