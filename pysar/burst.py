import xml.etree.ElementTree as ET
from datetime import datetime
from pysar import orbit


class Burst:
    def __init__(self):
        self.orbit = None
        self.start_time = None  # Start time as a datetime object
        self.range_time_to_first_pixel = None  # First pixel value as a float


def fromTSX(root: ET.Element) -> Burst:
    burst = Burst()

    # Find the <sceneInfo> tag
    product_info = root.find('productInfo')
    scene_info = product_info.find('sceneInfo')
    if scene_info is None:
        raise ValueError("No <sceneInfo> tag found in the XML document.")

    # Extract the start time (<start/timeUTC>)
    start_time_utc = scene_info.find('start/timeUTC')
    if start_time_utc is not None:
        time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
        burst.start_time = datetime.strptime(start_time_utc.text, time_format)

    # Extract the first pixel value (<rangeTime/firstPixel>)
    first_pixel = scene_info.find('rangeTime/firstPixel')
    if first_pixel is not None:
        burst.range_time_to_first_pixel = float(first_pixel.text)

    burst.orbit = orbit.fromTSX(root, burst.start_time)

    return burst
