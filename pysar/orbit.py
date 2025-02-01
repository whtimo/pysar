import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np


class Orbit:
    def __init__(self):
        self.times = []  # Time differences from reference time in seconds
        self.positions = []  # List of [posX, posY, posZ] lists
        self.velocities = []  # List of [velX, velY, velZ] lists

    def add_state(self, time_diff, position, velocity):
        self.times.append(time_diff)
        self.positions.append(position)
        self.velocities.append(velocity)


def fromTSX(root: ET.Element, reference_time: datetime) -> Orbit:
    orbit = Orbit()

    # Find the <orbit> tag
    platform = root.find('platform')
    orbit_tag = platform.find('orbit')
    if orbit_tag is None:
        raise ValueError("No <orbit> tag found in the XML document.")

    # Parse each <stateVec>
    for state_vec in orbit_tag.findall('stateVec'):
        time_utc = state_vec.find('timeUTC').text
        posX = float(state_vec.find('posX').text)
        posY = float(state_vec.find('posY').text)
        posZ = float(state_vec.find('posZ').text)
        velX = float(state_vec.find('velX').text)
        velY = float(state_vec.find('velY').text)
        velZ = float(state_vec.find('velZ').text)

        # Calculate time difference from reference time
        time_format = "%Y-%m-%dT%H:%M:%S.%f"
        current_time = datetime.strptime(time_utc, time_format)
        time_diff = (current_time - reference_time).total_seconds()

        # Add state to the orbit
        orbit.add_state(time_diff, [posX, posY, posZ], [velX, velY, velZ])

    return orbit
