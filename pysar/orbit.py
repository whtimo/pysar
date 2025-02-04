import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
import bisect
from scipy.interpolate import CubicHermiteSpline

class Orbit:
    def __init__(self):
        self.reference_time = None
        self.times = []  # Time differences from reference time in seconds
        self.positions = []  # List of [posX, posY, posZ] lists
        self.velocities = []  # List of [velX, velY, velZ] lists
        self._spline_x = None
        self._spline_y = None
        self._spline_z = None

    def seconds_from_reference_time(self, time):
        return (time - self.reference_time).total_seconds()

    def interpolate_position(self, time: float) -> np.array:
        # Check if the requested time is within the available data range
        if time < self.times[0] or time > self.times[-1]:
            raise ValueError(f"Time {time} is outside the available data range.")

        return np.array([self._spline_x(time), self._spline_y(time), self._spline_z(time)])

    def toXml(self, root: ET.Element):
        # Save reference_time
        if self.reference_time is not None:
            ref_time_elem = ET.SubElement(root, "reference_time")
            ref_time_elem.text = self.reference_time.isoformat()

        # Save times
        times_elem = ET.SubElement(root, "times")
        for time in self.times:
            time_elem = ET.SubElement(times_elem, "time")
            time_elem.text = str(time)

        # Save positions
        positions_elem = ET.SubElement(root, "positions")
        for pos in self.positions:
            pos_elem = ET.SubElement(positions_elem, "position")
            pos_elem.set("x", str(pos[0]))
            pos_elem.set("y", str(pos[1]))
            pos_elem.set("z", str(pos[2]))

        # Save velocities
        velocities_elem = ET.SubElement(root, "velocities")
        for vel in self.velocities:
            vel_elem = ET.SubElement(velocities_elem, "velocity")
            vel_elem.set("x", str(vel[0]))
            vel_elem.set("y", str(vel[1]))
            vel_elem.set("z", str(vel[2]))

def fromTSX(root: ET.Element) -> Orbit:
    orbit = Orbit()

    # Find the <orbit> tag
    platform = root.find('platform')
    orbit_tag = platform.find('orbit')

    if orbit_tag is None:
        raise ValueError("No <orbit> tag found in the XML document.")

    time_str = orbit_tag.find('orbitHeader/firstStateTime/firstStateTimeUTC').text
    time_format = "%Y-%m-%dT%H:%M:%S.%f"
    ta = datetime.strptime(time_str, time_format)
    ta2 = ta.replace(second=ta.second - 1)
    orbit.reference_time = ta2

    t = []
    p = []
    v = []

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

        current_time = datetime.strptime(time_utc, time_format)
        time_diff = (current_time - orbit.reference_time).total_seconds()

        t.append(time_diff)
        p.append([posX, posY, posZ])
        v.append([velX, velY, velZ])

    orbit.times = np.array(t)
    orbit.positions = np.array(p)  # Shape: (n, 3)
    orbit.velocities = np.array(v)  # Shape: (n, 3)
    orbit._spline_x = CubicHermiteSpline(orbit.times, orbit.positions[:, 0], orbit.velocities[:, 0])
    orbit._spline_y = CubicHermiteSpline(orbit.times, orbit.positions[:, 1], orbit.velocities[:, 1])
    orbit._spline_z = CubicHermiteSpline(orbit.times, orbit.positions[:, 2], orbit.velocities[:, 2])


    return orbit


def fromXml(root: ET.Element) -> Orbit:
    orbit = Orbit()

    # Load reference_time
    ref_time_elem = root.find("reference_time")
    if ref_time_elem is not None and ref_time_elem.text:
        orbit.reference_time = datetime.fromisoformat(ref_time_elem.text)

    t = []
    p = []
    v = []
    # Load times
    times_elem = root.find("times")
    if times_elem is not None:
        t = [float(time_elem.text) for time_elem in times_elem.findall("time")]

    # Load positions
    positions_elem = root.find("positions")
    if positions_elem is not None:
        p = [
            [float(pos_elem.get("x")), float(pos_elem.get("y")), float(pos_elem.get("z"))]
            for pos_elem in positions_elem.findall("position")
        ]

    # Load velocities
    velocities_elem = root.find("velocities")
    if velocities_elem is not None:
        v = [
            [float(vel_elem.get("x")), float(vel_elem.get("y")), float(vel_elem.get("z"))]
            for vel_elem in velocities_elem.findall("velocity")
        ]

    orbit.times = np.array(t)
    orbit.positions = np.array(p)  # Shape: (n, 3)
    orbit.velocities = np.array(v)  # Shape: (n, 3)
    orbit._spline_x = CubicHermiteSpline(orbit.times, orbit.positions[:, 0], orbit.velocities[:, 0])
    orbit._spline_y = CubicHermiteSpline(orbit.times, orbit.positions[:, 1], orbit.velocities[:, 1])
    orbit._spline_z = CubicHermiteSpline(orbit.times, orbit.positions[:, 2], orbit.velocities[:, 2])

    return orbit