import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
from scipy.interpolate import CubicHermiteSpline, lagrange, CubicSpline

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
        #if time < self.times[0] or time > self.times[-1]:
            #raise ValueError(f"Time {time} is outside the available data range.")

        return np.array([self._spline_x(time), self._spline_y(time), self._spline_z(time)])

    def toXml(self, root: ET.Element):
        # Save reference_time
        if self.reference_time is not None:
            ref_time_elem = ET.SubElement(root, "ReferenceTime")
            ref_time_elem.text = self.reference_time.isoformat()

        for ix in range(len(self.times)):
            time = self.times[ix]
            position = self.positions[ix]
            velocity = self.velocities[ix]
            vec_elem = ET.SubElement(root, "Vector")
            vec_elem.attrib["SecondsFromReference"] = str(time)
            pos_elem = ET.SubElement(vec_elem, "Position")
            x_elem = ET.SubElement(pos_elem, "X")
            x_elem.text = str(position[0])
            y_elem = ET.SubElement(pos_elem, "Y")
            y_elem.text = str(position[1])
            z_elem = ET.SubElement(pos_elem, "Z")
            z_elem.text = str(position[2])
            vel_elem = ET.SubElement(vec_elem, "Velocity")
            vx_elem = ET.SubElement(vel_elem, "X")
            vx_elem.text = str(velocity[0])
            vy_elem = ET.SubElement(vel_elem, "Y")
            vy_elem.text = str(velocity[1])
            vz_elem = ET.SubElement(vel_elem, "Z")
            vz_elem.text = str(velocity[2])


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
    # orbit._spline_x = CubicHermiteSpline(orbit.times, orbit.positions[:, 0], orbit.velocities[:, 0])
    # orbit._spline_y = CubicHermiteSpline(orbit.times, orbit.positions[:, 1], orbit.velocities[:, 1])
    # orbit._spline_z = CubicHermiteSpline(orbit.times, orbit.positions[:, 2], orbit.velocities[:, 2])
    # orbit._spline_x = CubicSpline(orbit.times, orbit.positions[:, 0])
    # orbit._spline_y = CubicSpline(orbit.times, orbit.positions[:, 1])
    # orbit._spline_z = CubicSpline(orbit.times, orbit.positions[:, 2])
    orbit._spline_x = lagrange(orbit.times, orbit.positions[:, 0])
    orbit._spline_y = lagrange(orbit.times, orbit.positions[:, 1])
    orbit._spline_z = lagrange(orbit.times, orbit.positions[:, 2])

    return orbit


def fromXml(root: ET.Element, ref_time: datetime = None) -> Orbit:
    orbit = Orbit()
    if ref_time is None:
        # Load reference_time
        ref_time_elem = root.find("ReferenceTime")
        if ref_time_elem is not None and ref_time_elem.text:
            orbit.reference_time = datetime.fromisoformat(ref_time_elem.text)
    else:
        orbit.reference_time = ref_time

    t = []
    p = []
    v = []
    # Load times
    for vector_elem in root.findall("Vector"):
        t.append(float(vector_elem.attrib["SecondsFromReference"]))
        pos_elem = vector_elem.find("Position")
        p.append([float(pos_elem.find("X").text), float(pos_elem.find("Y").text), float(pos_elem.find("Z").text)])
        vel_elem = vector_elem.find("Velocity")
        v.append([float(vel_elem.find("X").text), float(vel_elem.find("Y").text), float(vel_elem.find("Z").text)])

    orbit.times = np.array(t)
    orbit.positions = np.array(p)  # Shape: (n, 3)
    orbit.velocities = np.array(v)  # Shape: (n, 3)
    # orbit._spline_x = CubicHermiteSpline(orbit.times, orbit.positions[:, 0], orbit.velocities[:, 0])
    # orbit._spline_y = CubicHermiteSpline(orbit.times, orbit.positions[:, 1], orbit.velocities[:, 1])
    # orbit._spline_z = CubicHermiteSpline(orbit.times, orbit.positions[:, 2], orbit.velocities[:, 2])
    orbit._spline_x = lagrange(orbit.times, orbit.positions[:, 0])
    orbit._spline_y = lagrange(orbit.times, orbit.positions[:, 1])
    orbit._spline_z = lagrange(orbit.times, orbit.positions[:, 2])

    return orbit

def fromDim(root: ET.Element, ref_time: datetime) -> Orbit:
    orbit = Orbit()
    orbit.reference_time = ref_time

    t = []
    p = []
    v = []

    # Load times, positions, and velocities
    for md_elem in root.findall("MDElem"):
        time_str = md_elem.find("MDATTR[@name='time']").text
        time = datetime.strptime(time_str, "%d-%b-%Y %H:%M:%S.%f")
        delta_time = (time - ref_time).total_seconds()
        t.append(delta_time)

        x_pos = float(md_elem.find("MDATTR[@name='x_pos']").text)
        y_pos = float(md_elem.find("MDATTR[@name='y_pos']").text)
        z_pos = float(md_elem.find("MDATTR[@name='z_pos']").text)
        p.append([x_pos, y_pos, z_pos])

        x_vel = float(md_elem.find("MDATTR[@name='x_vel']").text)
        y_vel = float(md_elem.find("MDATTR[@name='y_vel']").text)
        z_vel = float(md_elem.find("MDATTR[@name='z_vel']").text)
        v.append([x_vel, y_vel, z_vel])

    orbit.times = np.array(t)
    orbit.positions = np.array(p)  # Shape: (n, 3)
    orbit.velocities = np.array(v)  # Shape: (n, 3)

    # Create splines for each dimension
    # orbit._spline_x = CubicHermiteSpline(orbit.times, orbit.positions[:, 0], orbit.velocities[:, 0])
    # orbit._spline_y = CubicHermiteSpline(orbit.times, orbit.positions[:, 1], orbit.velocities[:, 1])
    # orbit._spline_z = CubicHermiteSpline(orbit.times, orbit.positions[:, 2], orbit.velocities[:, 2])
    orbit._spline_x = lagrange(orbit.times, orbit.positions[:, 0])
    orbit._spline_y = lagrange(orbit.times, orbit.positions[:, 1])
    orbit._spline_z = lagrange(orbit.times, orbit.positions[:, 2])

    return orbit