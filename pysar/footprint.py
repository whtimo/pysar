import xml.etree.ElementTree as ET
from rasterio.windows import Window, bounds

class Footprint:
    def __init__(self, coords=None):
        # Combine all coordinates into a list
        if coords is None:
            coords = []

        if len(coords) == 4:
            # Sort by latitude (y-coordinate) in descending order to find top and bottom
            coords.sort(key=lambda x: x[0], reverse=True)

            # Top two coordinates are the upper ones
            upper_coords = coords[:2]
            # Bottom two coordinates are the lower ones
            lower_coords = coords[2:]

            # Sort upper coordinates by longitude (x-coordinate) to find upper left and upper right
            upper_coords.sort(key=lambda x: x[1])
            self.upper_left = upper_coords[0]
            self.upper_right = upper_coords[1]

            # Sort lower coordinates by longitude (x-coordinate) to find lower left and lower right
            lower_coords.sort(key=lambda x: x[1])
            self.lower_left = lower_coords[0]
            self.lower_right = lower_coords[1]
        else:
            self.upper_left = None
            self.upper_right = None
            self.lower_left = None
            self.lower_right = None

    def left(self):
        return min(self.upper_left[1], self.lower_left[1])

    def right(self):
        return max(self.upper_right[1], self.lower_right[1])

    def top(self):
        return max(self.upper_left[0], self.upper_right[0])

    def bottom(self):
        return min(self.lower_left[0], self.lower_right[0])

    def subset(self, window: Window, original_width: int, original_height:int):
        frac_x_ul = window.col_off / original_width
        frac_y_ul = window.row_off / original_height
        newul = interpolate_point(self.upper_left, self.upper_right, self.lower_left, self.lower_right, frac_y_ul, frac_x_ul)
        frac_x_ur = (window.col_off + window.width) / original_width
        frac_y_ur = window.row_off / original_height
        newur = interpolate_point(self.upper_left, self.upper_right, self.lower_left, self.lower_right, frac_y_ur,
                                  frac_x_ur)
        frac_x_ll = window.col_off / original_width
        frac_y_ll = (window.row_off + window.height) / original_height
        newll = interpolate_point(self.upper_left, self.upper_right, self.lower_left, self.lower_right, frac_y_ll, frac_x_ll)
        frac_x_lr = (window.col_off + window.width) / original_width
        frac_y_lr = (window.row_off + window.height) / original_height
        newlr = interpolate_point(self.upper_left, self.upper_right, self.lower_left, self.lower_right, frac_y_lr,
                                  frac_x_lr)
        fp = Footprint([newul, newur, newll, newlr])
        return fp

    def __str__(self):
        return (f"Footprint:\n"
                f"Upper Left: {self.upper_left}\n"
                f"Upper Right: {self.upper_right}\n"
                f"Lower Left: {self.lower_left}\n"
                f"Lower Right: {self.lower_right}\n"
                f"Left: {self.left()}, Right: {self.right()}, Top: {self.top()}, Bottom: {self.bottom()}")

    def toXml(self, root: ET.Element):
        # Save upper_left
        upper_left_elem = ET.SubElement(root, "CornerUpperLeft")
        ul_lat_elem = ET.SubElement(upper_left_elem, "Lat")
        ul_lat_elem.text = str(self.upper_left[0])
        ul_lon_elem = ET.SubElement(upper_left_elem, "Lon")
        ul_lon_elem.text = str(self.upper_left[1])

        # Save upper_right
        upper_right_elem = ET.SubElement(root, "CornerUpperRight")
        ur_lat_elem = ET.SubElement(upper_right_elem, "Lat")
        ur_lat_elem.text = str(self.upper_right[0])
        ur_lon_elem = ET.SubElement(upper_right_elem, "Lon")
        ur_lon_elem.text = str(self.upper_right[1])

        # Save lower_left
        lower_left_elem = ET.SubElement(root, "CornerLowerLeft")
        ll_lat_elem = ET.SubElement(lower_left_elem, "Lat")
        ll_lat_elem.text = str(self.lower_left[0])
        ll_lon_elem = ET.SubElement(lower_left_elem, "Lon")
        ll_lon_elem.text = str(self.lower_left[1])

        # Save lower_right
        lower_right_elem = ET.SubElement(root, "CornerLowerRight")
        lr_lat_elem = ET.SubElement(lower_right_elem, "Lat")
        lr_lat_elem.text = str(self.lower_right[0])
        lr_lon_elem = ET.SubElement(lower_right_elem, "Lon")
        lr_lon_elem.text = str(self.lower_right[1])

def interpolate_point(ul, ur, ll, lr, fraction_x, fraction_y):
    # Unpack the coordinates
    ul_lat, ul_lon = ul
    ur_lat, ur_lon = ur
    ll_lat, ll_lon = ll
    lr_lat, lr_lon = lr

    # Calculate top and lower points using fraction_x
    top_lat = ul_lat + fraction_x * (ur_lat - ul_lat)
    top_lon = ul_lon + fraction_x * (ur_lon - ul_lon)
    lower_lat = ll_lat + fraction_x * (lr_lat - ll_lat)
    lower_lon = ll_lon + fraction_x * (lr_lon - ll_lon)

    # Calculate left and right points using fraction_y
    left_lat = ul_lat + fraction_y * (ll_lat - ul_lat)
    left_lon = ul_lon + fraction_y * (ll_lon - ul_lon)
    right_lat = ur_lat + fraction_y * (lr_lat - ur_lat)
    right_lon = ur_lon + fraction_y * (lr_lon - ur_lon)

    # Compute direction vectors for the two lines
    dir1_lat = lower_lat - top_lat
    dir1_lon = lower_lon - top_lon
    dir2_lat = right_lat - left_lat
    dir2_lon = right_lon - left_lon

    # Setup the system of equations to solve for t and s
    a = dir1_lat
    b = -dir2_lat
    c = dir1_lon
    d = -dir2_lon
    e = left_lat - top_lat
    f_val = left_lon - top_lon

    # Calculate determinant
    det = a * d - b * c

    if det == 0:
        # Lines are parallel, handle accordingly (here, returning None)
        return None

    # Solve using Cramer's rule
    t = (e * d - b * f_val) / det
    s = (a * f_val - e * c) / det

    # Compute the intersection point using line 1 (top to lower)
    result_lat = top_lat + t * dir1_lat
    result_lon = top_lon + t * dir1_lon

    return (result_lat, result_lon)
def fromTSX(root: ET.Element) -> Footprint:
    # Find all <sceneCornerCoord> elements
    corner_coords = root.findall('.//sceneCornerCoord')

    # Extract latitude and longitude from each <sceneCornerCoord>
    coordinates = []
    for corner in corner_coords:
        lat = float(corner.find('lat').text)
        lon = float(corner.find('lon').text)
        coordinates.append((lat, lon))

    # Create a Footprint object using the extracted coordinates
    footprint = Footprint(coordinates)
    return footprint


def fromXml(root: ET.Element) -> Footprint:
    footprint = Footprint()
    # Load upper_left
    upper_left_elem = root.find("CornerUpperLeft")
    if upper_left_elem is not None:
        footprint.upper_left = [
            float(upper_left_elem.find("Lat").text),
            float(upper_left_elem.find("Lon").text)
        ]

    # Load upper_right
    upper_right_elem = root.find("CornerUpperRight")
    if upper_right_elem is not None:
        footprint.upper_right = [
            float(upper_right_elem.find("Lat").text),
            float(upper_right_elem.find("Lon").text)
        ]

    # Load lower_left
    lower_left_elem = root.find("CornerLowerLeft")
    if lower_left_elem is not None:
        footprint.lower_left = [
            float(lower_left_elem.find("Lat").text),
            float(lower_left_elem.find("Lon").text)
        ]

    # Load lower_right
    lower_right_elem = root.find("CornerLowerRight")
    if lower_right_elem is not None:
        footprint.lower_right = [
            float(lower_right_elem.find("Lat").text),
            float(lower_right_elem.find("Lon").text)
        ]

    return footprint

def fromDim(root: ET.Element) -> Footprint:
    coords = []
    coords.append((float(root.find(".//MDATTR[@name='first_near_lat']").text), float(root.find(".//MDATTR[@name='first_near_long']").text)))
    coords.append((float(root.find(".//MDATTR[@name='first_far_lat']").text),
                  float(root.find(".//MDATTR[@name='first_far_long']").text)))
    coords.append((float(root.find(".//MDATTR[@name='last_near_lat']").text),
                  float(root.find(".//MDATTR[@name='last_near_long']").text)))
    coords.append((float(root.find(".//MDATTR[@name='last_far_lat']").text),
                  float(root.find(".//MDATTR[@name='last_far_long']").text)))

    footprint = Footprint(coords)

    return footprint