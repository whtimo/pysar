import xml.etree.ElementTree as ET


class Footprint:
    def __init__(self, coord1, coord2, coord3, coord4):
        # Combine all coordinates into a list
        coords = [coord1, coord2, coord3, coord4]

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

    def left(self):
        return min(self.upper_left[1], self.lower_left[1])

    def right(self):
        return max(self.upper_right[1], self.lower_right[1])

    def top(self):
        return max(self.upper_left[0], self.upper_right[0])

    def bottom(self):
        return min(self.lower_left[0], self.lower_right[0])

    def __str__(self):
        return (f"Footprint:\n"
                f"Upper Left: {self.upper_left}\n"
                f"Upper Right: {self.upper_right}\n"
                f"Lower Left: {self.lower_left}\n"
                f"Lower Right: {self.lower_right}\n"
                f"Left: {self.left()}, Right: {self.right()}, Top: {self.top()}, Bottom: {self.bottom()}")


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
    footprint = Footprint(*coordinates)
    return footprint
