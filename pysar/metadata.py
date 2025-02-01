import xml.etree.ElementTree as ET
from pysar import burst


#Single Burst MetaData
class MetaData:
    def __init__(self):
        self.burst = None

def fromTSX(xml_path: str) -> MetaData:
    meta = MetaData()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    meta.burst = burst.fromTSX(root)

    return meta