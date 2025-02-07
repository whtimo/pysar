import numpy as np
from pysar import slc, baseline, metadata, coregistration, cpl_float_slcdata, cpl_float_memory_slcdata
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pathlib

class ResampledPair:
    def __init__(self, filepath:str = None):
        self.master = None
        self.resampled_slave = None
        self.perpendicular_baseline = None
        self.temporal_baseline = None
        self.shift_x = None
        self.shift_y = None

        if filepath:
            root = ET.parse(filepath).getroot()
            pair_elem = root.find("ResampledPair")
            if pair_elem:
                self.perpendicular_baseline =  float(pair_elem.attrib['perpendicular_baseline'])
                self.temporal_baseline = int(pair_elem.attrib['temporal_baseline'])
                self.shift_x = float(pair_elem.attrib['shift_x'])
                self.shift_y = float(pair_elem.attrib['shift_y'])
                self.master = slc.Slc()
                self.master.metadata = metadata.fromBzarXml(pair_elem.find("Master"))
                self.master.slcdata = cpl_float_slcdata.fromXml(pair_elem.find("Master"), filepath)
                self.slave = slc.Slc()
                self.slave.metadata = metadata.fromBzarXml(pair_elem.find("ResampledSlave"))
                self.slave.slcdata = cpl_float_slcdata.fromXml(pair_elem.find("ResampledSlave"), filepath)


    def save(self, filepath:pathlib.Path, master_tiff_path:pathlib.Path, slave_tiff_path: pathlib.Path):
        root = ET.Element("PySar")
        pair_elem = ET.SubElement(root, "ResampledPair")
        pair_elem.attrib["perpendicular_baseline"] = str(self.perpendicular_baseline)
        pair_elem.attrib["temporal_baseline"] = str(self.temporal_baseline)
        pair_elem.attrib["shift_x"] = str(self.shift_x)
        pair_elem.attrib["shift_y"] = str(self.shift_y)
        master_elem = ET.SubElement(pair_elem, "Master")
        self.master.metadata.toXml(master_elem)
        self.master.slcdata.toXml(master_elem, master_tiff_path.relative_to(filepath.parent))

        slave_elem = ET.SubElement(pair_elem, "Slave")
        self.slave.metadata.toXml(slave_elem)
        self.slave.slcdata.toXml(slave_elem, slave_tiff_path.relative_to(filepath.parent))

        xml_str = ET.tostring(root, encoding="utf-8")
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

        # Write the pretty-printed XML to a file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

def createResampledPair(master: slc.Slc, slave: slc.Slc, resampled_slave_data: np.ndarray, bese_line: baseline.Baseline = None):
    pair = ResampledPair()
    pair.master = master
    pair.slave = slave
    pair.slave.slcdata = cpl_float_memory_slcdata.CplFloatMemorySlcData(resampled_slave_data)
    if bese_line is None:
        base_line = baseline.Baseline(master.metadata, slave.metadata)

    pair.perpendicular_baseline = base_line.perpendicular_baseline(master.metadata.number_columns / 2, master.metadata.number_rows / 2)
    pair.temporal_baseline = base_line.temporal_baseline

    shift = coregistration.orbit_shift(master.metadata.burst, slave.metadata.burst)
    pair.shift_x = shift[0]
    pair.shift_y = shift[1]

    return pair

def createFilenames(pair: ResampledPair, directory:str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    xml_path = pathlib.Path(directory) / f'{pair.master.metadata.sensor}_{pair.master.metadata.acquisition_date.isoformat()}__{pair.slave.metadata.sensor}_{pair.slave.metadata.acquisition_date.isoformat()}.pysar.resampled.xml'
    master_tiff_path = pathlib.Path(directory) / f'{pair.master.metadata.sensor}_{pair.master.metadata.acquisition_date.isoformat()}.slc.tiff'
    slave_tiff_path = pathlib.Path(directory) / f'{pair.master.metadata.sensor}_{pair.master.metadata.acquisition_date.isoformat()}__{pair.slave.metadata.sensor}_{pair.slave.metadata.acquisition_date.isoformat()}.slc.resampled.tiff'

    return xml_path, master_tiff_path, slave_tiff_path