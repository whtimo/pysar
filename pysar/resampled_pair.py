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

        if filepath:
            root = ET.parse(filepath).getroot()
            pair_elem = root.find("ResampledPair")
            if pair_elem:
                self.perpendicular_baseline =  float(pair_elem.attrib['perpendicular_baseline'])
                self.temporal_baseline = int(pair_elem.attrib['temporal_baseline'])
                self.master = slc.Slc()
                self.master.metadata = metadata.fromXml(pair_elem.find("Master/MetaData"))
                self.master.slcdata = cpl_float_slcdata.fromXml(pair_elem.find("Master"), filepath)
                self.slave = slc.Slc()
                self.slave.metadata = metadata.fromXml(pair_elem.find("Slave/MetaData"))
                self.slave.slcdata = cpl_float_slcdata.fromXml(pair_elem.find("Slave"), filepath)


    def save(self,  directory: str = None, filename: str = None, master_tiff_filename: str = None, slave_tiff_filename: str = None, overwrite: bool = False)-> pathlib.Path:
        xml_filename = ''
        master_tiff_fn = ''
        slave_tiff_fn = ''

        if directory is None:
            if filename is not None and master_tiff_filename is not None and slave_tiff_filename is not None:
                xml_filename = filename
                master_tiff_fn = master_tiff_filename
                slave_tiff_fn = slave_tiff_filename
        else:
            counter = 0
            xml_filename = pathlib.Path(
                directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}__{self.slave.metadata.sensor}_{self.slave.metadata.acquisition_date.isoformat()}.pysar.resampled.xml'

            while not overwrite and xml_filename.exists():
                counter += 1
                xml_filename = pathlib.Path(
                    directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}__{self.slave.metadata.sensor}_{self.slave.metadata.acquisition_date.isoformat()}.pysar.resampled.xml'

            if master_tiff_filename is not None:
                master_tiff_fn = master_tiff_filename
            else:
                master_tiff_fn = pathlib.Path(
                    directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}.slc.tiff'

            if slave_tiff_filename is not None:
                slave_tiff_fn = slave_tiff_filename
            else:
                slave_tiff_fn = pathlib.Path(
                    directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}__{self.slave.metadata.sensor}_{self.slave.metadata.acquisition_date.isoformat()}.slc.resampled.tiff'
                while not overwrite and slave_tiff_fn.exists():
                    counter += 1
                    slave_tiff_fn = pathlib.Path(
                        directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}__{self.slave.metadata.sensor}_{self.slave.metadata.acquisition_date.isoformat()}.slc.resampled.tiff'

        if len(str(xml_filename)) > 0:
            root = ET.Element("PySar")
            pair_elem = ET.SubElement(root, "ResampledPair")
            pair_elem.attrib["perpendicular_baseline"] = str(self.perpendicular_baseline)
            pair_elem.attrib["temporal_baseline"] = str(self.temporal_baseline)
            master_elem = ET.SubElement(pair_elem, "Master")
            self.master.metadata.toXml(master_elem)
            if not pathlib.Path(master_tiff_fn).exists():
                self.master.slcdata.saveTiff(master_tiff_fn, self.master.slcdata.read())
            self.master.slcdata.toXml(master_elem, pathlib.Path(master_tiff_fn).relative_to(pathlib.Path(xml_filename).parent))

            slave_elem = ET.SubElement(pair_elem, "Slave")
            self.slave.metadata.toXml(slave_elem)
            if overwrite or not pathlib.Path(slave_tiff_fn).exists():
                self.slave.slcdata.saveTiff(slave_tiff_fn, self.slave.slcdata.read())
            self.slave.slcdata.toXml(slave_elem, pathlib.Path(slave_tiff_fn).relative_to(pathlib.Path(xml_filename).parent))

            xml_str = ET.tostring(root, encoding="utf-8")
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

            # Write the pretty-printed XML to a file
            with open(xml_filename, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

            return pathlib.Path(master_tiff_fn)
        else:
            return pathlib.Path()

def createResampledPair(master: slc.Slc, slave: slc.Slc, resampled_slave_data: np.ndarray, base_line: baseline.Baseline = None):
    pair = ResampledPair()
    pair.master = master
    pair.slave = slave
    pair.slave.slcdata = cpl_float_memory_slcdata.CplFloatMemorySlcData(resampled_slave_data)
    if base_line is None:
        base_line = baseline.Baseline(master.metadata, slave.metadata)

    pair.perpendicular_baseline = base_line.perpendicular_baseline(master.metadata.number_columns / 2, master.metadata.number_rows / 2)
    pair.temporal_baseline = base_line.temporal_baseline

    shift = coregistration.orbit_shift(master.metadata, slave.metadata)
    pair.shift_x = shift[0]
    pair.shift_y = shift[1]

    return pair

def fromBzarXml(xml_path: str) -> ResampledPair:
    pair = ResampledPair()

    root = ET.parse(xml_path).getroot()
    pair_elem = root.find("ResampledPair")
    if pair_elem:
        pair.perpendicular_baseline = float(pair_elem.attrib['perpendicularBaseline'])
        pair.temporal_baseline = int(pair_elem.attrib['temporalBaseline'])
        pair.master = slc.Slc()
        pair.master.metadata = metadata.fromBzarXml(pair_elem.find("MasterSlcImage/Band"))
        pair.master.slcdata = cpl_float_slcdata.fromXml(pair_elem.find("MasterSlcImage"), xml_path)
        pair.slave = slc.Slc()
        pair.slave.metadata = metadata.fromBzarXml(pair_elem.find("SlaveSlcImage/Band"))
        pair.slave.slcdata = cpl_float_slcdata.fromXml(pair_elem.find("SlaveSlcImage"), xml_path)

    return pair

# def createFilenames(pair: ResampledPair, directory:str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
#     xml_path = pathlib.Path(directory) / f'{pair.master.metadata.sensor}_{pair.master.metadata.acquisition_date.isoformat()}__{pair.slave.metadata.sensor}_{pair.slave.metadata.acquisition_date.isoformat()}.pysar.resampled.xml'
#     master_tiff_path = pathlib.Path(directory) / f'{pair.master.metadata.sensor}_{pair.master.metadata.acquisition_date.isoformat()}.slc.tiff'
#     slave_tiff_path = pathlib.Path(directory) / f'{pair.master.metadata.sensor}_{pair.master.metadata.acquisition_date.isoformat()}__{pair.slave.metadata.sensor}_{pair.slave.metadata.acquisition_date.isoformat()}.slc.resampled.tiff'
#
#     return xml_path, master_tiff_path, slave_tiff_path