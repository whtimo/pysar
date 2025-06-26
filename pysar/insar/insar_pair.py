from pysar.insar import coregistration, baseline
from pysar.sar import slc, cpl_float_slcdata, metadata
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pathlib
import os

class InSarPair:
    def __init__(self, filepath:str = None):
        self.master = None
        self.slave = None
        self.perpendicular_baseline = None
        self.temporal_baseline = None
        self.shift_x = None
        self.shift_y = None


        if filepath:
            root = ET.parse(filepath).getroot()
            pair_elem = root.find("Pair")
            if pair_elem:
                self.perpendicular_baseline =  float(pair_elem.attrib['perpendicular_baseline'])
                self.temporal_baseline = int(pair_elem.attrib['temporal_baseline'])
                self.shift_x = float(pair_elem.attrib['shift_x'])
                self.shift_y = float(pair_elem.attrib['shift_y'])
                self.master = slc.Slc()
                self.master.metadata = metadata.fromXml(pair_elem.find("Master/MetaData"))
                self.master.slcdata = cpl_float_slcdata.fromXml(pair_elem.find("Master"), filepath)
                self.slave = slc.Slc()
                self.slave.metadata = metadata.fromXml(pair_elem.find("Slave/MetaData"))
                self.slave.slcdata = cpl_float_slcdata.fromXml(pair_elem.find("Slave"), filepath)


    def __getTiffName(self, metadata, path, overwrite: bool = False):
        counter = 0
        tiff_name = f'{metadata.sensor}_{counter}_{metadata.acquisition_date.isoformat()}.slc.tiff'
        while (pathlib.Path(path) / tiff_name).exists() and not overwrite:
            counter += 1
            tiff_name = f'{metadata.sensor}_{counter}_{metadata.acquisition_date.isoformat()}.slc.tiff'

        return tiff_name

    def save(self,  directory: str = None, filename: str = None, master_tiff_filename: str = None, slave_tiff_filename: str = None, overwrite: bool = False):
        """
        Saves the InSAR pair data

        :param directory: Save into this directory with automatic created filenames (defaut)
        :param filename: Filename for the xml file (requires the directory to not be set)
        :param master_tiff_filename: Filename for the master tiff file (requires the directory to not be set). If directory is not set and the tiff file name is not set, the tiff files will not be saved. Also, if saveiff is False, they will not be saved.
        :param slave_tiff_filename: Filename for the slave tiff file (requires the directory to not be set) If directory is not set and the tiff file name is not set, the tiff files will not be saved. Also, if saveiff is False, they will not be saved.
        """

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
                directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}__{self.slave.metadata.sensor}_{self.slave.metadata.acquisition_date.isoformat()}.pysar.pair.xml'

            while not overwrite and xml_filename.exists():
                counter += 1
                xml_filename = pathlib.Path(
                    directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}__{self.slave.metadata.sensor}_{self.slave.metadata.acquisition_date.isoformat()}.pysar.pair.xml'

            if master_tiff_filename is not None:
                master_tiff_fn = master_tiff_filename
            else:
                master_tiff_fn = pathlib.Path(
                    directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}.slc.tiff'

            if slave_tiff_filename is not None:
                slave_tiff_fn = slave_tiff_filename
            else:
                slave_tiff_fn = pathlib.Path(
                    directory) / f'{self.slave.metadata.sensor}_{counter}_{self.slave.metadata.acquisition_date.isoformat()}.slc.tiff'


        if len(str(xml_filename)) > 0:
            root = ET.Element("PySar")
            pair_elem = ET.SubElement(root, "Pair")
            pair_elem.attrib["perpendicular_baseline"] = str(self.perpendicular_baseline)
            pair_elem.attrib["temporal_baseline"] = str(self.temporal_baseline)
            pair_elem.attrib["shift_x"] = str(self.shift_x)
            pair_elem.attrib["shift_y"] = str(self.shift_y)


            master_elem = ET.SubElement(pair_elem, "Master")
            self.master.metadata.toXml(master_elem)
            if not pathlib.Path(master_tiff_fn).exists():
                self.master.slcdata.saveTiff(master_tiff_fn, self.master.slcdata.read())
            self.master.slcdata.toXml(master_elem, pathlib.Path(master_tiff_fn).relative_to(pathlib.Path(xml_filename).parent) )

            slave_elem = ET.SubElement(pair_elem, "Slave")
            self.slave.metadata.toXml(slave_elem)
            if not pathlib.Path(slave_tiff_fn).exists():
                self.slave.slcdata.saveTiff(slave_tiff_fn, self.slave.slcdata.read())
            self.slave.slcdata.toXml(slave_elem,
                                          pathlib.Path(slave_tiff_fn).relative_to(pathlib.Path(xml_filename).parent))

            xml_str = ET.tostring(root, encoding="utf-8")
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

            # Write the pretty-printed XML to a file
            with open(xml_filename, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

def createInSarPair(master: slc.Slc, slave: slc.Slc, base_line: baseline.Baseline = None):
    pair = InSarPair()
    pair.master = master
    pair.slave = slave
    if base_line is None:
        base_line = baseline.Baseline(master.metadata, slave.metadata)

    pair.perpendicular_baseline = base_line.perpendicular_baseline(master.metadata.number_columns / 2, master.metadata.number_rows / 2)
    pair.temporal_baseline = base_line.temporal_baseline

    shift = coregistration.orbit_shift(master.metadata._burst, slave.metadata._burst)
    pair.shift_x = shift[0]
    pair.shift_y = shift[1]

    return pair



def fromBzarXml(xml_path: str) -> InSarPair:
    pair = InSarPair()

    root = ET.parse(xml_path).getroot()
    pair_elem = root.find("CoregistrationPair")
    if pair_elem:
        pair.shift_x = float(pair_elem.attrib['offset_sample'])
        pair.shift_y = float(pair_elem.attrib['offset_row'])
        pair.master = slc.Slc()
        pair.master.metadata = metadata.fromBzarXml(pair_elem.find("MasterSlcImage/Band"))
        pair.master.slcdata = cpl_float_slcdata.fromXml(pair_elem.find("MasterSlcImage"), xml_path)
        pair.slave = slc.Slc()
        pair.slave.metadata = metadata.fromBzarXml(pair_elem.find("SlaveSlcImage/Band"))
        pair.slave.slcdata = cpl_float_slcdata.fromXml(pair_elem.find("SlaveSlcImage"), xml_path)

        base = baseline.Baseline(pair.master.metadata, pair.slave.metadata)
        pair.perpendicular_baseline = base.perpendicular_baseline(pair.master.metadata.number_columns / 2, pair.master.metadata.number_rows / 2)
        pair.temporal_baseline = base.temporal_baseline

    return pair

