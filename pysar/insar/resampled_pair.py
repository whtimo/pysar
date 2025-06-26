import numpy as np
from pysar.insar import coregistration, baseline
from pysar.sar import slc, cpl_float_slcdata, iq_float_slcdata, metadata, cpl_float_memory_slcdata
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pathlib
from collections import defaultdict
import os
from rasterio.windows import Window


class ResampledPair:
    def __init__(self, filepath:str = None):
        self.master = None
        self.resampled_slave = None
        self.perpendicular_baseline = None
        self.temporal_baseline = None
        self.bistatic = False

        if filepath:
            root = ET.parse(filepath).getroot()
            pair_elem = root.find("ResampledPair")
            if pair_elem:
                self.perpendicular_baseline =  float(pair_elem.attrib['perpendicular_baseline'])
                self.temporal_baseline = int(pair_elem.attrib['temporal_baseline'])
                self.bistatic = bool(pair_elem.attrib['bistatic'])
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
                directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}__{self.slave.metadata.sensor}_{counter}_{self.slave.metadata.acquisition_date.isoformat()}.pysar.resampled.xml'

            while not overwrite and xml_filename.exists():
                counter += 1
                xml_filename = pathlib.Path(
                    directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}__{self.slave.metadata.sensor}_{counter}_{self.slave.metadata.acquisition_date.isoformat()}.pysar.resampled.xml'

            if master_tiff_filename is not None:
                master_tiff_fn = master_tiff_filename
            else:
                master_tiff_fn = pathlib.Path(
                    directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}.slc.tiff'

            if slave_tiff_filename is not None:
                slave_tiff_fn = slave_tiff_filename
            else:
                slave_tiff_fn = pathlib.Path(
                    directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}__{self.slave.metadata.sensor}_{counter}_{self.slave.metadata.acquisition_date.isoformat()}.slc.resampled.tiff'
                while not overwrite and slave_tiff_fn.exists():
                    counter += 1
                    slave_tiff_fn = pathlib.Path(
                        directory) / f'{self.master.metadata.sensor}_{counter}_{self.master.metadata.acquisition_date.isoformat()}__{self.slave.metadata.sensor}_{counter}_{self.slave.metadata.acquisition_date.isoformat()}.slc.resampled.tiff'

        if len(str(xml_filename)) > 0:
            root = ET.Element("PySar")
            pair_elem = ET.SubElement(root, "ResampledPair")
            pair_elem.attrib["perpendicular_baseline"] = str(self.perpendicular_baseline)
            pair_elem.attrib["temporal_baseline"] = str(self.temporal_baseline)
            if self.bistatic:
                pair_elem.attrib["bistatic"] = "true"
            else:
                pair_elem.attrib["bistatic"] = "false"

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

    def subset(self, window: Window, create_rpc = True):
        new_master_slc = slc.Slc()
        new_master_slc.metadata = self.master.metadata.subset(window, create_rpc=create_rpc)
        new_master_slc.slcdata = self.master.slcdata.subset(window)
        new_slave_slc = slc.Slc()
        new_slave_slc.metadata = self.slave.metadata.subset(window, create_rpc=create_rpc)
        new_slave_slc.slcdata = self.slave.slcdata.subset(window)

        pair = ResampledPair()
        pair.master = new_master_slc
        pair.slave = new_slave_slc
        base_line = baseline.Baseline(new_master_slc.metadata, new_slave_slc.metadata)

        pair.perpendicular_baseline = base_line.perpendicular_baseline(new_master_slc.metadata.number_columns / 2,
                                                                       new_master_slc.metadata.number_rows / 2)
        pair.temporal_baseline = base_line.temporal_baseline

        return pair

def createResampledPair(master: slc.Slc, slave: slc.Slc, resampled_slave_data: np.ndarray, base_line: baseline.Baseline = None):
    pair = ResampledPair()
    pair.master = master
    pair.slave = slave
    pair.slave.slcdata = cpl_float_memory_slcdata.CplFloatMemorySlcData(resampled_slave_data)
    if base_line is None:
        base_line = baseline.Baseline(master.metadata, slave.metadata)

    pair.perpendicular_baseline = base_line.perpendicular_baseline(master.metadata.number_columns / 2, master.metadata.number_rows / 2)
    pair.temporal_baseline = base_line.temporal_baseline

    # shift = coregistration.orbit_shift(master.metadata, slave.metadata)
    # pair.shift_x = shift[0]
    # pair.shift_y = shift[1]

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

def fromDim(dim_path: str):
    result = []

    root = ET.parse(dim_path).getroot()
    data_elem = root.find("Data_Access")

    data_dict = defaultdict(dict)

    for data_file in data_elem.findall(".//Data_File"):
        file_path = data_file.find("DATA_FILE_PATH").attrib["href"]
        filename = file_path.split("/")[-1]  # Extract the filename from the path
        parts = filename.split("_")
        date = parts[-1].replace(".hdr", "")  # Extract the date from the filename

        # Determine if it's a master or slave file
        if "_mst_" in filename:
            key = "master"
        elif "_slv" in filename:
            key = date
        else:
            continue  # Skip if it's neither master nor slave

        # Determine if it's an 'i' or 'q' file
        if filename.startswith("i_"):
            component = "i"
        elif filename.startswith("q_"):
            component = "q"
        else:
            continue  # Skip if it's neither 'i' nor 'q'

        # Add the filename to the result dictionary
        if key not in data_dict:
            data_dict[key] = {}
        data_dict[key][component] = (pathlib.Path(dim_path).parent / file_path).with_suffix(".img")


    sources_elem = root.find("Dataset_Sources")
    master_meta_elem = None
    slave_meta_elems = None

    for mdelem in sources_elem.findall("MDElem"):
        if mdelem.attrib["name"] == "metadata":
            for mdel in mdelem.findall("MDElem"):
                if mdel.attrib["name"] == "Abstracted_Metadata":
                    master_meta_elem = mdel
                if mdel.attrib["name"] == "Slave_Metadata":
                    slave_meta_elems = mdel

    if master_meta_elem is not None and slave_meta_elems is not None:
        for slv_meta_elem in slave_meta_elems.findall("MDElem"):
            pair = ResampledPair()
            pair.master = slc.Slc()
            pair.master.metadata = metadata.fromDim(master_meta_elem)
            pair.master.slcdata = iq_float_slcdata.IqFloatSlcData(data_dict['master']['i'], data_dict['master']['q'])
            pair.slave = slc.Slc()
            pair.slave.metadata = metadata.fromDim(slv_meta_elem)
            pair.slave.slcdata = iq_float_slcdata.IqFloatSlcData(data_dict[pair.slave.metadata.acquisition_date.strftime("%d%b%Y")]['i'], data_dict[pair.slave.metadata.acquisition_date.strftime("%d%b%Y")]['q'])

            baselines_elem = master_meta_elem.find(".//MDElem[@name='Baselines']")
            base_el = baselines_elem.find(f".//MDElem[@name='Ref_{pair.master.metadata.acquisition_date.strftime("%d%b%Y")}']")
            slv_base_el = base_el.find(f".//MDElem[@name='Secondary_{pair.slave.metadata.acquisition_date.strftime("%d%b%Y")}']")
            pair.perpendicular_baseline = float(slv_base_el.find(".//MDATTR[@name='Perp Baseline']").text)
            pair.temporal_baseline = -int(np.round(float(slv_base_el.find(".//MDATTR[@name='Temp Baseline']").text)))

            result.append(pair)

    return result

def createFromTanDemX(filepath : str, create_rpc = True):
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Initialize variables
    primary_fn = None
    secondary_fn = None
    effectiveBaseline = None
    heightOfAmbiguity = None

    # Find productComponents and process each component
    product_components = root.find("productComponents")
    if product_components is not None:
        for component in product_components.findall("component"):
            if component.get("componentClass") == "imageData":
                name_elem = component.find("name")
                if name_elem is not None:
                    component_type = name_elem.text

                    # Get the file location
                    file_elem = component.find("file")
                    if file_elem is not None:
                        location_elem = file_elem.find("location")
                        if location_elem is not None:
                            name_elem = location_elem.find("name")
                            if name_elem is not None:
                                file_name = name_elem.text

                                if component_type == "cossc_component_primary":
                                    primary_fn = file_name
                                elif component_type == "cossc_component_secondary":
                                    secondary_fn = file_name

    # Process commonAcquisitionInfo
    common_acq_info = root.find("commonAcquisitionInfo")
    if common_acq_info is not None:
        # Check for master ID swap
        insar_master_id = common_acq_info.find("inSARmasterID")
        if insar_master_id is not None and insar_master_id.text == "SAT2":
            primary_fn, secondary_fn = secondary_fn, primary_fn

        # Get acquisition geometry values
        acq_geometry = common_acq_info.find("acquisitionGeometry")
        if acq_geometry is not None:
            eff_baseline_elem = acq_geometry.find("effectiveBaseline")
            if eff_baseline_elem is not None and eff_baseline_elem.text:
                effectiveBaseline = float(eff_baseline_elem.text)

            hoa_elem = acq_geometry.find("heightOfAmbiguity")
            if hoa_elem is not None and hoa_elem.text:
                heightOfAmbiguity = float(hoa_elem.text)

    # Prepare the file paths (similar to the C++ version but simplified)
    root_dir = os.path.dirname(os.path.abspath(filepath))
    primary_path = os.path.join(root_dir, primary_fn + '/' + primary_fn + ".xml") if primary_fn else None
    secondary_path = os.path.join(root_dir, secondary_fn + '/' + secondary_fn + ".xml") if secondary_fn else None

    pair = ResampledPair()
    pair.master = slc.fromTSX(primary_path, swath_id=0, create_rpc=create_rpc)
    pair.slave = slc.fromTSX(secondary_path, swath_id=0, create_rpc=create_rpc)
    pair.perpendicular_baseline = effectiveBaseline
    pair.temporal_baseline = 0
    pair.shift_x = 0
    pair.shift_y = 0
    pair.bistatic = True

    return pair