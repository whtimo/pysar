import unittest

import pysar
from pysar import slc, coordinates
import datetime
import numpy as np
import pathlib

class ImportTsxTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.import_path_ = pathlib.Path('../data/import')
        for child in cls.import_path_.iterdir():
            if child.is_file():
                child.unlink()
    def test_importTsx_RapaNui(self):
        filename = '../data/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450.xml'
        out_filepath = '../data'

        tsx = pysar.slc.fromTSX(filename, 0)
        out_filename = pysar.slc.getPysarPathName(tsx, out_filepath, True)
        pysar.slc.saveToPysarXml(tsx, out_filename, None)

        slc = pysar.slc.fromPysarXml(out_filename)
        self.assertEqual('HH', slc.metadata.polarization)
        pos = slc.metadata._orbit.positions[0]
        vel = slc.metadata._orbit.velocities[1]
        self.assertEqual(-2.14701108317305800E+06, pos[0])
        self.assertEqual(-5.54098863828714285E+06, pos[1])
        self.assertEqual(-3.49369747675457923E+06, pos[2])
        self.assertEqual(-2.82165400000000000E+03, vel[0])
        self.assertEqual(-2.90622100000000000E+03, vel[1])
        self.assertEqual(6.52336300000000028E+03, vel[2])

        int_pos = slc.metadata._orbit.interpolate_position(65.0)
        self.assertAlmostEqual(-2324366.2779588, int_pos[0],
                               places=2)  # Compare to Lagrance. Cubic Hermite should be better
        self.assertAlmostEqual(-5716825.06516074, int_pos[1], places=2)
        self.assertAlmostEqual(-3070524.12834013, int_pos[2], places=2)

        self.assertEqual('TDX-1', slc.metadata.sensor)

        self.assertEqual(2023, slc.metadata.acquisition_date.year)
        self.assertEqual(10, slc.metadata.acquisition_date.month)
        self.assertEqual(5, slc.metadata.acquisition_date.day)

        self.assertEqual(2023, slc.metadata._orbit.reference_time.year)
        self.assertEqual(10, slc.metadata._burst.orbit.reference_time.month)
        self.assertEqual(5, slc.metadata._burst.orbit.reference_time.day)
        self.assertEqual(1, slc.metadata._orbit.reference_time.hour)
        self.assertEqual(44, slc.metadata._burst.orbit.reference_time.minute)
        self.assertEqual(1, slc.metadata._burst.orbit.reference_time.second)

        self.assertEqual(0.0019498737784892511, slc.metadata._burst.range_time_to_first_pixel)
        self.assertEqual(49.243694, slc.metadata._burst.first_azimuth_time)

        self.assertEqual(3.033443670357346e-09 / 2.0, slc.metadata._burst.column_spacing)
        self.assertEqual(2.364066193853428e-05, slc.metadata._burst.row_spacing)

        self.assertEqual(7322, slc.metadata.number_columns)
        self.assertEqual(17230, slc.metadata._burst.number_rows)


        center_lat = -2.70843292193757996E+01
        center_lon = -1.09317180639935202E+02
        geocentric = np.array(coordinates.geodetic_to_geocentric(center_lat, center_lon))
        center_pix = slc.metadata.pixel_from_geocentric(geocentric)
        self.assertAlmostEqual(7322 / 2.0, center_pix[0], delta=30)
        self.assertAlmostEqual(17230 / 2.0, center_pix[1], delta=0.2)

        self.assertEqual(-2.71032592707287279E+01, slc.metadata.footprint.lower_left[0])
        self.assertEqual(-1.09344243519846742E+02, slc.metadata.footprint.lower_left[1])
        self.assertEqual(-2.70902702982963852E+01, slc.metadata.footprint.lower_right[0])
        self.assertEqual(-1.09280669066291807E+02, slc.metadata.footprint.lower_right[1])
        self.assertEqual(-2.70647479804389377E+01, slc.metadata.footprint.upper_right[0])
        self.assertEqual(-1.09286999959474656E+02, slc.metadata.footprint.upper_right[1])
        self.assertEqual(-2.70776646487785442E+01, slc.metadata.footprint.upper_left[0])
        self.assertEqual(-1.09350220815192586E+02, slc.metadata.footprint.upper_left[1])

        self.assertEqual(-1.09350220815192586E+02, slc.metadata.footprint.left())
        self.assertEqual(-1.09280669066291807E+02, slc.metadata.footprint.right())
        self.assertEqual(-2.70647479804389377E+01, slc.metadata.footprint.top())
        self.assertEqual(-2.71032592707287279E+01, slc.metadata.footprint.bottom())

    def test_importTsx(self):
        path = '../data/TSX1_SAR__SSC______ST_S_SRA_20230425T222958_20230425T222958/TSX1_SAR__SSC______ST_S_SRA_20230425T222958_20230425T222958.xml'
        out_filepath = '../data'
        tsx = pysar.slc.fromTSX(path, 0)
        out_filename = pysar.slc.getPysarPathName(tsx, out_filepath, True)
        pysar.slc.saveToPysarXml(tsx, out_filename, None)

        slc = pysar.slc.fromPysarXml(out_filename)
        self.assertEqual(17392, slc.metadata.number_rows)
        self.assertEqual(datetime.date(2023, 4, 25), slc.metadata.acquisition_date)
        geoc = np.array(coordinates.geodetic_to_geocentric(lat=30.46797876432665, lon=114.53029015184815))
        self.assertAlmostEqual(-2284344.436283692, geoc[0], places=5)
        self.assertAlmostEqual(5005522.103689523, geoc[1], places=5)
        self.assertAlmostEqual(3215195.349620832, geoc[2], places=5)
        self.assertEqual(2.3584905660377357e-05, slc.metadata._burst.row_spacing)

        img = slc.metadata.pixel_from_geocentric(geoc)
        self.assertAlmostEqual(3358, img[0], delta=0.01)
        self.assertAlmostEqual(8696, img[1], delta=0.4)

        self.assertAlmostEqual(0.0310665821396723, slc.metadata.wavelength, places=10)
        self.assertEqual('TSX-1', slc.metadata.sensor)
        self.assertAlmostEqual(29.136, slc.metadata.get_incidence_angle(3358), places=3)


if __name__ == '__main__':
    unittest.main()
