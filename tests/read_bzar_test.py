import unittest

import pysar
from pysar import slc, coordinates
import datetime
import numpy as np
import pathlib

class ImportBzarTest(unittest.TestCase):

    def test_import_Bzar(self):
        filename = '../data/TSX-1_0_2023-04-25.bzar.slc.xml'
        slc_data = slc.fromBzarXml(filename)

        self.assertEqual(17392, slc_data.metadata.number_rows)
        self.assertEqual(datetime.date(2023, 4, 25), slc_data.metadata.acquisition_date)
        geoc = np.array(coordinates.geodetic_to_geocentric(lat=30.46797876432665, lon=114.53029015184815))
        self.assertAlmostEqual(-2284344.436283692, geoc[0], places=5)
        self.assertAlmostEqual(5005522.103689523, geoc[1], places=5)
        self.assertAlmostEqual(3215195.349620832, geoc[2], places=5)
        self.assertAlmostEqual(2.3584905660377357e-05, slc_data.metadata.burst.row_spacing, places=10)

        img = slc_data.metadata.burst.pixel_from_geocentric(geoc)
        self.assertAlmostEqual(3358, img[0], delta=0.01)
        self.assertAlmostEqual(8696, img[1], delta=0.4)

        self.assertEqual('VV', slc_data.metadata.polarization)
        self.assertEqual('descending', slc_data.metadata.orbit_direction)

if __name__ == '__main__':
    unittest.main()
