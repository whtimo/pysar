import unittest

import pysar
from pysar import slc
import datetime

class ReadTsxTest(unittest.TestCase):

    def test_readTsx(self):
        filename = '../data/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450.xml'
        slc = pysar.slc.fromTSX(filename, 0)
        self.assertEqual('HH', slc.polarization)
        self.assertEqual(7322, slc.slcdata.getWidth())
        self.assertEqual(17230, slc.slcdata.getHeight() )
        pos = slc.metadata.burst.orbit.positions[0]
        vel = slc.metadata.burst.orbit.velocities[1]
        self.assertEqual(-2.14701108317305800E+06, pos[0])
        self.assertEqual(-5.54098863828714285E+06, pos[1])
        self.assertEqual(-3.49369747675457923E+06, pos[2])
        self.assertEqual(-2.82165400000000000E+03, vel[0])
        self.assertEqual(-2.90622100000000000E+03, vel[1])
        self.assertEqual(6.52336300000000028E+03, vel[2])


if __name__ == '__main__':
    unittest.main()