# standard libraries
import logging
import unittest

# third party libraries
import numpy

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata


class TestExtendedData(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_rgb_data_constructs_with_default_calibrations(self):
        data = numpy.zeros((8, 8, 3), dtype=numpy.uint8)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        self.assertEqual(len(xdata.dimensional_shape), len(xdata.dimensional_calibrations))

    def test_rgb_data_slice_works_correctly(self):
        data = numpy.zeros((8, 8, 3), dtype=numpy.uint8)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        self.assertTrue(xdata.is_data_rgb_type)
        xdata_slice = xdata[2:6, 2:6]
        self.assertTrue(xdata_slice.is_data_rgb_type)
        self.assertTrue(xdata_slice.dimensional_shape, (4, 4))

    def test_data_slice_calibrates_correctly(self):
        data = numpy.zeros((100, 100), dtype=numpy.float)
        xdata = DataAndMetadata.new_data_and_metadata(data)
        calibrations = xdata[40:60, 40:60].dimensional_calibrations
        self.assertAlmostEqual(calibrations[0].offset, 40)
        self.assertAlmostEqual(calibrations[0].scale, 1)
        self.assertAlmostEqual(calibrations[1].offset, 40)
        self.assertAlmostEqual(calibrations[1].scale, 1)

    def test_data_slice_of_sequence_handles_calibrations(self):
        data = numpy.zeros((10, 100, 100), dtype=numpy.float)
        intensity_calibration = Calibration.Calibration(0.1, 0.2, "I")
        dimensional_calibrations = [Calibration.Calibration(0.11, 0.22, "S"), Calibration.Calibration(0.11, 0.22, "A"), Calibration.Calibration(0.111, 0.222, "B")]
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))
        self.assertFalse(xdata[3].is_sequence)
        self.assertTrue(xdata[3:4].is_sequence)
        self.assertAlmostEqual(xdata[3].intensity_calibration.offset, xdata.intensity_calibration.offset)
        self.assertAlmostEqual(xdata[3].intensity_calibration.scale, xdata.intensity_calibration.scale)
        self.assertEqual(xdata[3].intensity_calibration.units, xdata.intensity_calibration.units)
        self.assertAlmostEqual(xdata[3].dimensional_calibrations[0].offset, xdata.dimensional_calibrations[1].offset)
        self.assertAlmostEqual(xdata[3].dimensional_calibrations[0].scale, xdata.dimensional_calibrations[1].scale)
        self.assertEqual(xdata[3].dimensional_calibrations[0].units, xdata.dimensional_calibrations[1].units)
        self.assertAlmostEqual(xdata[3].dimensional_calibrations[1].offset, xdata.dimensional_calibrations[2].offset)
        self.assertAlmostEqual(xdata[3].dimensional_calibrations[1].scale, xdata.dimensional_calibrations[2].scale)
        self.assertEqual(xdata[3].dimensional_calibrations[1].units, xdata.dimensional_calibrations[2].units)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
