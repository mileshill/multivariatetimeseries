import unittest
from DataTransforms import DataTransforms


class DataTransformsTestCase(unittest.TestCase):

    def test_is_valid_file(self):
        valid_file_path = 'valid.csv'
        DataTransforms(file_path=valid_file_path)
        self.assertTrue(True)

    def test_is_invalid_file(self):
        invalid_file_path = 'invalid.txt'
        with self.assertRaises(AssertionError):
            DataTransforms(file_path=invalid_file_path)


if __name__ == '__main__':
    unittest.main()
