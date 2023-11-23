"""Unittest for directory.
"""
import os
import path
import unittest


class TestDirectory(unittest.TestCase):

    def test_directory_creation(self):
        directory_name = path.create_timestamped_directory(".")
        self.assertTrue(os.path.exists(directory_name))
        os.rmdir(directory_name)


if __name__ == "__main__":
    unittest.main()
