import unittest

import os
import sys

import qfed.grid


class test(unittest.TestCase):
    def test_basic_functionality(self):
        """
        Test basic functionality of Grid instances.
        """

        for name in ('c', 'e', '0.1x0.1', 'c90', 'c360'):
            grid = qfed.grid.Grid(name)

            print(f'{name = }')
            print(f'{grid.type = }')
            print(f'{grid.dimensions() = }')
            print(f'{grid.lon() = }')
            print(f'{grid.lat() = }')
            print()


if __name__ == '__main__':
    unittest.main()
