#!/usr/bin/env python

from setuptools import setup
import os


LONG_DESCRIPTION = '''
Text mode diagrams using UTF-8 characters and fancy colors (using Python).

Features
 * Axial graphs
 * Horizontal and vertical bar graphs
 * Supports both 3 bit (16 color) and 8 bit (256 color) mode colors
 * Supports various pre-defined palettes
 * UTF-8 text graphics
'''.strip()


setup(
    name='diagram',
    version='0.2.26',
    description='Text mode diagrams using UTF-8 characters and fancy colors',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/tehmaze/diagram',
    author='Wijnand Modderman-Lenstra',
    author_email='maze@pyth0n.org',
    keywords=['diagram', 'graph', 'ascii', 'ansi', 'text'],
    py_modules=['diagram'],
    scripts=[os.path.join('bin', 'diagram')],
)
