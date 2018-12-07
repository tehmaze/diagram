#!/usr/bin/env python

from setuptools import setup
import io
import os


def read(name):
    return open(os.path.join(os.path.dirname(__file__), name)).read()


setup(
    name='diagram',
    version='0.2.27',
    description='Text mode diagrams using UTF-8 characters and fancy colors',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/tehmaze/diagram',
    author='Wijnand Modderman-Lenstra',
    author_email='maze@pyth0n.org',
    keywords=['diagram', 'graph', 'ascii', 'ansi', 'text'],
    py_modules=['diagram'],
    scripts=[os.path.join('bin', 'diagram')],
)
