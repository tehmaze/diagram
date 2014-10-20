from distutils.core import setup
import os


setup(
    name='diagram',
    version='0.1',
    description='Text mode diagrams using UTF-8 characters and fancy colors',
    author='Wijnand Modderman-Lenstra',
    author_email='maze@pyth0n.org',
    py_modules=['diagram'],
    scripts=[os.path.join('bin', 'diagram')],
)
