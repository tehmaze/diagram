from distutils.core import setup
import os


setup(
    name='diagram',
    version='0.1',
    description='Draw UTF-8 diagrams on the console',
    author='Wijnand Modderman-Lenstra',
    author_email='maze@pyth0n.org',
    py_modules=['diagram'],
    scripts=[os.path.join('bin', 'diagram')],
)
