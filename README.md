# diagram

Text mode diagrams using UTF-8 characters and fancy colors (using Python).

## Features

 * Axial graphs
 * Horizontal and vertical bar graphs
 * Supports both 3 bit (16 color) and 8 bit (256 color) mode colors with
   various pre-defined palettes (see below)
 * UTF-8 text graphics

## Installation

It's recommended to use pip to install/update.

To install:

    $ sudo pip install diagram

To update:

    $ sudo pip install -U diagram

## Examples

Pictures say more than a thousand words.

### Axis graph

![Axis Graph](doc/axisgraph.png)

### Horizontal bar graph

![Horizontal bar graph](doc/horizontalbar.png)

Drawing characters used:

    ▏ ▎ ▍ ▌ ▋ ▊ ▉ █

### Vertical bar graph

![Vertical bar graph](doc/verticalbar.png)

Drawing characters used:

    ▁ ▂ ▃ ▄ ▅ ▆ ▇ █


## Usage

Use `diagram --help` for documentation:

    usage: diagram [-h] [-G] [-H] [-V] [-a] [-A] [-c] [-C] [-p PALETTE]
                   [-x characters] [-y characters] [-r] [-i file] [-o file]
                   [-e ENCODING]

    optional arguments:
      -h, --help            show this help message and exit

    optional drawing mode:
      -G, --graph           axis drawing mode (default)
      -H, --horizontal-bars
                            horizontal drawing mode
      -V, --vertical-bars   vertical drawing mode

    optional drawing arguments:
      -a, --axis            draw axis (default: yes)
      -A, --no-axis         don't draw axis
      -c, --color           use colors (default: yes)
      -C, --no-color        don't use colors
      -p palette, --palette palette
                            palette name, use "help" for a list
      -x characters, --width characters
                            drawing width (default: auto)
      -y characters, --height characters
                            drawing height (default: auto)
      -r, --reverse         reverse draw graph

    optional input and output arguments:
      -i file, --input file
                            input file (default: stdin)
      -o file, --output file
                            output file (default: stdout)
      -e ENCODING, --encoding ENCODING
                            output encoding (default: auto)

# `--palette ...`

## default / spectrum

![Palette Spectrum](doc/palette-spectrum.png)

## grey

![Palette Grey](doc/palette-grey.png)

## red

![Palette Red](doc/palette-red.png)

## green

![Palette Green](doc/palette-green.png)

## blue

![Palette Blue](doc/palette-blue.png)
