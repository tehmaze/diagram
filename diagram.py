#!/usr/bin/env python


"""Text mode diagrams using UTF-8 characters and fancy colors."""

from __future__ import print_function
from __future__ import unicode_literals

__author__ = 'Wijnand Modderman-Lenstra <maze@pyth0n.org>'
__copyright__ = 'Copyright 2014, maze.io labs'
__credits__ = ['Adam Tauber', 'Erik Rose', 'Jeff Quast', 'John-Paul Verkamp']
__license__ = 'MIT'


from collections import defaultdict
from operator import itemgetter
import curses
import locale
import math
import os
import re
import select
import sys
import time
import warnings

# fix for Python2
try:
    range = xrange
    chr = unichr
except NameError:
    pass


# Delimiters for key-value pairs
RE_VALUE_KEY = re.compile(r'[\s=:]+')

# Optionally import numpy for faster arithmetic and curve filtering functions
try:
    import numpy as np
except ImportError:
    np = None

# Setup locale
# Second argument should be a native str (bytes on python2, unicode on
# python3)
if sys.platform == 'darwin':
    locale.setlocale(locale.LC_CTYPE, str('UTF-8'))
else:
    locale.setlocale(locale.LC_ALL, str(''))


class Terminal(object):

    """Terminal manipulation."""

    def __init__(self):
        """Initialize curses."""
        curses.setupterm()

    @property
    def colors(self):
        """Get the number of colors supported by this terminal."""
        number = curses.tigetnum('colors') or 0
        return 16 if number == 8 else number

    @property
    def encoding(self):
        """Get the current terminal encoding."""
        _, encoding = locale.getdefaultlocale()
        return encoding

    @property
    def height(self):
        """Get the current terminal height."""
        return self.size[1]

    @property
    def size(self):
        """Get the current terminal size."""
        for fd in range(3):
            cr = self._ioctl_GWINSZ(fd)
            if cr:
                break
        if not cr:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY)
                cr = self._ioctl_GWINSZ(fd)
                os.close(fd)
            except Exception:
                pass

        if not cr:
            env = os.environ
            cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

        return int(cr[1]), int(cr[0])

    @property
    def width(self):
        """Get the current terminal width."""
        return self.size[0]

    def _ioctl_GWINSZ(self, fd):
        """Get terminal size using ``TIOCGWINSZ``.

        Internal function that will try to request the ``TIOCGWINSZ`` against
        the selected file descriptor ``fd``.
        """
        try:
            import fcntl
            import termios
            import struct
            return struct.unpack(
                'hh',
                fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234')
            )
        except Exception:
            return None

    def color(self, index):
        """Get the escape sequence for indexed color ``index``.

        The ``index`` is a color index in the 256 color space. The color space
        consists of:

        * 0x00-0x0f: default EGA colors
        * 0x10-0xe7: 6x6x6 RGB cubes
        * 0xe8-0xff: gray scale ramp
        """
        if self.colors == 16:
            if index >= 8:
                return self.csi('bold') + self.csi('setaf', index - 8)
            else:
                return self.csi('sgr0') + self.csi('setaf', index)
        else:
            return self.csi('setaf', index)

    def csi(self, capname, *args):
        """Return the escape sequence for the selected Control Sequence."""
        value = curses.tigetstr(capname)
        if value is None:
            return b''
        else:
            return curses.tparm(value, *args)

    def csi_wrap(self, value, capname, *args):
        """Return a value wrapped in the selected CSI and does a reset."""
        if isinstance(value, str):
            value = value.encode('utf-8')
        return b''.join([
            self.csi(capname, *args),
            value,
            self.csi('sgr0'),
        ])


# Bar characters (8 per map)
H_BAR = [(0x258f, 0x258e, 0x258d, 0x258c, 0x258b, 0x258a, 0x2589, 0x2588),
         (0x00a0, 0x2589, 0x258a, 0x258b, 0x258c, 0x258d, 0x258e, 0x258f)]
V_BAR = [(0x2581, 0x2582, 0x2583, 0x2584, 0x2585, 0x2586, 0x2587, 0x2588),
         (0x2588, 0x2587, 0x2586, 0x2585, 0x2584, 0x2583, 0x2582, 0x00a0)]


def filter_symlog(y, base=10.0):
    """Symmetrical logarithmic scale.

    Optional arguments:

    *base*:
        The base of the logarithm.
    """
    log_base = np.log(base)
    sign = np.sign(y)
    logs = np.log(np.abs(y) / log_base)
    return sign * logs


def filter_savitzky_golay(y, window_size=5, order=2, deriv=0, rate=1):
    """Smooth (and optionally differentiate) with a Savitzky-Golay filter."""
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError('window_size and order must be integers')

    if window_size % 2 != 1 or window_size < 1:
        raise ValueError('window_size size must be a positive odd number')
    if window_size < order + 2:
        raise ValueError('window_size is too small for the polynomials order')

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    # precompute limits
    minimum = np.min(y)
    maximum = np.max(y)
    # precompute coefficients
    b = np.mat([
        [k ** i for i in order_range]
        for k in range(-half_window, half_window + 1)
    ])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * math.factorial(deriv)
    # pad the signal at the extremes with values taken from the original signal
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.clip(
        np.convolve(m[::-1], y, mode='valid'),
        minimum,
        maximum,
    )

# Graph filter functions
FUNCTION = dict(
    log=filter_symlog,
    smooth=filter_savitzky_golay,
)
FUNCTION_CONSTANT = {
    'e': math.e,
    '-e': -math.e,
    'pi': math.pi,
    '-pi': -math.pi,
}

# Indexed palettes
PALETTE = dict(
    red={
        0x010: (1, 9),
        0x100: [(16 + (o * 36)) for o in range(1, 6)] + [9],
    },
    green={
        0x010: (2, 10),
        0x100: [(16 + (o * 6)) for o in range(1, 6)] + [10],
    },
    blue={
        0x010: (4, 12),
        0x100: list(range(17, 22)) + [12],
    },
    grey={
        0x010: (8, 7, 15),
        0x100: list(range(232, 257)) + [15],
    },
    spectrum={
        0x010: [14, 6, 2, 10, 11, 3, 9],
        0x100: [
            51 - x for x in range(6)                # blue -> green
        ] + [
            46 + x * 36 for x in range(6)          # green -> yellow
        ] + [
            226 - x * 6 for x in range(6)          # yellow -> red
        ]
    },
)
PALETTE.update(dict(
    gray=PALETTE['grey'],
    default=PALETTE['spectrum'],
))


class Point(object):

    """Holds a single, two-dimensional point."""

    def __init__(self, coordinates):
        """Point with ``(x, y)`` coordinates."""
        self.x = coordinates[0]
        self.y = coordinates[1]

    def __repr__(self):
        return 'Point(%r, %r)' % (self.x, self.y)

    def copy(self):
        """Return a fresh copy of the current point."""
        return Point((self.x, self.y))


class Screen(object):

    """Off-screen render buffer."""

    def __init__(self, size, encoding=None, extend_x=False, extend_y=False):
        if isinstance(size, Point):
            self.size = size.copy()
        else:
            self.size = Point(size)

        self.term = Terminal()
        self.encoding = encoding or self.term.encoding
        self.extend_x = extend_x
        self.extend_y = extend_y

        defaultdictint = lambda: defaultdict(int)
        self.canvas = defaultdict(defaultdictint)

    @property
    def width(self):
        """Get the buffer width."""
        return self.size.x

    @property
    def height(self):
        """Get the buffer height."""
        return self.size.y

    def __contains__(self, point):
        """Check if a point has a value."""
        if not isinstance(point, Point):
            point = Point(point)

        if point.y not in self.canvas:
            return False
        else:
            return point.x in self.canvas[point.y]

    def __repr__(self):
        return '%s(%d,%d)' % (self.__class__.__name__, self.width, self.height)

    def __setitem__(self, point, value):
        """Set a point value."""
        if not isinstance(point, Point):
            point = Point(point)

        if point.y > self.size.y:
            if self.extend_y:
                self.size.y = point.y
            else:
                raise OverflowError('%r overflow y = %d > %d' % (
                    self,
                    point.y,
                    self.size.y,
                ))

        if point.x > self.size.x:
            if self.extend_x:
                self.size.x = point.x
            else:
                raise OverflowError('%r overflow x = %d > %d' % (
                    self,
                    point.x,
                    self.size.x,
                ))

        self.canvas[point.y][point.x] = value

    def __getitem__(self, point):
        """Get a point value or None."""
        if not isinstance(point, Point):
            point = Point(point)
        return self.canvas[point.y][point.x]


class Graph(object):

    """Base class for graphs."""

    def __init__(self, size, option):
        self.size = size
        self.option = option

        # Internal cycle duty counter
        self.cycle = 0
        self.lines = 0
        self.term = Terminal()

        # Override in subclasses
        self.screen = None

        self.minimum = 0
        self.maximum = 0
        self.current = 0

    def consume(self, istream, ostream, batch=False):
        """Read points from istream and output to ostream."""
        datapoints = []  # List of 2-tuples

        if batch:
            sleep = max(0.01, self.option.sleep)
            fd = istream.fileno()
            while True:
                try:
                    if select.select([fd], [], [], sleep):
                        try:
                            line = istream.readline()
                            if line == '':
                                break
                            datapoints.append(self.consume_line(line))
                        except ValueError:
                            continue

                        if self.option.sort_by_column:
                            datapoints = sorted(datapoints, key=itemgetter(self.option.sort_by_column - 1))

                        if len(datapoints) > 1:
                            datapoints = datapoints[-self.maximum_points:]
                            self.update([dp[0] for dp in datapoints], [dp[1] for dp in datapoints])
                            self.render(ostream)

                        time.sleep(sleep)

                except KeyboardInterrupt:
                    break

        else:
            for line in istream:
                try:
                    datapoints.append(self.consume_line(line))
                except ValueError:
                    pass

            if self.option.sort_by_column:
                datapoints = sorted(datapoints, key=itemgetter(self.option.sort_by_column - 1))

            self.update([dp[0] for dp in datapoints], [dp[1] for dp in datapoints])
            self.render(ostream)

    def consume_line(self, line):
        """Consume data from a line."""
        data = RE_VALUE_KEY.split(line.strip(), 1)
        if len(data) == 1:
            return float(data[0]), None
        else:
            return float(data[0]), data[1].strip()

    @property
    def scale(self):
        """Graph scale."""
        return 1

    def update(self, points, values=None):
        """Add a set of data points."""
        self.values = values or [None] * len(points)

        if np is None:
            if self.option.function:
                warnings.warn('numpy not available, function ignored')
            self.points = points
            self.minimum = min(self.points)
            self.maximum = max(self.points)
            self.current = self.points[-1]

        else:
            self.points = self.apply_function(points)
            self.minimum = np.min(self.points)
            self.maximum = np.max(self.points)
            self.current = self.points[-1]

        if self.maximum == self.minimum:
            self.extents = 1
        else:
            self.extents = (self.maximum - self.minimum)
            self.extents = (self.maximum - self.minimum)

    def color_ramp(self, size):
        """Generate a color ramp for the current screen height."""
        color = PALETTE.get(self.option.palette, {})
        color = color.get(self.term.colors, None)
        color_ramp = []
        if color is not None:
            ratio = len(color) / float(size)
            for i in range(int(size)):
                color_ramp.append(self.term.color(color[int(ratio * i)]))

        return color_ramp

    def human(self, size, base=1000, units=' kMGTZ'):
        """Convert the input ``size`` to human readable, short form."""
        sign = '+' if size >= 0 else '-'
        size = abs(size)
        if size < 1000:
            return '%s%d' % (sign, size)
        for i, suffix in enumerate(units):
            unit = 1000 ** (i + 1)
            if size < unit:
                return ('%s%.01f%s' % (
                    sign,
                    size / float(unit) * base,
                    suffix,
                )).strip()
        raise OverflowError

    def apply_function(self, points):
        """Run the filter function on the provided points."""
        if not self.option.function:
            return points

        if np is None:
            raise ImportError('numpy is not available')

        if ':' in self.option.function:
            function, arguments = self.option.function.split(':', 1)
            arguments = arguments.split(',')
        else:
            function = self.option.function
            arguments = []

        # Resolve arguments
        arguments = list(map(self._function_argument, arguments))

        # Resolve function
        filter_function = FUNCTION.get(function)

        if filter_function is None:
            raise TypeError('Invalid function "%s"' % (function,))

        else:
            # We wrap in ``list()`` to consume generators and iterators, as
            # ``np.array`` doesn't do this for us.
            return filter_function(np.array(list(points)), *arguments)

    def _function_argument(self, value):
        """Resolve function, convert to float if not found."""
        if value in FUNCTION_CONSTANT:
            return FUNCTION_CONSTANT[value]
        else:
            return float(value)

    def line(self, p1, p2, resolution=1):
        """Resolve the points to make a line between two points."""
        xdiff = max(p1.x, p2.x) - min(p1.x, p2.x)
        ydiff = max(p1.y, p2.y) - min(p1.y, p2.y)
        xdir = [-1, 1][int(p1.x <= p2.x)]
        ydir = [-1, 1][int(p1.y <= p2.y)]
        r = int(round(max(xdiff, ydiff)))
        if r == 0:
            return

        for i in range((r + 1) * resolution):
            x = p1.x
            y = p1.y

            if xdiff:
                x += (float(i) * xdiff) / r * xdir / resolution
            if ydiff:
                y += (float(i) * ydiff) / r * ydir / resolution

            yield Point((x, y))

    @property
    def maximum_points(self):
        """Override in subclass."""
        raise NotImplementedError()

    def render(self, stream):
        """Render the graph to the selected output stream."""
        raise NotImplementedError()

    def round(self, value):
        """Get an integer value for the input value."""
        return int(value)

    def set_text(self, point, text):
        """Set a text value in the screen canvas."""
        if not self.option.legend:
            return

        if not isinstance(point, Point):
            point = Point(point)

        for offset, char in enumerate(str(text)):
            self.screen.canvas[point.y][point.x + offset] = char


class AxisGraphScreen(Screen):

    """Base class for axial graph buffers.

    The buffer size is larger than the actual screen size, because we have 8
    pixels per character. The number of pixels per character is 2 horizontal
    and 4 vertical pixels.
    """

    @property
    def width(self):
        """Buffer width."""
        return self.size.x * 2

    @property
    def height(self):
        """Buffer height."""
        return self.size.y * 4


class AxisGraph(Graph):

    """Base class for axial graphs."""

    # Graph characters, using braille characters
    base = 0x2800
    pixels = ((0x01, 0x08),
              (0x02, 0x10),
              (0x04, 0x20),
              (0x40, 0x80))

    def __init__(self, size, option):
        super(AxisGraph, self).__init__(size, option)

        self.size = Point((
            size.x or self.term.width,
            size.y or 10
        ))

    def render(self, stream):
        """Render graph to stream."""
        encoding = self.option.encoding or self.term.encoding or "utf8"

        if self.option.color:
            ramp = self.color_ramp(self.size.y)[::-1]
        else:
            ramp = None

        if self.cycle >= 1 and self.lines:
            stream.write(self.term.csi('cuu', self.lines))

        zero = int(self.null / 4)  # Zero crossing
        lines = 0
        for y in range(self.screen.size.y):
            if y == zero and self.size.y > 1:
                stream.write(self.term.csi('smul'))
            if ramp:
                stream.write(ramp[y])

            for x in range(self.screen.size.x):
                point = Point((x, y))
                if point in self.screen:
                    value = self.screen[point]
                    if isinstance(value, int):
                        stream.write(chr(self.base + value).encode(encoding))
                    else:
                        stream.write(self.term.csi('sgr0'))
                        stream.write(self.term.csi_wrap(
                            value.encode(encoding),
                            'bold'
                        ))
                        if y == zero and self.size.y > 1:
                            stream.write(self.term.csi('smul'))
                        if ramp:
                            stream.write(ramp[y])
                else:
                    stream.write(b' ')

            if y == zero and self.size.y > 1:
                stream.write(self.term.csi('rmul'))
            if ramp:
                stream.write(self.term.csi('sgr0'))

            stream.write(b'\n')
            lines += 1
        stream.flush()

        self.cycle = self.cycle + 1
        self.lines = lines

    @property
    def normalised(self):
        """Normalised data points."""
        if np is None:
            return self._normalised_python()
        else:
            return self._normalised_numpy()

    def _normalised_numpy(self):
        """Normalised data points using numpy."""
        dx = (self.screen.width / float(len(self.points)))
        oy = (self.screen.height)
        points = np.array(self.points) - self.minimum
        points = points * 4.0 / self.extents * self.size.y
        for x, y in enumerate(points):
            yield Point((
                dx * x,
                min(oy, oy - y),
            ))

    def _normalised_python(self):
        """Normalised data points using pure Python."""
        dx = (self.screen.width / float(len(self.points)))
        oy = (self.screen.height)
        for x, point in enumerate(self.points):
            y = (point - self.minimum) * 4.0 / self.extents * self.size.y
            yield Point((
                dx * x,
                min(oy, oy - y),
            ))

    @property
    def maximum_points(self):
        """Maximum width."""
        return self.size.x

    @property
    def null(self):
        """Zero crossing value."""
        if not self.option.axis:
            return -1
        else:
            return self.screen.height - (
                -self.minimum * 4.0 / self.extents * self.size.y
            )

    def set(self, point):
        """Set pixel at (x, y) point."""
        if not isinstance(point, Point):
            point = Point(point)

        rx = self.round(point.x)
        ry = self.round(point.y)

        item = Point((rx >> 1, min(ry >> 2, self.size.y)))
        self.screen[item] |= self.pixels[ry & 3][rx & 1]

    def unset(self, point):
        """Unset pixel at (x, y) point."""
        if not isinstance(point, Point):
            point = Point(point)

        x, y = self.round(point.x) >> 1, self.round(point.y) >> 2

        if (x, y) not in self.screen:
            return

        if isinstance(self.screen[y][x], int):
            self.screen[(x, y)] &= ~self.pixels[y & 3][x & 1]

        else:
            del self.screen[(x, y)]

        if not self.screen.canvas.get(y):
            del self.screen[y]

    def update(self, points, values=None):
        super(AxisGraph, self).update(points, values)

        self.screen = AxisGraphScreen(self.size)

        # Plot lines between the points
        prev = Point((0, self.null))
        for curr in self.normalised:
            for point in self.line(prev, curr):
                self.set(point)
            prev = curr

        zero = int(self.null / 4)  # Zero crossing
        if self.size.y > 1:
            self.set_text(Point((0, zero)), '0')
            self.set_text(Point((0, 0)), self.human(self.maximum))
            self.set_text(Point((0, self.size.y - 1)),
                          self.human(self.minimum))
            if self.option.batch:
                current = self.human(self.current)
                self.set_text(Point((self.size.x - len(current), 0)), current)


class BarGraph(Graph):

    """Base class for bar graphs."""

    @property
    def normalised(self):
        for point in self.points:
            yield (point - self.minimum) / self.extents * float(self.scale)


class HorizontalBarGraph(BarGraph):

    """Horizontal bar graph."""

    def __init__(self, size, option):
        super(HorizontalBarGraph, self).__init__(size, option)

        if size.y:
            warnings.warn('Ignoring height on horizontal bar graph')

        self.size = Point((
            size.x or self.term.width,
            1,
        ))

        self.screen = Screen(
            Point((self.size.x, 1)),
            extend_y=True,
        )

        # Select block characters
        if self.option.reverse:
            self.blocks = H_BAR[1]
        else:
            self.blocks = H_BAR[0]

    def bar(self, size, y):
        full, frac = divmod(self.round(size * 8), 8)

        x = 0
        o = self.offset
        if self.option.reverse:
            for x in range(full):
                xr = self.screen.size.x - x - o
                self.screen[(xr, y)] = self.blocks[-1]
            if frac:
                x = x + 1 if x else x
                xr = self.screen.size.x - x - o
                self.screen[(xr, y)] = self.blocks[frac]
        else:
            for x in range(full):
                xr = x + o
                self.screen[(xr, y)] = self.blocks[-1]
            if frac:
                x = x + 1 if x else x
                xr = x + o
                self.screen[(xr, y)] = self.blocks[frac]

        if self.option.keys and self.values[y] is not None:
            value = self.values[y]
            if self.option.reverse:
                point = Point((self.size.x - o, y))
                value = value.ljust(self.offset)
            else:
                point = Point((0, y))
            self.set_text(point, value)

    @property
    def maximum_points(self):
        if self.option.height:
            return self.option.height
        else:
            return 10

    @property
    def offset(self):
        try:
            return max(map(len, (v for v in self.values if v)))
        except (ValueError, TypeError):
            return 0

    @property
    def scale(self):
        size = self.screen.width - self.offset
        if size <= 0:
            raise ValueError('Terminal not wide enough to display data')
        else:
            return size

    def render(self, stream):
        encoding = self.option.encoding or self.term.encoding or "utf8"

        if self.option.color:
            ramp = self.color_ramp(self.scale)
            if self.option.reverse:
                ramp = ramp[::-1]
        else:
            ramp = None

        if self.cycle >= 1:
            stream.write(self.term.csi('cuu', self.lines))

        lines = 0
        stream.write(self.term.csi('el'))
        if self.option.legend:
            offset = self.offset
            minimum_text = self.human(self.minimum)
            maximum_text = self.human(self.maximum)
            minimum_text = minimum_text.ljust(self.scale - len(maximum_text))

            padding_text = ''
            if not self.option.reverse:
                padding_text = ' ' * offset

            stream.write(self.term.csi_wrap(
                ''.join([padding_text, minimum_text, maximum_text]),
                'bold',
            ))
            stream.write(b'\n')
            lines += 1

        for y in range(self.screen.size.y):
            prev_color = ''
            stream.write(self.term.csi('el'))
            for x in range(self.screen.size.x):
                if ramp:
                    try:
                        if self.option.reverse:
                            curr_color = ramp[x]
                        else:
                            curr_color = ramp[x - self.offset]
                    except IndexError:
                        curr_color = self.term.csi('sgr0')
                    if not self.option.reverse and curr_color != prev_color:
                        stream.write(curr_color)
                        prev_color = curr_color

                point = Point((x, y))
                if point in self.screen:
                    value = self.screen[point]
                    if isinstance(value, int):
                        if self.option.reverse:
                            stream.write(curr_color)
                            stream.write(self.term.csi('rev'))
                        stream.write(chr(value).encode(encoding))
                    else:
                        stream.write(self.term.csi('sgr0'))
                        stream.write(self.term.csi_wrap(
                            value.encode(encoding),
                            'bold'
                        ))
                        if ramp:
                            stream.write(curr_color)
                        if self.option.reverse:
                            stream.write(self.term.csi('rev'))
                else:
                    stream.write(b' ')

            if ramp:
                stream.write(self.term.csi('sgr0'))
            stream.write(b'\n')
            lines += 1

        self.cycle = self.cycle + 1
        self.lines = lines

    def update(self, points, values=None):
        super(HorizontalBarGraph, self).update(points, values)

        # Clear screen
        self.screen = Screen(
            self.screen.size.copy(),
            extend_y=True,
        )

        # Plot bars for each line
        for y, size in enumerate(self.normalised):
            self.bar(size, y)


class VerticalBarGraph(BarGraph):

    """Vertical bar graph."""

    def __init__(self, size, option):
        super(VerticalBarGraph, self).__init__(size, option)

        if size.x:
            warnings.warn('Ignoring width on vertical bar graph')

        # Select block characters
        if self.option.reverse:
            self.blocks = V_BAR[1]
        else:
            self.blocks = V_BAR[0]

    def bar(self, size, x):
        full, frac = divmod(self.round(size * 8), 8)

        y = 0
        if self.option.reverse:
            for y in range(full):
                self.screen[(x, y)] = self.blocks[-1]
            if frac:
                y = y + 1 if y else y
                self.screen[(x, y)] = self.blocks[-frac]
        else:
            for y in range(self.size.y, self.size.y - full - 1, -1):
                self.screen[(x, y)] = self.blocks[-1]
            if frac:
                y = self.size.y - full - 1
                self.screen[(x, y)] = self.blocks[frac]

    @property
    def maximum_points(self):
        return self.term.width

    @property
    def scale(self):
        return float(self.screen.height)

    def render(self, stream):
        encoding = self.option.encoding or self.term.encoding or "utf8"

        if self.option.color:
            ramp = self.color_ramp(self.size.y)
            if not self.option.reverse:
                ramp = ramp[::-1]
        else:
            ramp = []

        if self.cycle >= 1:
            stream.write(self.term.csi('cuu', self.lines))

        lines = 0
        for y in range(self.size.y):
            if ramp:
                stream.write(ramp[y])
            for x in range(self.screen.size.x):
                point = Point((x, y))
                if point in self.screen:
                    value = self.screen[point]
                    if isinstance(value, int):
                        if self.option.reverse:
                            if ramp:
                                stream.write(ramp[y])
                            stream.write(self.term.csi_wrap(
                                chr(value),
                                'rev'
                            ))
                        else:
                            stream.write(chr(value).encode(encoding))
                    else:
                        stream.write(self.term.csi('sgr0'))
                        stream.write(self.term.csi_wrap(
                            value.encode(encoding),
                            'bold'
                        ))
                        if ramp:
                            stream.write(ramp[y])

                else:
                    stream.write(b' ')

            if ramp:
                stream.write(self.term.csi('sgr0'))
            stream.write(b'\n')
            lines += 1

        self.cycle = self.cycle + 1
        self.lines = lines

    def update(self, points, values=None):
        if self.option.reverse:
            points = points[::-1]

        maximum_width = self.maximum_points
        if len(points) > maximum_width:
            points = points[-self.maximum_points:]

        # If the legend is enabled, and we have sufficient room to shift the
        # columns to the right, we do so.
        elif len(points) < (maximum_width - 6) and self.option.legend and \
                self.cycle == 0:
            for x in range(6):
                points.insert(0, min(points))

        super(VerticalBarGraph, self).update(points)

        self.size = Point((
            len(self.points),
            self.option.size.y or 10,
        ))

        self.screen = Screen(
            Point((1, self.size.y)),
            extend_x=True,
            extend_y=True,
        )

        # Plot bars for each line
        for x, size in enumerate(self.normalised):
            self.bar(size, x)

        # Plot legend, if there is sufficient space
        if self.size.y > 1 and self.option.legend:
            self.set_text(Point((0, 0)), self.human(self.maximum))
            self.set_text(Point((0, self.size.y - 1)),
                          self.human(self.minimum))
            if self.option.batch:
                current = self.human(self.current)
                self.set_text(Point((self.size.x - len(current), 0)), current)


def usage_function(parser):
    """Show usage and available curve functions."""
    parser.print_usage()
    print('')
    print('available functions:')
    for function in sorted(FUNCTION):
        doc = FUNCTION[function].__doc__.strip().splitlines()[0]
        print('    %-12s %s' % (function + ':', doc))

    return 0


def usage_palette(parser):
    """Show usage and available palettes."""
    parser.print_usage()
    print('')
    print('available palettes:')
    for palette in sorted(PALETTE):
        print('    %-12s' % (palette,))

    return 0


def run():
    """Main entrypoint if invoked via the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Text mode diagrams using UTF-8 characters and fancy colors.'
        ),
        epilog="""
    (1): only works for the horizontal bar graph, the first argument is the key
    and the second value is the data point.
""",
    )

    group = parser.add_argument_group('optional drawing mode')
    group.add_argument(
        '-G', '--graph',
        dest='mode', action='store_const', const='g',
        help='axis drawing mode (default)',
    )
    group.add_argument(
        '-H', '--horizontal-bars',
        dest='mode', action='store_const', const='h',
        help='horizontal drawing mode',
    )
    group.add_argument(
        '-V', '--vertical-bars',
        dest='mode', action='store_const', const='v',
        help='vertical drawing mode',
    )

    group = parser.add_argument_group('optional drawing arguments')
    group.add_argument(
        '-a', '--axis',
        dest='axis', action='store_const', const=True, default=True,
        help='draw axis (default: yes)',
    )
    group.add_argument(
        '-A', '--no-axis',
        dest='axis', action='store_const', const=False,
        help="don't draw axis",
    )
    group.add_argument(
        '-c', '--color',
        dest='color', action='store_const', const=True, default=True,
        help='use colors (default: yes)',
    )
    group.add_argument(
        '-C', '--no-color',
        dest='color', action='store_const', const=False,
        help="don't use colors",
    )
    group.add_argument(
        '-l', '--legend',
        dest='legend', action='store_const', const=True, default=True,
        help='draw y-axis legend (default: yes)',
    )
    group.add_argument(
        '-L', '--no-legend',
        dest='legend', action='store_const', const=False,
        help="don't draw y-axis legend",
    )
    group.add_argument(
        '-f', '--function',
        default=None, metavar='function',
        help='curve manipulation function, use "help" for a list',
    )
    group.add_argument(
        '-p', '--palette',
        default='default', metavar='palette',
        help='palette name, use "help" for a list',
    )
    group.add_argument(
        '-x', '--width',
        default=0, type=int, metavar='characters',
        help='drawing width (default: auto)',
    )
    group.add_argument(
        '-y', '--height',
        default=0, type=int, metavar='characters',
        help='drawing height (default: auto)',
    )
    group.add_argument(
        '-r', '--reverse',
        default=False, action='store_true',
        help='reverse draw graph',
    )
    group.add_argument(
        '--sort-by-column',
        default=0, type=int, metavar='index',
        help='sort input data based on given column',
    )

    group = parser.add_argument_group('optional input and output arguments')
    group.add_argument(
        '-b', '--batch',
        default=False, action='store_true',
        help='batch mode (default: no)',
    )
    group.add_argument(
        '-k', '--keys',
        default=False, action='store_true',
        help='input are key-value pairs (default: no) (1)',
    )
    group.add_argument(
        '-s', '--sleep',
        default=0, type=float,
        help='batch poll sleep time (default: none)',
    )
    group.add_argument(
        '-i', '--input',
        default='-', metavar='file',
        help='input file (default: stdin)',
    )
    group.add_argument(
        '-o', '--output',
        default='-', metavar='file',
        help='output file (default: stdout)',
    )
    group.add_argument(
        '-e', '--encoding',
        dest='encoding', default='',
        help='output encoding (default: auto)',
    )

    option = parser.parse_args()

    if option.function == 'help':
        return usage_function(parser)

    if option.palette == 'help':
        return usage_palette(parser)

    option.mode = option.mode or 'g'
    option.size = Point((option.width, option.height))

    if option.input in ['-', 'stdin']:
        istream = sys.stdin
    else:
        istream = open(option.input, 'r')

    if option.output in ['-', 'stdout']:
        try:
            ostream = sys.stdout.buffer
        except AttributeError:
            ostream = sys.stdout
    else:
        ostream = open(option.output, 'wb')

    option.encoding = option.encoding or Terminal().encoding

    if option.mode == 'g':
        engine = AxisGraph(option.size, option)

    elif option.mode == 'h':
        engine = HorizontalBarGraph(option.size, option)

    elif option.mode == 'v':
        engine = VerticalBarGraph(option.size, option)

    else:
        parser.error('invalid mode')
        return 1

    engine.consume(istream, ostream, batch=option.batch)


class DOption(object):
    """ Placeholder class for diagram options.
    """

    def __init__(self):
        self.width = 0
        self.height = 0
        self.reverse = False
        self.function = None
        self.batch = False
        self.legend = True
        self.encoding = ''
        self.color = True
        self.palette = 'default'
        self.size = Point([self.width, self.height])
        self.mode = 'v'
        self.axis = True
        self.keys = False

        # Make graph look 'ok' instead of crashing on MS windows
        import platform
        if 'Windows' in platform.platform():
            self.encoding = 'utf-8'


class DGWrapper(object):
    """ Wrapper around a bar graph from the awesome diagram pacakge.
    """

    def __init__(self, dg_option=None, ostream=None, data=None):
        """ Handle some of the setup functions for the graph in the
        diagram package. Specifically hide all of the requirements that
        are computed in run() inside diagram.py.
        """

        # Create a pre-populated object similar to the results of
        # argparse
        self.dg_option = dg_option
        if self.dg_option == None:
            self.dg_option = DOption()

        # handle buffered and non buffered input gracefully.
        self.ostream = ostream
        if self.ostream == None:
            try:
                self.ostream = sys.stdout.buffer
            except AttributeError:
                self.ostream = sys.stdout


        if self.dg_option.mode == 'h':
            self.dg = HorizontalBarGraph(self.dg_option.size,
                                       self.dg_option)

        elif self.dg_option.mode == 'v':
            self.dg = VerticalBarGraph(self.dg_option.size,
                                       self.dg_option)

        else:
            self.dg = AxisGraph(self.dg_option.size,
                            self.dg_option)

        # off-screen render the graph with points, values
        self.dg.update(data[0], data[1])

    def show(self):
        """ Actually show the graph on screen.
        """
        self.dg.render(self.ostream)


if __name__ == '__main__':
    sys.exit(run())

# pylint: disable=C0102
