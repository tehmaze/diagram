from collections import defaultdict
import curses
import locale
import os
import sys
import unicodedata
import warnings


# Setup locale
if sys.platform == 'darwin':
    locale.setlocale(locale.LC_CTYPE, 'UTF-8')
else:
    locale.setlocale(locale.LC_ALL, '')


class Terminal(object):
    '''
    Terminal manipulation.
    '''

    def __init__(self):
        curses.setupterm()

    @property
    def colors(self):
        number = curses.tigetnum('colors') or 0
        return 16 if number == 8 else number

    @property
    def encoding(self):
        lang, encoding = locale.getdefaultlocale()
        return encoding

    @property
    def height(self):
        return self.size[1]

    @property
    def size(self):
        env = os.environ
        def ioctl_GWINSZ(fd):
            try:
                import fcntl, termios, struct, os
                cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
            except:
                return
            else:
                return cr

        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        if not cr:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY)
                cr = ioctl_GWINSZ(fd)
                os.close(fd)
            except:
                pass

        if not cr:
            cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

        return int(cr[1]), int(cr[0])

    @property
    def width(self):
        return self.size[0]

    def color(self, index):
        if self.colors == 16:
            if index >= 8:
                return self.csi('bold') + self.csi('setaf', index - 8)
            else:
                return self.csi('sgr0') + self.csi('setaf', index)
        else:
            return self.csi('setaf', index)

    def csi(self, capname, *args):
        value = curses.tigetstr(capname)
        if value is None:
            return ''
        else:
            return curses.tparm(value, *args)

    def csi_wrap(self, value, capname, *args):
        return u''.join([
            self.csi(capname, *args),
            value,
            self.csi('sgr0'),
        ])


# Bar characters (8 per map)
H_BAR = [(0x258f, 0x258e, 0x258d, 0x258c, 0x258b, 0x258a, 0x2589, 0x2588),
         (0x00a0, 0x2589, 0x258a, 0x258b, 0x258c, 0x258d, 0x258e, 0x258f)]
V_BAR = [(0x2581, 0x2582, 0x2583, 0x2584, 0x2585, 0x2586, 0x2587, 0x2588),
         (0x2588, 0x2587, 0x2586, 0x2585, 0x2584, 0x2583, 0x2582, 0x00a0)]

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
        0x100: range(17, 22) + [12],
    },
    grey={
        0x010: (8, 7, 15),
        0x100: range(232, 257) + [15],
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
    gray    = PALETTE['grey'],
    default = PALETTE['spectrum'],
))


class Point(object):
    '''
    Holds a single, two-dimensional point.
    '''

    def __init__(self, coordinates):
        self.x = coordinates[0]
        self.y = coordinates[1]

    def __repr__(self):
        return 'Point(%r, %r)' % (self.x, self.y)

    def copy(self):
        return Point((self.x, self.y))


class Screen(object):
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
        return self.size.x

    @property
    def height(self):
        return self.size.y

    def __contains__(self, point):
        if not isinstance(point, Point):
            point = Point(point)

        if not point.y in self.canvas:
            return False
        else:
            return point.x in self.canvas[point.y]

    def __iter__(self):
        for y in range(self.size.y):
            yield self.canvas[y]

    def __repr__(self):
        return '%s(%d, %d)' % (self.__class__.__name__, self.width, self.height)

    def __setitem__(self, point, value):
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
        if not isinstance(point, Point):
            point = Point(point)
        return self.canvas[point.y][point.x]


class Graph(object):
    def __init__(self, size, points, option):
        self.size = size
        self.points = points
        self.option = option

        self.term = Terminal()

        self.minimum = min(self.points)
        self.maximum = max(self.points)
        self.extents = (self.maximum - self.minimum)

    def color_ramp(self, size):
        '''
        Generate a color ramp for the current screen height.
        '''
        color = PALETTE.get(self.option.palette, {}).get(self.term.colors, None)
        color_ramp = []
        if color is not None:
            ratio = len(color) / float(size)
            for i in range(int(size)):
                color_ramp.append(self.term.color(color[int(ratio * i)]))

        return color_ramp

    def human(self, size, base=1000, units=' kMGTZ'):
        sign = '+' if size >= 0 else '-'
        size = abs(size)
        if size < 1000:
            return ('%s%d' % (sign, size))
        for i, suffix in enumerate(units):
            unit = 1000 ** (i + 1)
            if size < unit:
                return ('%s%.01f%s' % (
                    sign,
                    size / float(unit) * base,
                    suffix,
                )).strip()
        raise OverflowError

    def line(self, p1, p2, resolution=1):
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

    def render(self, stream):
        raise NotImplementedError()

    def round(self, value):
        if isinstance(value, (int, long)):
            return value

        else:
            return long(round(value))

    def set_text(self, point, text):
        if not self.option.axis:
            return

        if not isinstance(point, Point):
            point = Point(point)

        for offset, char in enumerate(str(text)):
            self.screen.canvas[point.y][point.x + offset] = char


class AxisGraphScreen(Screen):
    @property
    def width(self):
        return self.size.x * 2

    @property
    def height(self):
        return self.size.y * 4


class AxisGraph(Graph):
    # Graph characters, using braille characters
    base = 0x2800
    pixels = ((0x01, 0x08),
              (0x02, 0x10),
              (0x04, 0x20),
              (0x40, 0x80))

    def __init__(self, size, points, option):
        super(AxisGraph, self).__init__(size, points, option)

        self.size = Point((
            size.x or self.term.width,
            size.y or 10
        ))
        self.screen = AxisGraphScreen(self.size)

        # Plot lines between the points
        prev = Point((0, self.null))
        for curr in self.normalised:
            for point in self.line(prev, curr):
                self.set(point)

            prev = curr

    def render(self, stream):
        encoding = self.option.encoding or self.term.encoding

        zero = int(self.null / 4)  # Zero crossing
        if self.size.y > 1:
            self.set_text(Point((0, zero)), '0')
            self.set_text(Point((0, 0)), self.human(self.maximum))
            self.set_text(Point((0, self.size.y - 1)), self.human(self.minimum))

        prev_color = ''
        if self.option.color:
            ramp = self.color_ramp(self.size.y)[::-1]
        else:
            ramp = None

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
                        stream.write(unichr(self.base + value).encode(encoding))
                    else:
                        stream.write(self.term.csi('sgr0'))
                        stream.write(self.term.csi('bold'))
                        stream.write(unicode(value).encode(encoding))
                        # There's no capability to just turn off bold, meh.
                        stream.write(self.term.csi('sgr0'))
                        if y == zero and self.size.y > 1:
                            stream.write(self.term.csi('smul'))
                        if ramp:
                            stream.write(ramp[y])
                else:
                    stream.write(' ')

            if y == zero and self.size.y > 1:
                stream.write(self.term.csi('rmul'))
            if ramp:
                stream.write(self.term.csi('sgr0'))

            stream.write('\n')


    @property
    def normalised(self):
        dx = (self.screen.width / float(len(self.points)))
        oy = (self.screen.height)
        for x, point in enumerate(self.points):
            y = (point - self.minimum) * 4.0 / self.extents * self.size.y
            yield Point((
                dx * x,
                oy - y,
            ))

    @property
    def null(self):
        if not self.option.axis:
            return -1
        else:
            return self.screen.height - (-self.minimum * 4.0 / self.extents * self.size.y)

    def set(self, point):
        if not isinstance(point, Point):
            point = Point(point)

        rx = self.round(point.x)
        ry = self.round(point.y)

        item = Point((rx >> 1, min(ry >> 2, self.size.y)))
        self.screen[item] |= self.pixels[ry & 3][rx & 1]

    def unset(self, point):
        if not isinstance(point, Point):
            point = Point(point)

        x, y = self.round(point.x) >> 1, self.round(point.y) >> 2

        if (x, y) not in self.screen:
            return

        if isinstance(self.screen[y][x], (int, long)):
            self.screen[(x, y)] &= ~self.pixels[y & 3][x & 1]

        else:
            del self.screen[(x, y)]

        if not self.canvas.get(y):
            del self.screen[y]


class BarGraph(Graph):
    @property
    def normalised(self):
        for point in self.points:
            yield (point - self.minimum) / self.extents * self.scale


class HorizontalBarGraph(BarGraph):
    def __init__(self, size, points, option):
        super(BarGraph, self).__init__(size, points, option)

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
            self.blocks = map(
                lambda b: self.term.csi_wrap(unichr(b), 'rev'),
                H_BAR[1],
            )
        else:
            self.blocks = map(unichr, H_BAR[0])

        # Plot bars for each line
        for y, size in enumerate(self.normalised):
            self.bar(size, y)

    def bar(self, size, y):
        full, frac = divmod(self.round(size * 8), 8)

        x = 0
        if self.option.reverse:
            for x in range(full):
                xr = self.screen.size.x - x
                self.screen[(xr, y)] = self.blocks[-1]
            if frac:
                x = x + 1 if x else x
                xr = self.screen.size.x - x
                self.screen[(xr, y)] = self.blocks[frac]
        else:
            for x in range(full):
                self.screen[(x, y)] = self.blocks[-1]
            if frac:
                x = x + 1 if x else x
                self.screen[(x, y)] = self.blocks[frac]

    @property
    def scale(self):
        return float(self.screen.width)

    def render(self, stream):
        encoding = self.option.encoding or term_encoding()

        if self.option.color:
            ramp = self.color_ramp(self.screen.size.x)
            if self.option.reverse:
                ramp = ramp[::-1]
        else:
            ramp = None

        for y in range(self.screen.size.y):
            prev_color = ''
            for x in range(self.screen.size.x):
                if ramp:
                    curr_color = ramp[x]
                    if self.option.reverse:
                        stream.write(curr_color)
                    elif curr_color != prev_color:
                        stream.write(curr_color)
                        prev_color = curr_color

                point = Point((x, y))
                if point in self.screen:
                    value = self.screen[point]
                    if isinstance(value, int):
                        stream.write(unichr(value).encode(encoding))
                    else:
                        stream.write(unicode(value).encode(encoding))

                else:
                    stream.write(' ')

            if ramp:
                stream.write(self.term.csi('sgr0'))
            stream.write('\n')


class VerticalBarGraph(BarGraph):
    def __init__(self, size, points, option):
        super(BarGraph, self).__init__(size, points, option)

        if size.x:
            warnings.warn('Ignoring width on horizontal bar graph')

        maximum_width = self.term.width
        if len(points) > maximum_width:
            self.points = points[-maximum_width:]

        self.size = Point((
            len(self.points),
            size.y or 10,
        ))

        self.screen = Screen(
            Point((1, self.size.y)),
            extend_x=True,
            extend_y=True,
        )

        # Select block characters
        if self.option.reverse:
            self.blocks = map(
                lambda b: self.term.csi_wrap(unichr(b), 'rev'),
                V_BAR[1],
            )
        else:
            self.blocks = map(unichr, V_BAR[0])

        # Plot bars for each line
        for x, size in enumerate(self.normalised):
            self.bar(size, x)

    def bar(self, size, x):
        full, frac = divmod(self.round(size * 8), 8)

        y = 0
        if self.option.reverse:
            for y in range(full):
                self.screen[(x, y)] = self.blocks[-1]
            if frac:
                y = y + 1 if y else y
                self.screen[(x, y)] = self.blocks[frac]
        else:
            for y in range(self.size.y, self.size.y - full - 1, -1):
                self.screen[(x, y)] = self.blocks[-1]
            if frac:
                y = self.size.y - full - 1
                self.screen[(x, y)] = self.blocks[frac]

    @property
    def scale(self):
        return float(self.screen.height)

    def render(self, stream):
        encoding = self.option.encoding or term_encoding()

        if self.option.color:
            ramp = self.color_ramp(self.size.y)
            if not self.option.reverse:
                ramp = ramp[::-1]
        else:
            ramp = []

        if self.size.y > 1:
            self.set_text(Point((0, 0)), self.human(self.maximum))
            self.set_text(Point((0, self.size.y - 1)), self.human(self.minimum))

        for y in range(self.screen.size.y):
            if ramp:
                stream.write(ramp[y])
            for x in range(self.screen.size.x):
                point = Point((x, y))
                if point in self.screen:
                    value = self.screen[point]
                    if isinstance(value, int):
                        stream.write(unichr(value).encode(encoding))
                    else:
                        stream.write(unicode(value).encode(encoding))

                else:
                    stream.write(' ')

            if ramp:
                stream.write(self.term.csi('sgr0'))
            stream.write('\n')


def run():
    import argparse

    parser = argparse.ArgumentParser()

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

    group = parser.add_argument_group('optional input and output arguments')
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
    option.mode = option.mode or 'g'
    option.size = Point((option.width, option.height))

    if option.input in ['-', 'stdin']:
        istream = sys.stdin
    else:
        istream = open(option.input, 'r')

    if option.output in ['-', 'stdout']:
        ostream = sys.stdout
    else:
        ostream = open(option.output, 'r')

    option.encoding = option.encoding or Terminal().encoding

    points = []
    for line in istream:
        for point in line.split():
            try:
                points.append(float(point))
            except ValueError:
                pass

    if option.mode == 'g':
        engine = AxisGraph(option.size, points, option)

    elif option.mode == 'h':
        engine = HorizontalBarGraph(option.size, points, option)

    elif option.mode == 'v':
        engine = VerticalBarGraph(option.size, points, option)

    else:
        parser.error('invalid mode')
        return 1

    engine.render(ostream)


if __name__ == '__main__':
    sys.exit(run())
