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

# Setup terminal
curses.setupterm()


def term_colors():
    return curses.tigetnum('colors')


def term_csi(capname, *args):
    '''
    Returns the matching ANSI Control Sequence Introducer (CSI).
    '''
    return curses.tparm(curses.tigetstr(capname), *args)


def term_csi_wrap(data, capname, *args):
    '''
    Wraps ``data`` in a CSI sequence and resets it afterwards.
    '''
    return ''.join([
        term_csi(capname, *args),
        data,
        term_csi('sgr0'),
    ])


def term_encoding():
    lang, encoding = locale.getdefaultlocale()
    return encoding


def term_size():
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


def term_width():
    return term_size()[0]


def term_height():
    return term_size()[1]


def unirev(i):
    '''
    Reverse video unicode character.
    '''
    return term_csi_wrap(unichr(i), 'rev')


# Bar characters (8 per map)
HBARN = map(unichr, (0x258f, 0x258e, 0x258d, 0x258c,
                     0x258b, 0x258a, 0x2589, 0x2588))
HBARR = map(unirev, (0x00a0, 0x2589, 0x258a, 0x258b,
                     0x258c, 0x258d, 0x258e, 0x258f))
VBARN = map(unichr, (0x2581, 0x2582, 0x2583, 0x2584,
                     0x2585, 0x2586, 0x2587, 0x2588))
VBARR = map(unirev, (0x2588, 0x2587, 0x2586, 0x2585,
                     0x2584, 0x2583, 0x2582, 0x00a0))

# Color ramp maps
RAMPS = {
    0x010: [curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED],
    0x100: [
        51 - x for x in range(6)                # blue -> green
    ] + [
        46 + x * 36 for x in range(6)          # green -> yellow
    ] + [
        226 - x * 6 for x in range(6)          # yellow -> red
    ]
}

# Units
UNITS = ' kMGT'


class Point(object):
    def __init__(self, coordinates):
        self.x = coordinates[0]
        self.y = coordinates[1]

    def __repr__(self):
        return 'Point(%r, %r)' % (self.x, self.y)


class Screen(object):
    def __init__(self, size, encoding=None, extend_x=False, extend_y=False):
        self.set_size(size)
        self.encoding = encoding or term_encoding()
        self.extend_x = extend_x
        self.extend_y = extend_y

        defaultdictint = lambda: defaultdict(int)
        self.canvas = defaultdict(defaultdictint)

    def set_size(self, size):
        if not isinstance(size, Point):
            size = Point(size)
        print 'set size', size
        self.size = size

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

        if not self.extend_y and point.y > self.size.y:
            raise OverflowError('%r overflow y = %d > %d' % (self, point.y, self.size.y))
        else:
            self.size.y = point.y

        if not self.extend_x and point.x > self.size.x:
            raise OverflowError('%r overflow x = %d > %d' % (self, point.x, self.size.x))
        else:
            self.size.x = point.x

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

        self.minimum = min(self.points)
        self.maximum = max(self.points)
        self.extents = (self.maximum - self.minimum)

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
            size.x or term_width(),
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
        encoding = self.option.encoding or term_encoding()

        zero = int(self.null / 4)  # Zero crossing
        self.set_text(Point((0, zero)), '0')
        self.set_text(Point((0, 0)), _human(self.maximum))
        self.set_text(Point((0, self.size.y - 1)), _human(self.minimum))
        
        for y in range(self.screen.size.y):
            if y == zero:
                stream.write(term_csi('smul'))

            for x in range(self.screen.size.x):
                point = Point((x, y))
                if point in self.screen:
                    value = self.screen[point]
                    if isinstance(value, int):
                        stream.write(unichr(self.base + value).encode(encoding))
                    else:
                        stream.write(term_csi('bold'))
                        stream.write(unicode(value).encode(encoding))
                        # There's no capability to just turn off bold, meh.
                        stream.write(term_csi('sgr0'))
                        if y == zero:
                            stream.write(term_csi('smul'))
                else:
                    stream.write(' ')
            
            if y == zero:
                stream.write(term_csi('rmul'))
            
            stream.write('\n')


    @property
    def normalised(self):
        dx = (self.screen.width / len(self.points))
        oy = (self.screen.height)
        for x, point in enumerate(self.points):
            y = (point - self.minimum) * 4.0 / self.extents * self.size.y
            yield Point((
                dx * x,
                oy - y,
            ))

    @property
    def null(self):
        return self.screen.height - (-self.minimum * 4.0 / self.extents * self.size.y)

    def set(self, point):
        if not isinstance(point, Point):
            point = Point(point)
        
        rx = self.round(point.x)
        ry = self.round(point.y)
        
        item = Point((rx >> 1, ry >> 2))
        self.screen[item] |= self.pixels[ry % 4][rx % 2]

    def set_text(self, point, text):
        if not isinstance(point, Point):
            point = Point(point)

        for offset, char in enumerate(str(text)):
            self.screen.canvas[point.y][point.x + offset] = char

    def unset(self, point):
        if not isinstance(point, Point):
            point = Point(point)

        x, y = self.round(point.x) >> 1, self.round(point.y) >> 2

        if (x, y) not in self.screen:
            return

        if isinstance(self.screen[y][x], (int, long)):
            self.screen[(x, y)] &= ~self.pixels[y % 4][x % 2]

        else:
            del self.screen[(x, y)]

        if not self.canvas.get(y):
            del self.screen[y]


class BarGraph(Graph):
    @property
    def normalised(self):
        # map(lambda p: (p - minimum) / extents * width, points)
        for point in self.points:
            yield (point - self.minimum) / self.extents * self.scale


class HorizontalBarGraph(BarGraph):
    def __init__(self, size, points, option):
        super(BarGraph, self).__init__(size, points, option)

        if size.y:
            warnings.warn('Ignoring height on horizontal bar graph')

        self.size = Point((
            size.x or term_width(),
            1,
        ))
        print self.size
        self.screen = Screen(
            Point((self.size.x * 8, 1)),
            extend_y=True,
        )

        # Select block characters
        if self.option.reverse:
            self.blocks = HBARR
        else:
            self.blocks = HBARN

        # Plot bars for each line
        for y, size in enumerate(self.normalised):
            self.bar(size, y)

    def bar(self, size, y):
        full, frac = divmod(self.round(size * 8), 8)

        print self.screen

        x = 0
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

        for y in range(self.screen.size.y):
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
            stream.write('\n')


def goto(x, y):
    return term_csi('cup', y, x)


def color_ramp(size):
    if size is None:
        raise ValueError("Can't calculate color ramp without a size")

    scale = []
    cramp = RAMPS.get(term_colors())
    ratio = float(len(cramp)) / float(size)
    if cramp:
        for i in range(size):
            scale.append(
                term_csi('setaf', cramp[int(ratio * i)])
            )
    return scale





def _human(size, base=1000, units=' kMGTZ'):
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


def hbar(size, width, stream=sys.stdout, reverse=False, encoding=None, ramp=[]):
    '''
    Draws a horizontal bar with `size` width, can be a float.
    '''

    encoding = encoding or term_encoding()

    size, frac = divmod(size, 1)
    size = int(size)
    frac = int(frac * 8)

    if reverse:
        stream.write(term_csi('cuf', width))

    o = 0
    for o in range(size):
        if reverse:
            block = HBARR[0]
        else:
            block = HBARN[-1]

        if ramp:
            stream.write(ramp[o])
        stream.write(block.encode(encoding))
        if reverse:
            stream.write(term_csi('cub', 2))

    if frac > 0:
        if reverse:
            block = HBARR[-frac]
        else:
            block = HBARN[frac - 1]

        if ramp:
            stream.write(ramp[o])
        stream.write(block.encode(encoding))
        if reverse:
            stream.write(term_csi('cub', 2))

    # Reset color
    if ramp:
        stream.write(term_csi('sgr0'))


def hbargraph(points, width, stream, reverse=False, color=False):
    if width == 0:
        width = term_width()

    minimum = min(points)
    maximum = max(points)
    extents = maximum - minimum
    normals = map(lambda p: (p - minimum) / extents * width, points)

    if color:
        ramp = color_ramp(width)
    else:
        ramp = []

    for y, point in enumerate(normals):
        hbar(
            point,
            width,
            stream=stream,
            reverse=reverse,
            ramp=ramp,
        )
        sys.stdout.write('\n')


def vbar(size, height, stream=sys.stdout, reverse=True, encoding=None, ramp=[]):
    '''
    Draws a vertical bar with `size` height, can be a float.
    '''

    encoding = encoding or term_encoding()

    size, frac = divmod(size, 1)
    size = int(size)
    full = VBARR[-1] if reverse else VBARN[-1]

    if reverse:
        # Move up height rows
        if height > 1:
            stream.write(term_csi('cuu', height - 1))

    # Draw columns bottom up
    o = 0
    u = 0
    if size:
        for o in xrange(size):
            if ramp:
                stream.write(ramp[o])
            stream.write(full.encode(encoding))
            if reverse:
                stream.write(term_csi('cud', 1)) # go down a row
            else:
                stream.write(term_csi('cuu', 1)) # go back a row
            stream.write(term_csi('cub', 1)) # go back a column
            u += 1

    if frac > 0:
        bfrac = int(frac * 8)
        block = VBARR[bfrac] if reverse else VBARN[bfrac]
        if ramp:
            stream.write(ramp[o])
        stream.write(block.encode(encoding))
        if reverse:
            stream.write(term_csi('cud', 1)) # go down a row
        else:
            stream.write(term_csi('cuu', 1)) # go back a row
        u += 1

    # Move to the original position
    if reverse:
        if height - u > 0:
            stream.write(term_csi('cud', height - u))
    else:
        stream.write(term_csi('cud', u))

    # Reset color
    if ramp:
        stream.write(term_csi('sgr0'))


def vbargraph(points, height, stream, reverse=False, color=False):
    if height == 0:
        height = 1

    minimum = min(points)
    maximum = max(points)
    extents = maximum - minimum
    normals = map(lambda p: (p - minimum) / extents * height, points)

    ramp = color_ramp(height)
    if height - 1 > 0:
        stream.write('\n' * (height - 1))
    for x, point in enumerate(normals):
        vbar(
            point,
            height,
            stream=stream,
            reverse=reverse,
            ramp=ramp,
        )
    sys.stdout.write('\r\n')


def run():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-A', '--axis',
        dest='mode', action='store_const', const='a',
        help='axis drawing mode',
    )
    parser.add_argument(
        '-H', '--horizontal',
        dest='mode', action='store_const', const='h',
        help='horizontal drawing mode',
    )
    parser.add_argument(
        '-V', '--vertical',
        dest='mode', action='store_const', const='v', default='v',
        help='vertical drawing mode',
    )
    parser.add_argument(
        '-c', '--color',
        dest='color', action='store_const', const=True, default=True,
    )
    parser.add_argument(
        '-C', '--no-color',
        dest='color', action='store_const', const=False,
    )
    parser.add_argument(
        '-r', '--reverse', default=False, action='store_true',
        help='reverse draw graph',
    )
    parser.add_argument(
        '-x', '--width', default=0, type=int, metavar='characters',
        help='drawing width (default: auto)',
    )
    parser.add_argument(
        '-y', '--height', default=0, type=int, metavar='characters',
        help='drawing height (default: auto)',
    )
    parser.add_argument(
        '-i', '--input', default='-', metavar='file',
        help='input file (default: stdin)',
    )
    parser.add_argument(
        '-o', '--output', default='-', metavar='file',
        help='output file (default: stdout)',
    )
    parser.add_argument(
        '-e', '--encoding',
        dest='encoding', default='',
        help='output encoding (default: auto)',
    )
    
    option = parser.parse_args()
    option.size = Point((option.width, option.height))

    if option.input in ['-', 'stdin']:
        istream = sys.stdin
    else:
        istream = open(option.input, 'r')

    if option.output in ['-', 'stdout']:
        ostream = sys.stdout
    else:
        ostream = open(option.output, 'r')

    option.encoding = option.encoding or term_encoding()

    points = []
    for line in istream:
        for point in line.split():
            try:
                points.append(float(point))
            except ValueError:
                pass

    if option.mode == 'a':
        # graph(points, option.size, stream=ostream)
        engine = AxisGraph(option.size, points, option)
        
    elif option.mode == 'v':
        #engine = VerticalBarGraph(option.size, points)
        pass

    elif option.mode == 'h':
        engine = HorizontalBarGraph(option.size, points, option)

    engine.render(ostream, option)

if __name__ == '__main__':
    sys.exit(run())
