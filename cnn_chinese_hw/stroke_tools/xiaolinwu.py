"""
This file uses functions from the following file for antialised lines using
the "Xiaolin Wu" algorithm:
https://github.com/rcarmo/pngcanvas/blob/master/pngcanvas.py

File under the MIT License (MIT)

Copyright (c) 2013 Rui Carmo

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def point(array, x, y, alpha=255):
    """
    Set a pixel
    """
    if x < 0 or y < 0 or x > len(array[0]) - 1 or y > len(array[1]) - 1:  #CHECK ARRAY IDX!!!
        return
    array[y, x] = min(array[y, x] + alpha, 255)


def intensity(c, i):
    """
    Compute a new alpha given a 0-0xFF intensity
    """
    return [c[0], c[1], c[2], (c[3] * i) >> 8]


def line(array, x0, y0, x1, y1):
    """
    Draw a line using Xiaolin Wu's antialiasing technique
    """
    # clean params
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if y0 > y1:
        y0, y1, x0, x1 = y1, y0, x1, x0
    dx = x1 - x0
    if dx < 0:
        sx = -1
    else:
        sx = 1
    dx *= sx
    dy = y1 - y0

    # 'easy' cases
    if dy == 0:
        for x in range(x0, x1, sx):
            point(array, x, y0)
        return
    if dx == 0:
        for y in range(y0, y1):
            point(array, x0, y)
        point(array, x1, y1)
        return
    if dx == dy:
        for x in range(x0, x1, sx):
            point(array, x, y0)
            y0 += 1
        return

    # main loop
    point(array, x0, y0)
    e_acc = 0
    if dy > dx:  # vertical displacement
        e = (dx << 16) // dy
        for i in range(y0, y1 - 1):
            e_acc_temp, e_acc = e_acc, (e_acc + e) & 0xFFFF
            if e_acc <= e_acc_temp:
                x0 += sx
            w = 0xFF-(e_acc >> 8)
            point(array, x0, y0, w)
            y0 += 1
            point(array, x0 + sx, y0, 0xFF - w)
        point(array, x1, y1)
        return

    # horizontal displacement
    e = (dy << 16) // dx
    for i in range(x0, x1 - sx, sx):
        e_acc_temp, e_acc = e_acc, (e_acc + e) & 0xFFFF
        if e_acc <= e_acc_temp:
            y0 += 1
        w = 0xFF-(e_acc >> 8)
        point(array, x0, y0, w)
        x0 += sx
        point(array, x0, y0 + 1, 0xFF-w)
    point(array, x1, y1)
