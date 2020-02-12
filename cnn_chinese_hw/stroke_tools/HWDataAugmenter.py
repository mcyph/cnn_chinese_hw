import math
import random
import matplotlib
import numpy as np
from PIL import Image
from matplotlib import cm
from random import randint
from cnn_chinese_hw.stroke_tools.points_normalized import points_normalized
from cnn_chinese_hw.stroke_tools.points_to_plot import points_to_plot, draw_in_place
from cnn_chinese_hw.stroke_tools.get_vertex import get_vertex


class HWStrokesAugmenter:
    def __init__(self, strokes, find_vertices=False, vertice_error_scale=1.0):
        """

        """
        strokes = points_normalized(
            strokes, width=1000, height=1000
        )
        if find_vertices:
            strokes = [get_vertex(stroke, vertice_error_scale) for stroke in strokes]
        self.strokes = strokes

    def augment_strokes(self):
        """

        :return:
        """

        out_strokes = []
        for points in self.strokes:
            points = self.__points_rotated(points)  # Rotate strokes individually
            points = self.__points_scaled(points)
            points = self.__points_displaced(points)
            points = self.__points_randomised(points)
            out_strokes.append(points)
        out_strokes = points_normalized(out_strokes)

        out_strokes = self.__strokes_distorted_from_center(out_strokes)
        out_strokes = points_normalized(out_strokes)
        out_strokes = self.strokes_rotated(out_strokes)  # Rotate the whole thing
        out_strokes = points_normalized(out_strokes)

        return out_strokes

    def raster_strokes(self,
                       on_val=1, image_size=24,
                       do_augment=True):
        """

        :param raster_size:
        :return:
        """
        if do_augment:
            strokes = self.augment_strokes()
        else:
            strokes = self.strokes[:]

        # Make the points in `strokes` not exceed the rastered image size
        strokes = points_normalized(strokes, image_size - 1, image_size - 1)

        # Draw using the xiaolin wu antialised line algorithm,
        # outputting to a single-dimensional numpy array
        a = np.zeros(shape=(image_size, image_size), dtype=np.uint8)
        draw_in_place(a, strokes)
        return a

    def raster_strokes_multiple_times(self, on_val=10, image_size=24):
        a = np.zeros(shape=(image_size, image_size), dtype=np.uint8)

        for x in range(255//on_val):
            strokes = self.augment_strokes()
            strokes = points_normalized(strokes, image_size - 1, image_size - 1)
            draw_in_place(a, strokes)
        return a

    def raster_strokes_as_pil(self, myarray=None, image_size=48):
        """

        :return:
        """
        if myarray is None:
            myarray = self.raster_strokes(on_val=255, image_size=image_size)
        im = Image.fromarray(np.uint8(cm.gist_earth(myarray) * 255))
        return im

    def __get_mid_coord(self, points):
        """

        :param points:
        :return:
        """
        x_starts = min([i[0] for i in points])
        y_starts = min([i[1] for i in points])
        x_ends = max([i[0] for i in points])
        y_ends = max([i[1] for i in points])
        x_mid = x_starts + (x_ends - x_starts) // 2
        y_mid = y_starts + (y_ends - y_starts) // 2
        return x_mid, y_mid

    def __get_rotated(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = (
            ox +
            math.cos(angle) * (px - ox) -
            math.sin(angle) * (py - oy)
        )
        qy = (
            oy +
            math.sin(angle) * (px - ox) +
            math.cos(angle) * (py - oy)
        )
        return qx, qy

    def strokes_rotated(self, strokes):
        """

        :param strokes:
        :return:
        """
        # Note that I'm assuming more people will be right handed than
        # left, and thus will be more likely to rotate to the right.
        rotate_deg = random.uniform(-0.2, 0.5)
        out_strokes = []
        for points in strokes:
            out_strokes.append(self.__points_rotated(points, rotate_deg))
        return out_strokes

    def __points_rotated(self, points, rotate_deg=None):
        """

        :param points:
        :return:
        """
        out_points = []
        origin = self.__get_mid_coord(points)
        for point in points:
            if rotate_deg is None:
                rotate_deg = random.uniform(-0.05, 0.10) # Note this is in radians
            out_points.append(
                self.__get_rotated(origin, point, rotate_deg)
            )
        return out_points

    def __points_scaled(self, points):
        """

        :param points:
        :return:
        """
        xmid, ymid = self.__get_mid_coord(points)
        scale_by_x = random.uniform(0.8, 1.2)
        scale_by_y = random.uniform(0.8, 1.2)

        out_points = []
        for point in points:
            x = point[0]-xmid
            y = point[1]-ymid
            x *= scale_by_x
            y *= scale_by_y
            x += xmid
            y += ymid
            out_points.append((x, y))
        return out_points

    def __points_displaced(self, points):
        """

        :param points:
        :return:
        """
        xd = randint(-10, 10)  # ~1/10th the grid
        yd = randint(-10, 10)

        out_points = []
        for point in points:
            out_points.append((point[0]+xd, point[1]+yd))
        return out_points

    def __points_randomised(self, points):
        """

        :param points:
        :return:
        """
        out_points = []
        for point in points:
            xd = randint(-10, 10)  # 1/20th the grid
            yd = randint(-10, 10)
            out_points.append((point[0] + xd, point[1] + yd))
        return out_points

    def __strokes_distorted_from_center(self, strokes):
        # NOTE: I've made it max 1.3 as 2.0 can make things unrecognisably warped.
        # e.g. 30**1.3 = 83; 30**1.4 = 117
        # so even small changes can make a large difference
        # (1.2 may actually be better here!)
        x_sqrt = random.choice([False, True])
        x_pow = random.uniform(1.0, 1.25)
        y_sqrt = random.choice([False, True])
        y_pow = random.uniform(1.0, 1.25)

        LOut = []
        for points in strokes:
            LOut.append(self.__points_distorted_from_center(
                points, x_sqrt, x_pow, y_sqrt, y_pow
            ))
        return LOut

    def __points_distorted_from_center(self, points, x_sqrt, x_pow, y_sqrt, y_pow):
        """

        :param points:
        :return:
        """

        def randomise(x, use_sqrt, pow):
            if use_sqrt:
                # Square root (inverse on axis)
                # 1000x **
                return (1000*x ** (1.0 / pow)) / (1000 ** (1.0 / pow))
            else:
                # Square it
                # (1000x ** [2+]) / (1000 ** [2+])
                return (1000*x ** pow) / (1000 ** pow)

        out_points = []
        for point in points:
            i_point = (
                round(randomise(point[0], x_sqrt, x_pow)),
                round(randomise(point[1], y_sqrt, y_pow))
            )
            #print(point, i_point)
            out_points.append(i_point)
        return out_points


if __name__ == '__main__':
    def im_joined(images):
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        return new_im


    import time
    from cnn_chinese_hw.parse_data.StrokeData import StrokeData
    sd = StrokeData()
    SKIP = 3000

    # TODO: Maybe should look at Handright -
    #  https://github.com/Gsllchb/Handright
    # This may be able to augment the dataset

    for ord_, LStrokes in sd.iter():
        if SKIP:
            SKIP -= 1
            continue

        for i_LStrokes in LStrokes:
            strokes = [i.LPoints for i in i_LStrokes]
            #print(chr(ord_), ord_, strokes)
            aug = HWStrokesAugmenter(strokes)

            # Render on top lots of times, to give an idea
            # of where the lines will end up on average.
            LRastered = [
                aug.raster_strokes(on_val=255)
                for x in range(256)
            ]
            images = [aug.raster_strokes_as_pil(rastered)
                      for rastered in LRastered]
            im_joined(images).show()
            time.sleep(10)
    time.sleep(10)
