#!/usr/bin/env python

'''
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, version 3.
 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


# convert a single hue to bgr
def hue_to_bgr(hue):
    hsv = np.array([[[hue, 255, 255]]], dtype='uint8')
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return [int(i) for i in color]


# convert a single hue to rgb
def hue_to_rgb(hue):
    hsv = np.array([[[hue, 255, 255]]], dtype='uint8')
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return [int(i) for i in color]


class ColorDetector:

    HUES = {
        'red': [-6, 14],
        'green': [50, 70],
        'blue': [107, 127],
    }
    HUE_THRESHOLD = 800
    CAMERA_CTRLS = {
        'auto_exposure': 1,
        # TODO
    }

    def __init__(self):
        self.array = self.Array()
        self.result = [None] * len(self.array)
        self.colors = self._compute_colors()

    def load_img(self, img):
        self.img = self.Image(img)

    def detect_colors(self):

        # return the indices between start and end with n periodicity
        def span(start, end, n):
            while start != end:
                start = start % n
                yield start
                start = start + 1

        # return the color in a region of interest
        def color(roi):
            for name, hues in self.HUES.items():
                hues = list(span(*hues, self.img.HIST_SIZE))
                if np.sum(self.img.hist(roi)[hues]) > self.HUE_THRESHOLD:
                    return name
            return None

        self.result = [color(roi) for roi in self.array.slices()]
        return self.result

    def plot_hist(self, i):

        # plot histogram
        roi = self.array.slices()[i]
        lines = plt.plot(self.img.hist(roi), color='black')

        # plot vertical line at each hue range
        for hues in self.HUES.values():
            for hue in hues:
                # color = np.array(hue % self.Image.HIST_SIZE) / 255
                hue = hue % self.Image.HIST_SIZE
                color = np.array(hue_to_rgb(hue)) / 255
                lines += [plt.axvline(x=hue, color=color, linestyle='--')]

        return lines

    def draw_circles(self, img):
        return self.array.draw(img, [self.colors.get(res)
                                     for res in self.result])

    def _compute_colors(self):
        return {k: hue_to_bgr(int(np.sum(v) / 2) % self.Image.HIST_SIZE)
                for k, v in self.HUES.items()}

    class Image:

        SATURATION = [100, 255]
        VALUE = [90, 255]
        HIST_SIZE = 180

        def __init__(self, img):
            self.rgb = img
            self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self.filter = self._create_filter(self.hsv)

        def hist(self, roi):
            return cv2.calcHist(images=[self.hsv[roi]],
                                channels=[0],
                                mask=self.filter[roi],
                                histSize=[self.HIST_SIZE],
                                ranges=[0, 180]).reshape(-1)

        @property
        def size(self):
            return self.rgb.shape[:2]

        @property
        def filtered(self):
            return cv2.bitwise_and(self.rgb, self.rgb, mask=self.filter)

        def _create_filter(self, hsv):

            # enough saturation
            mask_sat = cv2.inRange(hsv[:, :, 1], *self.SATURATION)

            # bright enough
            mask_val = cv2.inRange(hsv[:, :, 2], *self.VALUE)

            return cv2.bitwise_and(mask_sat, mask_val)

    class Array:

        RADIUS = 50
        ARRAY_SIZE = (5, 4)
        ARRAY_PITCH = (114, 112)
        ARRAY_POSITION = (340, 250)
        ARRAY_ROTATION = 1.5

        def __init__(self):
            self.positions = self._compute_positions()

        def slices(self):
            r = self.RADIUS
            return [(slice(y - r, y + r), slice(x - r, x + r))
                    for x, y in self.positions]


        def draw(self, img, colors, thickness=1):

            for pos, col in zip(self.positions, colors):

                if col is None:
                    col = [255, 255, 255]

                cv2.circle(img=img,
                           center=tuple(pos),
                           radius=self.RADIUS,
                           color=col,
                           thickness=thickness)

            return img

        def _compute_positions(self):
            nx, ny = self.ARRAY_SIZE
            dx, dy = self.ARRAY_PITCH
            px, py = self.ARRAY_POSITION
            positions_x = np.linspace(-1, 1, nx) * (nx - 1) / 2 * dx + px
            positions_y = np.linspace(-1, 1, ny) * (ny - 1) / 2 * dy + py

            rot = np.radians(self.ARRAY_ROTATION)
            rot = np.array([[np.cos(rot), np.sin(rot)],
                            [-np.sin(rot), np.cos(rot)]])

            def iter_positions():
                for y in positions_y:
                    for x in positions_x:
                        pos = rot.dot([x, y])
                        yield np.round(pos).astype(int)

            return list(iter_positions())

        def __len__(self):
            return self.ARRAY_SIZE[0] * self.ARRAY_SIZE[1]


def animate(i, camera, detector):
    if camera.isOpened():

        # read camera
        #ret, img = camera.read()
        img = cv2.imread('./potion_maker_test.png')

        # make the detector do its work
        detector.load_img(img)
        img = detector.draw_circles(detector.img.filtered)
        cv2.imshow('Frame', img)

        print('>>', detector.detect_colors())

    return detector.plot_hist(11)


if __name__ == '__main__':

    # open camera
    camera = cv2.VideoCapture(0)
    #camera.set(44, 0)

    # create color detector
    detector = ColorDetector()

    # matplotlib
    fig, ax = plt.subplots()
    ma = animation.FuncAnimation(fig,
                                 lambda i: animate(i, camera, detector),
                                 interval=10,
                                 blit=True,
                                 repeat=True)
    plt.show()

    camera.release()
    cv2.destroyAllWindows()
