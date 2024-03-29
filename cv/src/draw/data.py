import numpy as np
import tensorflow as tf
import cv2 as cv 
import random
import math

# https://coolors.co/227c9d-17c3b2-ffcb77-fef9ef-fe6d73
# Note: OpenCV represents colors in BGR format (not RGB).
_colors = {'blue':   [0x9d, 0x7c, 0x22], 
           'green':  [0xb2, 0xc3, 0x17],
           'yellow': [0x77, 0xcb, 0xff],
           'red':    [0x73, 0x6d, 0xfe]}

def _draw_circle(img, center, diameter, color):
    # thickness=-1 results in filling the shape.
    cv.circle(img, tuple(center), diameter//2, color, thickness=-1, lineType=cv.LINE_AA)


def _draw_square(img, center, length, color):
    top_left = (center[0] - length//2, center[1] - length//2)
    bottom_right = (center[0] + length//2, center[1] + length//2)
    cv.rectangle(img, top_left, bottom_right, color, thickness=-1, lineType=cv.LINE_AA)


def _draw_equilateral(img, center, length, color):
    # It seems that fillConvexPoly needs an np array.
    # p1 = (center[0], int(center[0] + math.sqrt(3)/4))
    # p2 = (center[0] + length//2, int(center[1] - math.sqrt(3)/4))
    # p3 = (center[0] - length//2, int(center[1] - math.sqrt(3)/4))
    # triangle = [p1, p2, p3]
    # print(p1); print(p2); print(p3)
    triangle = np.array([
        [center[0]            , int(center[1] - length * math.sqrt(3)/4)],
        [center[0] + length//2, int(center[1] + length * math.sqrt(3)/4)],
        [center[0] - length//2, int(center[1] + length * math.sqrt(3)/4)]])
    cv.fillConvexPoly(img, triangle, color, lineType=cv.LINE_AA)
    

def _draw_shape(img):
    shape_fns = (_draw_circle, _draw_square, _draw_equilateral)
#    shape_fns = (_draw_square, _draw_equilateral)
    shape_fn = random.choice(shape_fns)
    min_len = 6
    max_len = max(img.shape)//3
    length = random.randrange(min_len, max_len + 1)
    center = (random.randrange(length//2, img.shape[0]), 
              random.randrange(length//2, img.shape[1]))
    assert all(i >= 0 for i in center)
    color = random.choice(list(_colors.values()))
    shape_fn(img, center, length, color)


def _shape_generator(num_samples, img_shape, seed=None):
    i = 0
    infinite = num_samples is None
    while infinite or i < num_samples:
        color_dim_len = 3
        # 1s or 255?
        init_color = 255
        img = np.full((*img_shape, color_dim_len), init_color, np.uint8)
        max_shapes = 4
        min_shapes = 1
        num_shapes = random.randrange(min_shapes, max_shapes + 1)
        for s in range(num_shapes):
            _draw_shape(img)
        yield img
        i += 1


def _black_white_shape_generator(num_samples, img_shape, seed=None):
    shape_gen = _shape_generator(num_samples, img_shape, seed)
    i = 0
    infinite = num_samples is None
    while infinite or i < num_samples:
        color_img = next(shape_gen)
        grey_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
        threshold = 250
        (_, bw_img) = cv.threshold(grey_img, threshold, 255, cv.THRESH_BINARY)
        yield bw_img
        i += 1


class ShapeDataset1(tf.data.Dataset):

    def __new__(cls, num_samples=None, img_shape=[64, 64]):
        return tf.data.Dataset.from_generator(
            _shape_generator,
            output_types=tf.dtypes.uint8,
            output_shapes=(*img_shape, 3),
            args=(num_samples, img_shape))

class BlackWhiteShapeDataset1(tf.data.Dataset):

    def __new__(cls, num_samples=None, img_shape=[64, 64]):

        return tf.data.Dataset.from_generator(
            lambda :_black_white_shape_generator(num_samples, img_shape),
            output_types=tf.dtypes.uint8,
            output_shapes=img_shape)


if __name__ == "__main__":
    print("Writing images...")
    # for idx, sample in enumerate(_shape_generator(10, (64, 64))):
    for idx, sample in enumerate(_black_white_shape_generator(10, (64, 64))):
        cv.imwrite(f'{str(idx)}.png', sample)
    print("Done")
        
