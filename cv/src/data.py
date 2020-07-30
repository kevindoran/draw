import numpy as np
import tensorflow as tf
import cv2 as cv 
import random
import math

# https://coolors.co/227c9d-17c3b2-ffcb77-fef9ef-fe6d73
_colors = {'blue':   [0x22, 0x7c, 0x9d], 
           'green':  [0x17, 0xc3, 0xb2],
           'yellow': [0xff, 0xcb, 0x77],
           'red':    [0xfe, 0x6d, 0x73]}

def _draw_circle(img, center, diameter, color):
    # thickness=-1 results in filling the shape 
    cv.circle(img, tuple(center), diameter//2, color, thickness=-1, lineType=cv.LINE_AA)


def _draw_square(img, center, length, color):
    top_left = (center[0] - length//2, center[1] - length//2)
    bottom_right = (center[0] + length//2, center[1] + length//2)
    print(f'{top_left}, {bottom_right}')
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
    for n in range(num_samples):
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


class ShapeDataset1(tf.data.Dataset):

    def __new__(cls, num_samples=10, img_shape=[64, 64]):
        return tf.data.Dataset.from_generator(
            _shape_generator,
            output_types=tf.dtypes.uint8,
            output_shapes=(*img_shape, 3),
            args=(num_samples, img_shape))


if __name__ == "__main__":
    print("Writing images...")
    for idx, sample in enumerate(_shape_generator(10, (64, 64))):
        cv.imwrite(f'{str(idx)}.png', sample)
    print("Done")
        
