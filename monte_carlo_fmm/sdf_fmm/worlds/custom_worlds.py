from sdf_gen.random_shapes import box_sdf, circle_sdf
from sdf_gen.utils import combine_sdfs
import numpy as np

def custom_two_box_world(nx, ny, map_size):
    cx1, cy1 = -1.0, 0.5
    hw1, hh1 = 0.7, 0.3
    box1 = box_sdf(nx, ny, cx1, cy1, hw1, hh1)

    cx2, cy2 = 1.0, -0.5
    hw2, hh2 = 0.4, 0.4
    box2 = box_sdf(nx, ny, cx2, cy2, hw2, hh2)

    return combine_sdfs(box1, box2)

def custom_circle_box_world(nx, ny, map_size):
    circle = circle_sdf(nx, ny, cx=0.0, cy=0.0, r=1.0)
    box    = box_sdf(nx, ny, cx=1.5, cy=1.0, half_w=0.5, half_h=0.4)
    return combine_sdfs(circle, box)

def custom_world(nx, ny):
    sdf = np.ones((nx, ny)) * 999  # everything is free
    sdf[20:40, 10:20] = -1          # mark a block as obstacle
    return sdf
