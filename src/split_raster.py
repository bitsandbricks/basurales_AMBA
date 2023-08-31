#!/usr/bin/env python
# coding: utf-8

import subprocess
import gdal
import sys


def gdalwarp(*args):
    return subprocess.check_call(['gdalwarp'] + list(args))


src_path = sys.argv[1]
ds = gdal.Open(src_path)

try:
    out_base = sys.argv[2]
except IndexError:
    out_base = '/tmp/test_'

gt = ds.GetGeoTransform()

width_px = ds.RasterXSize
height_px = ds.RasterYSize

# Get coords for lower left corner
xmin = int(gt[0])
xmax = int(gt[0] + (gt[1] * width_px))

# get coords for upper right corner
if gt[5] > 0:
    ymin = int(gt[3] - (gt[5] * height_px))
else:
    ymin = int(gt[3] + (gt[5] * height_px))

ymax = int(gt[3])

# split height and width into n - i.e. this will produce 9 tiles
tile_width = (xmax - xmin) // 3
tile_height = (ymax - ymin) // 3

for x in range(xmin, xmax, tile_width):
    for y in range(ymin, ymax, tile_height):
        gdalwarp('-te', str(x), str(y), str(x + tile_width),
                 str(y + tile_height), '-multi', '-wo', 'NUM_THREADS=ALL_CPUS',
                 '-wm', '500', src_path, out_base + '{}_{}.tif'.format(x, y))