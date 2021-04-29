from osgeo import gdal 
import matplotlib.pyplot as plt 
from matplotlib import colors
files = [
    'data/test/belgrad/90.tif',
    'data/output/belgrad_predict.tif',
    'data/output/transformed.tif',
]

bounds = [
    (1000+5, 3000-5, 5, 2500-5),
    None,
    None,
]

output_names = [
    'belgrad_orig.png',
    'belgrad_predict.png',
    'belgrad_predict_no_colors.png',
]

for filename, bound, out_name in zip(files, bounds, output_names):

    dt = gdal.Open(filename)

    print(dt.RasterCount)

    bands = dt.GetRasterBand(1).ReadAsArray()

    if bound is not None:
        bands = bands[bound[0]:bound[1], bound[2]:bound[3]]

    cmap = colors.ListedColormap(['blue','green','black','yellow', 'grey', 'red'])
    bounds=[0, 1, 2, 3, 4, 5, 6]

    plt.imshow(bands, cmap=cmap)
    plt.colorbar() 
    plt.savefig(out_name) 
    plt.show()