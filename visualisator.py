from osgeo import gdal 
import matplotlib.pyplot as plt 
from matplotlib import colors
files = [
    'data/test/tst/85.tif',
    'data/output/85_predict.tif',
]

bounds = [
    (0+5, 3000-5, 0, 3000-5),
    None,
]

output_names = [
    '82_orig.png',
    '82_predict.png',
]

for filename, bound, out_name in zip(files, bounds, output_names):

    dt = gdal.Open(filename)

    print(dt.RasterCount)

    bands = dt.GetRasterBand(1).ReadAsArray()

    if bound is not None:
        bands = bands[bound[0]:bound[1], bound[2]:bound[3]]
    
    cmap = colors.ListedColormap(['blue','green','black','yellow', 'grey', 'red'])

    plt.imshow(bands, cmap=cmap,aspect='equal',vmax=6, vmin=0.5)
    plt.colorbar() 
    plt.savefig(out_name) 
    plt.show()