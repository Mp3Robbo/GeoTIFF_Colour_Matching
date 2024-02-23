This python script runs in QGIS and takes in two tifs, one for correction and another as a target, and adjusts the pixel values of the first to try to match the colour scheme of the second

It does this by creating a scatter of points in the area where the tifs overlap, and compares the RGB values

From this it creates a multiple regression formula for the new RGB pixel values of the tif needing correction

____________________________________________

I haven't fully tidied up this script, nor thoroughly tested it so let me know if there are any issues

_____________________________________________

It appears Esri have their own tool for this which mosaicks many rasters together
