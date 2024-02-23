import numpy
import time
from sklearn import linear_model
from datetime import datetime

#Where the processing folder will be made
rootDirectory = 'D:/Other Tasks/Image Colour Matching/'

#The image that needs correction
initialImage = 'D:/Other Tasks/Image Colour Matching/CKingTest.tif'

#The image that has the desired look
targetImage = 'D:/Other Tasks/Image Colour Matching/MSandflyAllensRivuletFringe10cm.tif'

#Amount of correction to apply, 1.0 can be a bit extreme, 0.6 is a nice value to start with
correctionAmount = 0.5

#Tif export options
compressOptions = 'COMPRESS=LZW|PREDICTOR=2|NUM_THREADS=8|BIGTIFF=IF_SAFER|TILED=YES'
compressOptionsFloat = 'COMPRESS=LZW|PREDICTOR=1|NUM_THREADS=8|BIGTIFF=IF_SAFER|TILED=YES'
gdalOptions = '--config GDAL_NUM_THREADS ALL_CPUS -overwrite'


"""
##############################################################################
Initial prep
"""

#Set up the layer name for the raster calculations
initImName = initialImage.split("/")
initImName = initImName[-1]
initImName = initImName[:len(initImName)-4]

directory = rootDirectory + initImName + datetime.now().strftime("%Y%m%d%H%M") + '/'
try:
    os.mkdir(directory)
except BaseException as e:
    print(e)
    

#Determine extents and pixel sizes for later processing
initRas = QgsRasterLayer(initialImage)
pixelSizeXInit = initRas.rasterUnitsPerPixelX()
pixelSizeYInit = initRas.rasterUnitsPerPixelY()
pixelSizeAveInit = (pixelSizeXInit + pixelSizeYInit) / 2

targetRas = QgsRasterLayer(targetImage)
pixelSizeXTarget = targetRas.rasterUnitsPerPixelX()
pixelSizeYTarget = targetRas.rasterUnitsPerPixelY()
pixelSizeAveTarget = (pixelSizeXTarget + pixelSizeYTarget) / 2

pixelSizeMax = max(pixelSizeAveInit,pixelSizeAveTarget)
pixelSizeResamp = pixelSizeMax * 2

initRasBounds = initRas.extent() 
xminInitRas = initRasBounds.xMinimum()
xmaxInitRas = initRasBounds.xMaximum()
yminInitRas = initRasBounds.yMinimum()
ymaxInitRas = initRasBounds.yMaximum()
coordsInitRas = "%f %f %f %f" %(xminInitRas, yminInitRas, xmaxInitRas, ymaxInitRas)
coordsExpandedInitRas = "%f %f %f %f" %(xminInitRas - (pixelSizeResamp * 6), ymaxInitRas + (pixelSizeResamp * 6), xmaxInitRas + (pixelSizeResamp * 6), yminInitRas - (pixelSizeResamp * 6))

provider = initRas.dataProvider()
stats = provider.bandStatistics(1, QgsRasterBandStats.All, initRasBounds, 0)
maximumPixelVal = stats.maximumValue

"""
##############################################################################
Image prep and sampling
"""

#Resample to the larger pixel size so that any parallax-type issues are mitigated
processing.run("gdal:translate", {'INPUT':targetImage,'TARGET_CRS':None,'NODATA':None,'COPY_SUBDATASETS':False,'OPTIONS':compressOptions,'EXTRA':'-tr ' + str(pixelSizeResamp) + ' ' + str(pixelSizeResamp) + ' -r cubic','DATA_TYPE':0,'OUTPUT':directory + 'TargetResample.tif'})
processing.run("gdal:translate", {'INPUT':initialImage,'TARGET_CRS':None,'NODATA':None,'COPY_SUBDATASETS':False,'OPTIONS':compressOptions,'EXTRA':'-tr ' + str(pixelSizeResamp) + ' ' + str(pixelSizeResamp) + ' -r cubic','DATA_TYPE':0,'OUTPUT':directory + 'InitialResample.tif'})


"""
##############################################################################
Grass


#Varibility
processing.run("gdal:translate", {'INPUT':initialImage,'TARGET_CRS':None,'NODATA':-1,'COPY_SUBDATASETS':False,'OPTIONS':compressOptions,
'EXTRA':'-tr ' + str(4*pixelSizeResamp) + ' ' + str(4*pixelSizeResamp) + ' -r nearest -projwin ' + coordsExpandedInitRas,'DATA_TYPE':0,'OUTPUT':directory + 'InitialResampleAgain.tif'})
processing.run("grass7:r.neighbors", {'input':directory + 'InitialResampleAgain.tif','selection':None,'method':6,'size':35,'gauss':5,
'quantile':None,'-c':True,'-a':True,'weight':'','output':directory + 'StdDevOfInitial.tif','nprocs':8,'GRASS_REGION_PARAMETER':None,'GRASS_REGION_CELLSIZE_PARAMETER':0,'GRASS_RASTER_FORMAT_OPT':'','GRASS_RASTER_FORMAT_META':''})
processing.run("gdal:warpreproject", {'INPUT':directory + 'StdDevOfInitial.tif','TARGET_RESOLUTION':None,'RESAMPLING':3,'OPTIONS':compressOptionsFloat + '|DISCARD_LSB=14','DATA_TYPE':6,'MULTITHREADING':True,
'EXTRA':'-te ' + coordsInitRas + ' -tr ' + str(pixelSizeXInit) + ' ' + str(pixelSizeYInit),'OUTPUT':directory + 'StdDevOfInitialResamp.tif'})

sDevRas = QgsRasterLayer(directory + 'StdDevOfInitial.tif')
provider = sDevRas.dataProvider()
stats = provider.bandStatistics(1, QgsRasterBandStats.All, initRasBounds, 0)
maximumPixelValSDev = stats.maximumValue


processing.run("gdal:rastercalculator", {'INPUT_A':initialImage,'BAND_A':1,'INPUT_B':initialImage,'BAND_B':2,'INPUT_C':initialImage,'BAND_C':3,'INPUT_D':directory + 'StdDevOfInitialResamp.tif','BAND_D':1,
'FORMULA':'((float64(B)-((float64(A)+float64(C))/2))**0.9)*((' + str(maximumPixelValSDev) + '- float64(D))**((float64(B)/'+str(maximumPixelVal)+')**0.2))',
'NO_DATA':0,'RTYPE':5,'OPTIONS':compressOptionsFloat + '|DISCARD_LSB=14','EXTRA':'--overwrite','OUTPUT':directory + 'GrassChance.tif'})

processing.run("gdal:warpreproject", {'INPUT':directory + 'GrassChance.tif','TARGET_RESOLUTION':None,'RESAMPLING':0,'OPTIONS':compressOptions,'DATA_TYPE':3,'MULTITHREADING':True,
'EXTRA':'-srcnodata None -dstnodata None','OUTPUT':directory + 'GrassChanceClean.tif'})


##############################################################################
Colour
"""

processing.run("gdal:rastercalculator", {'INPUT_A':initialImage,'BAND_A':1,'INPUT_B':initialImage,'BAND_B':2,'INPUT_C':initialImage,'BAND_C':3,
        'FORMULA':'float64(A) - ((float64(B)+float64(C))/2)'
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat + '|DISCARD_LSB=14','EXTRA':'--overwrite','OUTPUT':directory + 'RelativeRed.tif'})
        
processing.run("gdal:rastercalculator", {'INPUT_A':initialImage,'BAND_A':1,'INPUT_B':initialImage,'BAND_B':2,'INPUT_C':initialImage,'BAND_C':3,
        'FORMULA':'float64(B) - ((float64(A)+float64(C))/2)'
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat + '|DISCARD_LSB=14','EXTRA':'--overwrite','OUTPUT':directory + 'RelativeGreen.tif'})
        
processing.run("gdal:rastercalculator", {'INPUT_A':initialImage,'BAND_A':1,'INPUT_B':initialImage,'BAND_B':2,'INPUT_C':initialImage,'BAND_C':3,
        'FORMULA':'float64(C) - ((float64(A)+float64(B))/2)'
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat + '|DISCARD_LSB=14','EXTRA':'--overwrite','OUTPUT':directory + 'RelativeBlue.tif'})



processing.run("gdal:buildvirtualraster", {'INPUT':[directory + 'RelativeRed.tif',directory + 'RelativeGreen.tif',directory + 'RelativeBlue.tif'],
'RESOLUTION':2,'SEPARATE':True,'PROJ_DIFFERENCE':True,'ADD_ALPHA':False,'ASSIGN_CRS':None,'RESAMPLING':0,'SRC_NODATA':'','EXTRA':'','OUTPUT':directory + 'RelativeVirtual.vrt'})

#Export this out to a tif
processing.run("gdal:warpreproject", {'INPUT':directory + 'RelativeVirtual.vrt','SOURCE_CRS':None,'TARGET_CRS':None,'RESAMPLING':0,'NODATA':None,'TARGET_RESOLUTION':None,'OPTIONS':compressOptionsFloat,'DATA_TYPE':0,'TARGET_EXTENT':None,
'TARGET_EXTENT_CRS':None,'MULTITHREADING':True,'EXTRA':'-co \"PHOTOMETRIC=RGB\" -srcalpha -dstalpha ' + gdalOptions,'OUTPUT':directory + 'RelativeTogether.tif'})

"""
##############################################################################
Image prep and sampling
"""


#Get the extent of the target raster
processing.run("gdal:translate", {'INPUT':targetImage,'TARGET_CRS':None,'NODATA':None,'COPY_SUBDATASETS':False,'OPTIONS':compressOptions,'EXTRA':'-b 4 -scale_1 128 255 -1000 1255','DATA_TYPE':0,'OUTPUT':directory + 'TargetAlpha.tif'})
processing.run("gdal:polygonize", {'INPUT':directory + 'TargetAlpha.tif','BAND':1,'FIELD':'DN','EIGHT_CONNECTEDNESS':False,'EXTRA':'','OUTPUT':directory + 'TargetAlphaExtent.gpkg'})
processing.run("native:fixgeometries", {'INPUT':directory + 'TargetAlphaExtent.gpkg','OUTPUT':directory + 'TargetAlphaExtentFix.gpkg'})
processing.run("native:extractbyexpression", {'INPUT':directory + 'TargetAlphaExtentFix.gpkg','EXPRESSION':' \"DN\" > 245','OUTPUT':directory + 'TargetAlphaExtentFilt.gpkg'})

#Get the extent of the initial raster
processing.run("gdal:translate", {'INPUT':initialImage,'TARGET_CRS':None,'NODATA':None,'COPY_SUBDATASETS':False,'OPTIONS':compressOptions,'EXTRA':'-b 4 -scale_1 128 255 -1000 1255','DATA_TYPE':0,'OUTPUT':directory + 'InitialAlpha.tif'})
processing.run("gdal:polygonize", {'INPUT':directory + 'InitialAlpha.tif','BAND':1,'FIELD':'DN','EIGHT_CONNECTEDNESS':False,'EXTRA':'','OUTPUT':directory + 'InitialAlphaExtent.gpkg'})
processing.run("native:fixgeometries", {'INPUT':directory + 'InitialAlphaExtent.gpkg','OUTPUT':directory + 'InitialAlphaExtentFix.gpkg'})
processing.run("native:extractbyexpression", {'INPUT':directory + 'InitialAlphaExtentFix.gpkg','EXPRESSION':' \"DN\" > 245','OUTPUT':directory + 'InitialAlphaExtentFilt.gpkg'})

#Find the intersection of the extents and buffer it in a touch
processing.run("native:clip", {'INPUT':directory + 'InitialAlphaExtentFilt.gpkg','OVERLAY':directory + 'TargetAlphaExtentFilt.gpkg','OUTPUT':directory + 'UnionExtent.gpkg'})
processing.run("native:buffer", {'INPUT':directory + 'UnionExtent.gpkg','DISTANCE':-1*pixelSizeResamp,'SEGMENTS':5,'END_CAP_STYLE':0,'JOIN_STYLE':0,'MITER_LIMIT':2,'DISSOLVE':False,'OUTPUT':directory + 'UnionExtentInBuff.gpkg'})

#Make a large amount of points for pixel value comparison
processing.run("native:randompointsinpolygons", {'INPUT':directory + 'UnionExtentInBuff.gpkg','POINTS_NUMBER':6000,'MIN_DISTANCE':0,'MIN_DISTANCE_GLOBAL':0,'MAX_TRIES_PER_POINT':10,'SEED':None,'INCLUDE_POLYGON_ATTRIBUTES':False,'OUTPUT':directory + 'RandomPoints.gpkg'})

#Sample the rasters so that the values can be compared
processing.run("native:rastersampling", {'INPUT':directory + 'RandomPoints.gpkg','RASTERCOPY':directory + 'TargetResample.tif','COLUMN_PREFIX':'PixelValTarget_','OUTPUT':directory + 'RandomPointsSampleTarget.gpkg'})
processing.run("native:rastersampling", {'INPUT':directory + 'RandomPointsSampleTarget.gpkg','RASTERCOPY':directory + 'InitialResample.tif','COLUMN_PREFIX':'PixelValInit_','OUTPUT':directory + 'RandomPointsSampleTargetInit.gpkg'})
processing.run("native:rastersampling", {'INPUT':directory + 'RandomPointsSampleTargetInit.gpkg','RASTERCOPY':directory + 'RelativeTogether.tif','COLUMN_PREFIX':'PixelValRel_','OUTPUT':directory + 'RandomPointsSampleTargetInitRel.gpkg'})


"""
#############################################################################
Multiple regession
"""

#Get the points from the gpkg
points = QgsVectorLayer(directory + 'RandomPointsSampleTargetInitRel.gpkg')
l = [i.attributes() for i in points.getFeatures()]

#Get the values from the attribute table and put them into lists
targetRed = [item[2] for item in l]
targetGreen = [item[3] for item in l]
targetBlue = [item[4] for item in l]
initialRed = [item[6] for item in l]
initialGreen = [item[7] for item in l]
initialBlue = [item[8] for item in l]
initRelRed = [item[10] for item in l]
initRelGreen = [item[11] for item in l]
initRelBlue = [item[12] for item in l]


#Determine the residual between the initial and the target raster, as a way of trend removal
residualRed = [b - a for a, b in zip(initialRed, targetRed)]
residualGreen = [b - a for a, b in zip(initialGreen, targetGreen)]
residualBlue = [b - a for a, b in zip(initialBlue, targetBlue)]

#Create 3rd order polynomal predictors for putting into the multiple regression
allPredictorsRed = list(zip(initialRed,[x**2 for x in initialRed],[x**3 for x in initialRed],initialGreen,[x**2 for x in initialGreen],[x**3 for x in initialGreen],initialBlue,[x**2 for x in initialBlue],[x**3 for x in initialBlue],initRelRed,initRelGreen,initRelBlue))
allPredictorsGreen = list(zip(initialRed,[x**2 for x in initialRed],[x**3 for x in initialRed],initialGreen,[x**2 for x in initialGreen],[x**3 for x in initialGreen],initialBlue,[x**2 for x in initialBlue],[x**3 for x in initialBlue],initRelRed,initRelGreen,initRelBlue))
allPredictorsBlue = list(zip(initialRed,[x**2 for x in initialRed],[x**3 for x in initialRed],initialGreen,[x**2 for x in initialGreen],[x**3 for x in initialGreen],initialBlue,[x**2 for x in initialBlue],[x**3 for x in initialBlue],initRelRed,initRelGreen,initRelBlue))

#Prep the model from sklearn
regrRed = linear_model.LinearRegression()
regrGreen = linear_model.LinearRegression()
regrBlue = linear_model.LinearRegression()

#Correlate the residual of red to the predictors
X = allPredictorsRed
y = residualRed
regrRed.fit(X, y)
print(regrRed.coef_)

#Then for green
X = allPredictorsGreen
y = residualGreen
regrGreen.fit(X, y)
print(regrGreen.coef_)

#Then for blue
X = allPredictorsBlue
y = residualBlue
regrBlue.fit(X, y)
print(regrBlue.coef_)



"""
##############################################################################
Prep tasks for multiprocessing (saves a bit of time)
"""

def redCalc(task):
    try:
        #Use the coefficients from the multiple regression to calculate the new values for the image
        processing.run("gdal:rastercalculator", {'INPUT_A':initialImage,'BAND_A':1,'INPUT_B':initialImage,'BAND_B':2,'INPUT_C':initialImage,'BAND_C':3,
        'INPUT_D':directory + 'RelativeTogether.tif','BAND_D':1,'INPUT_E':directory + 'RelativeTogether.tif','BAND_E':2,'INPUT_F':directory + 'RelativeTogether.tif','BAND_F':3,
        'FORMULA':'(((' +
        str(regrRed.coef_[0]) + ' * float64(A)) + (' + str(regrRed.coef_[1]) + ' * (float64(A)**2)) + (' + str(regrRed.coef_[2]) + ' * (float64(A)**3)) + (' + 
        str(regrRed.coef_[3]) + ' * float64(B)) + (' + str(regrRed.coef_[4]) + ' * (float64(B)**2)) + (' + str(regrRed.coef_[5]) + ' * (float64(B)**3)) + (' +
        str(regrRed.coef_[6]) + ' * float64(C)) + (' + str(regrRed.coef_[7]) + ' * (float64(C)**2)) + (' + str(regrRed.coef_[8]) + ' * (float64(C)**3)) + (' + 
        str(regrRed.coef_[9]) + ' * float64(D)) + (' + str(regrRed.coef_[10]) + ' * float64(E)) + (' + str(regrRed.coef_[11]) + ' * float64(F)) + ' + str(regrRed.intercept_) + ')*' + str(correctionAmount) + ') + float64(A)'
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat + '|DISCARD_LSB=14','EXTRA':'--overwrite','OUTPUT':directory + 'RedCalced.tif'})
    except BaseException as e:
        print(e)

def greenCalc(task):
    try:
        #Same for green, note the correction amount being applied
        processing.run("gdal:rastercalculator", {'INPUT_A':initialImage,'BAND_A':1,'INPUT_B':initialImage,'BAND_B':2,'INPUT_C':initialImage,'BAND_C':3,
        'INPUT_D':directory + 'RelativeTogether.tif','BAND_D':1,'INPUT_E':directory + 'RelativeTogether.tif','BAND_E':2,'INPUT_F':directory + 'RelativeTogether.tif','BAND_F':3,
        'FORMULA':'(((' +
        str(regrGreen.coef_[0]) + ' * float64(A)) + (' + str(regrGreen.coef_[1]) + ' * (float64(A)**2)) + (' + str(regrGreen.coef_[2]) + ' * (float64(A)**3)) + (' + 
        str(regrGreen.coef_[3]) + ' * float64(B)) + (' + str(regrGreen.coef_[4]) + ' * (float64(B)**2)) + (' + str(regrGreen.coef_[5]) + ' * (float64(B)**3)) + (' +
        str(regrGreen.coef_[6]) + ' * float64(C)) + (' + str(regrGreen.coef_[7]) + ' * (float64(C)**2)) + (' + str(regrGreen.coef_[8]) + ' * (float64(C)**3)) + (' + 
        str(regrGreen.coef_[9]) + ' * float64(D)) + (' + str(regrGreen.coef_[10]) + ' * float64(E)) + (' + str(regrGreen.coef_[11]) + ' * float64(F)) + ' + str(regrGreen.intercept_) + ')*' + str(correctionAmount) + ') + float64(B)'
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat + '|DISCARD_LSB=14','EXTRA':'--overwrite','OUTPUT':directory + 'GreenCalced.tif'})        
    except BaseException as e:
        print(e)

def blueCalc(task):
    try:
        #Same for blue, note the removal of 14 bits as 32 is excessive if there will be a later final 8 bit export
        processing.run("gdal:rastercalculator", {'INPUT_A':initialImage,'BAND_A':1,'INPUT_B':initialImage,'BAND_B':2,'INPUT_C':initialImage,'BAND_C':3,
        'INPUT_D':directory + 'RelativeTogether.tif','BAND_D':1,'INPUT_E':directory + 'RelativeTogether.tif','BAND_E':2,'INPUT_F':directory + 'RelativeTogether.tif','BAND_F':3,
        'FORMULA':'(((' +
        str(regrBlue.coef_[0]) + ' * float64(A)) + (' + str(regrBlue.coef_[1]) + ' * (float64(A)**2)) + (' + str(regrBlue.coef_[2]) + ' * (float64(A)**3)) + (' + 
        str(regrBlue.coef_[3]) + ' * float64(B)) + (' + str(regrBlue.coef_[4]) + ' * (float64(B)**2)) + (' + str(regrBlue.coef_[5]) + ' * (float64(B)**3)) + (' +
        str(regrBlue.coef_[6]) + ' * float64(C)) + (' + str(regrBlue.coef_[7]) + ' * (float64(C)**2)) + (' + str(regrBlue.coef_[8]) + ' * (float64(C)**3)) + (' + 
        str(regrBlue.coef_[9]) + ' * float64(D)) + (' + str(regrBlue.coef_[10]) + ' * float64(E)) + (' + str(regrBlue.coef_[11]) + ' * float64(F)) + ' + str(regrBlue.intercept_) + ')*' + str(correctionAmount) + ') + float64(C)'
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat + '|DISCARD_LSB=14','EXTRA':'--overwrite','OUTPUT':directory + 'BlueCalced.tif'})
    except BaseException as e:
        print(e)

"""
##############################################################################
Run the tasks in parallel and wait for them to finish
"""

#Start up the tasks
redTask = QgsTask.fromFunction('RedCalc', redCalc)
QgsApplication.taskManager().addTask(redTask)
greenTask = QgsTask.fromFunction('GreenCalc', greenCalc)
QgsApplication.taskManager().addTask(greenTask)
blueTask = QgsTask.fromFunction('BlueCalc', blueCalc)
QgsApplication.taskManager().addTask(blueTask)

#Wait til done
try:
    redTask.waitForFinished(timeout = 900000)
except BaseException as e:
    print(e)  
try:
    greenTask.waitForFinished(timeout = 900000)
except BaseException as e:
    print(e)  
try:
    blueTask.waitForFinished(timeout = 900000)
except BaseException as e:
    print(e)  

"""
##############################################################################
Final bringing together of bands
"""

#Grab the alpha band of the initial image ready for the vrt
processing.run("gdal:translate", {'INPUT':initialImage,'TARGET_CRS':None,'NODATA':None,'COPY_SUBDATASETS':False,'OPTIONS':compressOptionsFloat,'EXTRA':'-b 4','DATA_TYPE':6,'OUTPUT':directory + 'AlphaBand.tif'})

#Bring together the bands into a vrt
processing.run("gdal:buildvirtualraster", {'INPUT':[directory + 'RedCalced.tif',directory + 'GreenCalced.tif',directory + 'BlueCalced.tif',directory + 'AlphaBand.tif'],
'RESOLUTION':2,'SEPARATE':True,'PROJ_DIFFERENCE':True,'ADD_ALPHA':False,'ASSIGN_CRS':None,'RESAMPLING':0,'SRC_NODATA':'','EXTRA':'','OUTPUT':directory + 'Band123A.vrt'})

#Export this out to a tif
processing.run("gdal:warpreproject", {'INPUT':directory + 'Band123A.vrt','SOURCE_CRS':None,'TARGET_CRS':None,'RESAMPLING':0,'NODATA':None,'TARGET_RESOLUTION':None,'OPTIONS':compressOptionsFloat,'DATA_TYPE':0,'TARGET_EXTENT':None,
'TARGET_EXTENT_CRS':None,'MULTITHREADING':True,'EXTRA':'-co \"PHOTOMETRIC=RGB\" -srcalpha -dstalpha ' + gdalOptions,'OUTPUT':directory + initImName + 'Corrected.tif'})

#Build pyramids to view the final image more easily
processing.run("gdal:overviews", {'INPUT':directory + initImName + 'Corrected.tif','CLEAN':False,'LEVELS':'','RESAMPLING':0,'FORMAT':1,'EXTRA':''})

#Bring in the final product
layer1 = iface.addRasterLayer(directory + initImName + 'Corrected.tif', initImName + 'Corrected' , '')
#From here you may need to manually apply per-band min/max stretches, saturation adjustments and tinting to get it closer to the target look

