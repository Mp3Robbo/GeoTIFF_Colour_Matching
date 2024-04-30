import numpy
import time
from sklearn import linear_model
from datetime import datetime

"""
##############################################################################
User options
"""

#Where the processing folder will be made
rootDirectory =         'C:/Temp/'

#The image that needs correction
initialImage =          'C:/Temp/TintedImage.tif'

#The image that has the desired look
targetImage =           'C:/Temp/TargetImage.tif'

#Amount of correction to apply, 1.0 can be a bit extreme, 0.5 is a nice value to start with
correctionAmount =      0.5

#Tif export options
compressOptions =       'COMPRESS=ZSTD|PREDICTOR=1|ZSTD_LEVEL=1|NUM_THREADS=ALL_CPUS|BIGTIFF=IF_SAFER|TILED=YES'
compressOptionsFloat =  'COMPRESS=ZSTD|PREDICTOR=1|ZSTD_LEVEL=1|NUM_THREADS=ALL_CPUS|BIGTIFF=IF_SAFER|TILED=YES|DISCARD_LSB=14'
gdalOptions =           '--config GDAL_NUM_THREADS ALL_CPUS -overwrite'


"""
##############################################################################
##############################################################################
Initial prep
"""

#Set up the layer name for the raster calculations
initImName = initialImage.split("/")
initImName = initImName[-1]
initImName = initImName[:len(initImName)-4]

#Make the processing directory
directory = rootDirectory + initImName + '/'
try:
    os.mkdir(directory)
except BaseException as e:
    print(e)
    

#Determine the pixel pixel sizes for later processing
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
Image resampling
"""

def initialResampling(task):
    #Resample to the larger pixel size so that any parallax-type issues are mitigated
    processing.run("gdal:translate", {'INPUT':targetImage,'TARGET_CRS':None,'NODATA':None,'COPY_SUBDATASETS':False,'OPTIONS':compressOptions,'EXTRA':'-tr ' + str(pixelSizeResamp) + ' ' + str(pixelSizeResamp) + ' -r cubic','DATA_TYPE':0,'OUTPUT':directory + 'TargetResample.tif'})
    processing.run("gdal:translate", {'INPUT':initialImage,'TARGET_CRS':None,'NODATA':None,'COPY_SUBDATASETS':False,'OPTIONS':compressOptions,'EXTRA':'-tr ' + str(pixelSizeResamp) + ' ' + str(pixelSizeResamp) + ' -r cubic','DATA_TYPE':0,'OUTPUT':directory + 'InitialResample.tif'})


"""
##############################################################################
Relative colour
"""

def relativeRedCalc(task):
    #Relative red
    processing.run("gdal:rastercalculator", {'INPUT_A':initialImage,'BAND_A':1,'INPUT_B':initialImage,'BAND_B':2,'INPUT_C':initialImage,'BAND_C':3,
            'FORMULA':'float64(A) - ((float64(B)+float64(C))/2)'
            ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat,'EXTRA':'--overwrite','OUTPUT':directory + 'RelativeRed.tif'})

def relativeGreenCalc(task):     
    #Relative green
    processing.run("gdal:rastercalculator", {'INPUT_A':initialImage,'BAND_A':1,'INPUT_B':initialImage,'BAND_B':2,'INPUT_C':initialImage,'BAND_C':3,
            'FORMULA':'float64(B) - ((float64(A)+float64(C))/2)'
            ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat,'EXTRA':'--overwrite','OUTPUT':directory + 'RelativeGreen.tif'})
            
def relativeBlueCalc(task):
    #Relative blue        
    processing.run("gdal:rastercalculator", {'INPUT_A':initialImage,'BAND_A':1,'INPUT_B':initialImage,'BAND_B':2,'INPUT_C':initialImage,'BAND_C':3,
            'FORMULA':'float64(C) - ((float64(A)+float64(B))/2)'
            ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat,'EXTRA':'--overwrite','OUTPUT':directory + 'RelativeBlue.tif'})



"""
##############################################################################
Run the tasks in parallel and wait for them to finish
"""

#Start up the tasks
initialResamplingTask = QgsTask.fromFunction('initialResampling', initialResampling)
QgsApplication.taskManager().addTask(initialResamplingTask)

relativeRedTask = QgsTask.fromFunction('relativeRedCalc', relativeRedCalc)
QgsApplication.taskManager().addTask(relativeRedTask)
relativeGreenTask = QgsTask.fromFunction('relativeGreenCalc', relativeGreenCalc)
QgsApplication.taskManager().addTask(relativeGreenTask)
relativeBlueTask = QgsTask.fromFunction('relativeBlueCalc', relativeBlueCalc)
QgsApplication.taskManager().addTask(relativeBlueTask)


#Wait til done
try:
    initialResamplingTask.waitForFinished(timeout = 3000000)
except BaseException as e:
    print(e) 
try:
    relativeRedTask.waitForFinished(timeout = 3000000)
except BaseException as e:
    print(e)  
try:
    relativeGreenTask.waitForFinished(timeout = 3000000)
except BaseException as e:
    print(e)  
try:
    relativeBlueTask.waitForFinished(timeout = 3000000)
except BaseException as e:
    print(e)


"""
##############################################################################
Joining the relative colour bands together
"""


#Bring the bands together
processing.run("gdal:buildvirtualraster", {'INPUT':[directory + 'RelativeRed.tif',directory + 'RelativeGreen.tif',directory + 'RelativeBlue.tif'],
'RESOLUTION':2,'SEPARATE':True,'PROJ_DIFFERENCE':True,'ADD_ALPHA':False,'ASSIGN_CRS':None,'RESAMPLING':0,'SRC_NODATA':'','EXTRA':'','OUTPUT':directory + 'RelativeVirtual.vrt'})

#Export this out to a tif
processing.run("gdal:warpreproject", {'INPUT':directory + 'RelativeVirtual.vrt','SOURCE_CRS':None,'TARGET_CRS':None,'RESAMPLING':0,'NODATA':None,'TARGET_RESOLUTION':None,'OPTIONS':compressOptionsFloat,'DATA_TYPE':0,'TARGET_EXTENT':None,
'TARGET_EXTENT_CRS':None,'MULTITHREADING':True,'EXTRA':'-co \"PHOTOMETRIC=RGB\" ' + gdalOptions,'OUTPUT':directory + 'RelativeTogether.tif'})

"""
##############################################################################
Image prep and sampling
"""


#Get the extent of the target raster
processing.run("gdal:translate", {'INPUT':targetImage,'TARGET_CRS':None,'NODATA':None,'COPY_SUBDATASETS':False,'OPTIONS':compressOptions,'EXTRA':'-b 4 -scale_1 128 255 -1000 1255','DATA_TYPE':0,'OUTPUT':directory + 'TargetAlpha.tif'})
processing.run("gdal:polygonize", {'INPUT':directory + 'TargetAlpha.tif','BAND':1,'FIELD':'DN','EIGHT_CONNECTEDNESS':False,'EXTRA':'','OUTPUT':directory + 'TargetAlphaExtent.gpkg'})
processing.run("native:fixgeometries", {'INPUT':directory + 'TargetAlphaExtent.gpkg','OUTPUT':directory + 'TargetAlphaExtentFix.gpkg'})
processing.run("native:extractbyexpression", {'INPUT':directory + 'TargetAlphaExtentFix.gpkg','EXPRESSION':' \"DN\" > 245','OUTPUT':directory + 'TargetAlphaExtentFixFilt.gpkg'})

#Get the extent of the initial raster
processing.run("gdal:translate", {'INPUT':initialImage,'TARGET_CRS':None,'NODATA':None,'COPY_SUBDATASETS':False,'OPTIONS':compressOptions,'EXTRA':'-b 4 -scale_1 128 255 -1000 1255','DATA_TYPE':0,'OUTPUT':directory + 'InitialAlpha.tif'})
processing.run("gdal:polygonize", {'INPUT':directory + 'InitialAlpha.tif','BAND':1,'FIELD':'DN','EIGHT_CONNECTEDNESS':False,'EXTRA':'','OUTPUT':directory + 'InitialAlphaExtent.gpkg'})
processing.run("native:fixgeometries", {'INPUT':directory + 'InitialAlphaExtent.gpkg','OUTPUT':directory + 'InitialAlphaExtentFix.gpkg'})
processing.run("native:extractbyexpression", {'INPUT':directory + 'InitialAlphaExtentFix.gpkg','EXPRESSION':' \"DN\" > 245','OUTPUT':directory + 'InitialAlphaExtentFixFilt.gpkg'})

#Find the intersection of the extents and buffer it in a touch
processing.run("native:clip", {'INPUT':directory + 'InitialAlphaExtentFixFilt.gpkg','OVERLAY':directory + 'TargetAlphaExtentFixFilt.gpkg','OUTPUT':directory + 'UnionExtent.gpkg'})
processing.run("native:buffer", {'INPUT':directory + 'UnionExtent.gpkg','DISTANCE':-1*pixelSizeResamp,'SEGMENTS':5,'END_CAP_STYLE':0,'JOIN_STYLE':0,'MITER_LIMIT':2,'DISSOLVE':True,'OUTPUT':directory + 'UnionExtentInBuff.gpkg'})

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

#Because the regression didn't determine what the new values should be, but rather what difference needs to be applied to the original values,
#the below raster calcs add the difference that has been determined to be applied to the original image
#Despite this, there will still be significant regression to the mean in the output image

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
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat,'EXTRA':'--overwrite','OUTPUT':directory + 'RedCalced.tif'})
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
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat,'EXTRA':'--overwrite','OUTPUT':directory + 'GreenCalced.tif'})        
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
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat,'EXTRA':'--overwrite','OUTPUT':directory + 'BlueCalced.tif'})
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
    redTask.waitForFinished(timeout = 3000000)
except BaseException as e:
    print(e)  
try:
    greenTask.waitForFinished(timeout = 3000000)
except BaseException as e:
    print(e)  
try:
    blueTask.waitForFinished(timeout = 3000000)
except BaseException as e:
    print(e)  


"""
##############################################################################
Stretching the values so that they're closer to 0-255
"""

#Bring together the bands into a temp vrt
processing.run("gdal:buildvirtualraster", {'INPUT':[directory + 'RedCalced.tif',directory + 'GreenCalced.tif',directory + 'BlueCalced.tif'],
'RESOLUTION':2,'SEPARATE':True,'PROJ_DIFFERENCE':True,'ADD_ALPHA':False,'ASSIGN_CRS':None,'RESAMPLING':0,'SRC_NODATA':'','EXTRA':'','OUTPUT':directory + 'CalcedBandsTogether.vrt'})

#Ensure that the extent is not separate polygons
processing.run("native:dissolve", {'INPUT':directory + 'InitialAlphaExtentFixFilt.gpkg','FIELD':[],'OUTPUT':directory + 'InitialAlphaExtentFixFiltDissolve.gpkg'})

#Make random points in the whole extent of the input image
processing.run("native:randompointsinpolygons", {'INPUT':directory + 'InitialAlphaExtentFixFiltDissolve.gpkg','POINTS_NUMBER':6000,'MIN_DISTANCE':0,
'MIN_DISTANCE_GLOBAL':0,'MAX_TRIES_PER_POINT':10,'SEED':None,'INCLUDE_POLYGON_ATTRIBUTES':False,'OUTPUT':directory + 'RandomPointsFull.gpkg'})

#Find the rgb values so far
processing.run("native:rastersampling", {'INPUT':directory + 'RandomPointsFull.gpkg','RASTERCOPY':directory + 'CalcedBandsTogether.vrt',
'COLUMN_PREFIX':'PixelValCalced_','OUTPUT':directory + 'RandomPointsFullSampleCalced.gpkg'})

#Bring them in as variables
calcedPoints = QgsVectorLayer(directory + 'RandomPointsFullSampleCalced.gpkg')
calcedAttribs = [i.attributes() for i in calcedPoints.getFeatures()]
calcedRed = [item[2] for item in calcedAttribs]
calcedGreen = [item[3] for item in calcedAttribs]
calcedBlue = [item[4] for item in calcedAttribs]

#Find the min and max rgb values
maxRed = numpy.percentile(calcedRed, 99.9)
maxGreen = numpy.percentile(calcedGreen, 99.9)
maxBlue = numpy.percentile(calcedBlue, 99.9)
minRed = numpy.percentile(calcedRed, 0.1)
minGreen = numpy.percentile(calcedGreen, 0.1)
minBlue = numpy.percentile(calcedBlue, 0.1)

#See how far we can stretch without clipping
totalMax = max(maxRed,maxGreen,maxBlue)
totalMin = min(minRed,minGreen,minBlue)

#Apply the stretch
processing.run("gdal:rastercalculator", {'INPUT_A':directory + 'RedCalced.tif','BAND_A':1,
        'FORMULA':'(A - ' + str(totalMin) + ') * ' + str(255/(totalMax-totalMin))
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat,'EXTRA':'--overwrite','OUTPUT':directory + 'RedScaled.tif'})
        
processing.run("gdal:rastercalculator", {'INPUT_A':directory + 'GreenCalced.tif','BAND_A':1,
        'FORMULA':'(A - ' + str(totalMin) + ') * ' + str(255/(totalMax-totalMin))
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat,'EXTRA':'--overwrite','OUTPUT':directory + 'GreenScaled.tif'})
        
processing.run("gdal:rastercalculator", {'INPUT_A':directory + 'BlueCalced.tif','BAND_A':1,
        'FORMULA':'(A - ' + str(totalMin) + ') * ' + str(255/(totalMax-totalMin))
        ,'NO_DATA':None,'RTYPE':5,'OPTIONS':compressOptionsFloat,'EXTRA':'--overwrite','OUTPUT':directory + 'BlueScaled.tif'})


"""
##############################################################################
Final bringing together of bands
"""


#Grab the alpha band of the initial image ready for the vrt
processing.run("gdal:translate", {'INPUT':initialImage,'TARGET_CRS':None,'NODATA':None,'COPY_SUBDATASETS':False,'OPTIONS':compressOptionsFloat,'EXTRA':'-b 4','DATA_TYPE':6,'OUTPUT':directory + 'AlphaBand.tif'})

#Bring together the bands into a vrt
processing.run("gdal:buildvirtualraster", {'INPUT':[directory + 'RedScaled.tif',directory + 'GreenScaled.tif',directory + 'BlueScaled.tif',directory + 'AlphaBand.tif'],
'RESOLUTION':2,'SEPARATE':True,'PROJ_DIFFERENCE':True,'ADD_ALPHA':False,'ASSIGN_CRS':None,'RESAMPLING':0,'SRC_NODATA':'','EXTRA':'','OUTPUT':directory + 'Band123A.vrt'})

#Export this out to a tif
processing.run("gdal:warpreproject", {'INPUT':directory + 'Band123A.vrt','SOURCE_CRS':None,'TARGET_CRS':None,'RESAMPLING':0,'NODATA':None,'TARGET_RESOLUTION':None,'OPTIONS':compressOptions,'DATA_TYPE':1,'TARGET_EXTENT':None,
'TARGET_EXTENT_CRS':None,'MULTITHREADING':True,'EXTRA':'-co \"PHOTOMETRIC=RGB\" -srcalpha -dstalpha ' + gdalOptions,'OUTPUT':directory + initImName + 'Corrected.tif'})

#Build pyramids to view the final image more easily
processing.run("gdal:overviews", {'INPUT':directory + initImName + 'Corrected.tif','CLEAN':False,'LEVELS':'','RESAMPLING':0,'FORMAT':1,'EXTRA':'--config COMPRESS_OVERVIEW JPEG'})

#Bring in the final product
layer1 = iface.addRasterLayer(directory + initImName + 'Corrected.tif', initImName + 'Corrected' , '')
#From here you may need to manually apply per-band min/max stretches, saturation adjustments and tinting to get it closer to the target look


"""
##############################################################################
Bring the layer into the project
"""

provider=layer1.dataProvider()
statsRed = provider.cumulativeCut(1,0.001,0.999,sampleSize=1000) #adjust these values depending on the stretch you want
statsGreen = provider.cumulativeCut(2,0.001,0.999,sampleSize=1000)
statsBlue = provider.cumulativeCut(3,0.001,0.999,sampleSize=1000)
try:
    del(min)
except:
    print('')
try:
    del(max)
except:
    print('')
minimum = min(statsRed[0],statsGreen[0],statsBlue[0])#find the lowest val
maximum = max(statsRed[1],statsGreen[1],statsBlue[1])#find the highest val
renderer=layer1.renderer()
myType = renderer.dataType(1)
myEnhancement = QgsContrastEnhancement(myType)
Renderer = QgsMultiBandColorRenderer(provider,1,1,1) 
contrast_enhancement = QgsContrastEnhancement.StretchToMinimumMaximum
myEnhancement.setContrastEnhancementAlgorithm(contrast_enhancement,True)
myEnhancement.setMinimumValue(minimum)#where the minimum value goes in
myEnhancement.setMaximumValue(maximum)
layer1.setRenderer(Renderer)
layer1.renderer().setRedBand(1)#band 1 is red
layer1.renderer().setGreenBand(2)
layer1.renderer().setBlueBand(3)
layer1.renderer().setRedContrastEnhancement(myEnhancement)#the same contrast enhancement is applied to all
layer1.renderer().setGreenContrastEnhancement(myEnhancement)
layer1.renderer().setBlueContrastEnhancement(myEnhancement)
layer1.triggerRepaint() #refresh
