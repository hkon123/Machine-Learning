## Step 1 - `Weather` class
#
# Begin by importing the Weather class.
# Weather data is accessed through an instantation of the provided `Weather` class.

from Weather import *

## Step 2 - Load Weather data
#
# Create a new object (e.g. `weather`) of type `Weather`.
#
# As part of the instantiation provide:
# - weatherFile: The filename (either basic.txt or advanced.txt)
# - fileSlice: The number of lines to read from the chosen input file (0 is all)
#
# Use the fileSlice to limit the sample size for early evaluation

weatherFile = 'data/basic.txt'
fileSlice = 0
weather = Weather(weatherFile, fileSlice)

## Step 3 - Inspect the Data
'''
print '#'*50
print '# Step 3: Inspect the Data'
print '#'*50
print '\n'

# print data
print 'Weather Data:'
print weather.data

# print number of entries
print 'Number of entries: %s' % (weather.getNrEntries())

# print target names
print 'Number of targets: %s' % (weather.getNrTargets())

print 'Target names: %s' % (weather.getTargetNames())

# print features
print 'Number of features: %s' % (weather.getNrFeatures())

print 'Feature names: %s' % (weather.getFeatures())

# uncomment below to print station data
# print 'Number of weather stations: %s' % (weather.getNrStations())
# print 'Stations (ID, Name, Latitude, Longitude)'
# print weather.getStationData('all')

# Edinburgh and Shap station details
print 'Station data for EDINBURGH/GOGARBANK: %s' % (weather.getStationData('EDINBURGH/GOGARBANK'))
print 'Station data for ID 3225: %s' % (weather.getStationData('3225'))

# get data from one feature
print 'Temperature data: %s' % (weather.getFeatureData('Temperature'))
'''
## Step 4 - Recovering Incomplete Data
#
# Some of the observation values have a value of `-99999`
# This is a default value I inserted to indicate that the feature data was either not collected
# at the time of the observation or had a null value.
#
# Any data points that contain null observations need to be corrected to avoid problems
# with subsequent filtering and modifications.
#
# In some cases null values can either be interpolated or set to a default value.
#
# The large majority of null data is from the `Gust` measurement.
# Here I assume than no observation is the same as a zero value

# zero any null gust measurements
newG = ['0' if g == '-99999' else g for g in weather.getFeatureData('Gust')]
weather.modify('Gust', newG)
newPt = ['F' if g == '-99999' else g for g in weather.getFeatureData('Pressure Trend')]
weather.modify('Pressure Trend', newPt)
#print weather.getFeatureData('Gust')

## Step 5 - Removing Incomplete Data

# After recovering any data ensure you run the `discard()` method to
# remove any data with remaining null observations.

weather.discard()

## Step 6 - Data Conversion

# Some of the features have observation values that will be difficult for a machine learning estimator
# to interpret correctly (e.g. Wind Direction).

# You should ensure that all the features selected for a machine learning classification have a numeric value.

# In example 1 the pressure trend is changed from Falling, Static, Rising to 0,1,2
# In example 2 the Wind Direction is changed to a 16 point index starting from direction NNE.

# **Important**: Due to the limitations with the `Weather` class ensure that any observation data
# remains type `string` (e.g store '1' **not** 1).
# The `export()` method will convert all the values from `string` to `float` just before the export.

# Example 1 - Enumerate Pressure Trend (-1 falling, 0 static, 1 rising)

# define types
pTType = ['F', 'S', 'R']

# generate new pressure trend values
newPT = [str(pTType.index(p) - 1) for p in weather.getFeatureData('Pressure Trend') ]

# modify dataset
weather.modify('Pressure Trend', newPT)

#print 'Pressure Trend: %s' % (weather.getFeatureData('Pressure Trend'))

# Example 2 - Enumerate Wind direction (use 16 point compass index)

# define types
compassRose = ['NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']

# generate and modify Wind direction
weather.modify('Wind Direction', [compassRose.index(w) for w in weather.getFeatureData('Wind Direction')])

#print 'Wind Direction: %s' % (weather.getFeatureData('Wind Direction'))

## Step 7 - Data Extraction

# The `getObservations()` method will enable you to filter the available data by
# Station ID, date, time and a selected feature.

# This may be helpful if you want to build an additional input feature for classification based
# on contextual information

# The example below retrieves the temperature and dew point for Edinburgh for 24th October
'''
print '\n'
print '#'*50
print '# Step 7: Data Extraction'
print '#'*50
print '\n'

stationId = weather.getStationData('EDINBURGH/GOGARBANK')
features = ['Time since midnight', 'Temperature', 'Dew Point']

print 'Temperature and Dew Point measurements for Edinburgh 24th October'
print '(Time since midnight (min), Temperature, Dew Point)'
print weather.getObservations('3166', obsDate='2017-10-24', features=features)
'''
# This can then be combined with location data
# Here, the Pressure, Pressure Trend and Wind direction from the nearest weather station
# 100km NW of Edinburgh for 24th October is shown
# (uncomment section)

# stationId = weather.getStationData('EDINBURGH/GOGARBANK')
#
# # get nearest stations 100k NW of Edinburgh station within a 75km threshold
# nearestStations = weather.findStations([stationId[2], stationId[3]], ['100', '-45'], maxThreshold=75)
#
# print '\n'
# print 'Nearest stations 100km NW of EDINBURGH/GOGARBANK'
# for s in nearestStations:
#     print s
#
# # use nearest station (index 0 )
# nearStationId = nearestStations[0]
#
# # get observations from nearest station on 24/10
# obsDate='2017-10-24'
# print '\n'
# print 'Using station %s on %s' % (nearStationId[1], obsDate)
# features = ['Time since midnight', 'Pressure', 'Pressure Trend', 'Wind Direction']
# print '(Time since midnight (min), Pressure, Pressure Trend, Wind Direction)'
# print weather.getObservations(nearStationId[0], obsDate=obsDate, features=features)

## Step 8 - Add new features

# You may get better insights into underlying patterns in the observations
# by extacting the provided data to generate new features.
#
# An example using the wind direction is shown below.
# The direction *relative* to the North and to the West is generated and appended to the dataset.

# set relative direction (assume 16 points)
points = 16
north = [abs((points / 2) - int(w))%(points / 2) for w in weather.getFeatureData('Wind Direction')]
west = [abs((points / 2) - int(w) - (points / 4))%(points / 2) for w in weather.getFeatureData('Wind Direction')]

# append to dataset
weather.append('Wind Relative North', north)
weather.append('Wind Relative West', west)

## Step 9 - Select features

# To finish create an array of strings containing a subset of the features
# you feel will perform best in the classification. Call the select() method to filter the data

features = ['Temperature',\
 'Visibilty',\
  'Pressure',\
   'Pressure Trend',\
    'Humidity',\
     'Dew Point',\
      'Wind Direction',\
       'Latitude',\
        'Longitude',\
        #'Time since midnight',\
        ]
weather.setSelectedFeatures(features)

weather.select(features)

## Step 10 - Export data

# Run the `export()` method to write the data of your selected features to file as a `pickle` object.

# This will move the target data ('Weather Type') into a new variable (`target`).

# **Note**: It is assumed that the *Station ID*  and *Station Name* will not be used
# as features for classification and are automatically stripped. The `export()` method
# will also strip out incomplete data before exporting to file.

weather.export('data/mldata.p')
#weather.export('data/mldata1.p')
#weather.export('data/mldata2.p')
