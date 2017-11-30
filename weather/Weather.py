
import math
import pickle
import datetime, time
import numpy as np

class Weather:

    def __init__(self, wfile, lines=0):
        """ Weather Constructor
        Args:
            wfile (str): input weather file
            lines (int, optional): number of lines of the input file to read
        """

        self.wfile = wfile
        self.lines = lines

        # load data from file
        isLoaded = self._load()
        if (isLoaded == -1):
            print 'Error reading weather data'
            return -1

        # add data descriptions
        self._setTargetNames()
        self._setFeatureNames()
        self._setStationData()

        # set member data defaults
        self.obsStart = [2017,10,23]

        return

    ## GET METHODS

    def getNrEntries(self):
        """ Get number of weather observations read from file
        Returns:
            Number of weather observations
        """
        return len(self.data)

    def getTargetNames(self):
        """ Get target names
        Returns:
            Target names
        """
        return self.targetNames

    def getNrTargets(self):
        """ Get number of targets

        Returns:
            Number of targets
        """
        return self.targetNames.size

    def getFeatures(self):
        """ Get feature names
        Returns:
            Feature names
        """
        return self.featureNames

    def getNrFeatures(self):
        """ Get number of features
        Returns:
            Number of features
        """
        return self.featureNames.size

    def getFeatureData(self, feature):
        """ Get data for chosen feature
        Args:
            feature (str): selected feature
        Returns:
            Observation data of the selected feature (list)
        """
        return self.data[:,self._getFIdx(feature)]

    def getStationData(self, stationId):
        """ Get data for chosen station
        Args:
            stationId (str): selected station
        Returns:
            Observation data of the selected station (list)
        """
        if (stationId == 'all'):
            return self.stationData
        else:
            station = np.where(self.stationData == stationId)[0][0]
            return self.stationData[station]

    def getNrStations(self):
        """ Get number of observation stations
        Returns:
            Number of observation stations
        """
        return len(self.stationData)

    ## DATA MANIPULATION METHODS

    def modify(self, feature, newValues):
        """ Replace the data of a chosen feature with a given list of new values
        Args:
            feature (str): selected feature
            newValues (list(str)): New set of values to overwrite old data
        Returns:
            0 for success
        """
        self.data[:,self._getFIdx(feature)] = newValues
        return 0

    def append(self, featureName, featureData):
        """ Append the data with a new feature and list of new values
        Args:
            featureName (str): name of new feature to add
            featureData (list(str)): New set of values to add
        Returns:
            0 for success
        """
        self.data = np.concatenate((self.data, np.array([featureData]).T), axis=1)
        self.featureNames = np.append(self.featureNames, featureName)
        return 0

    def select(self, features):
        """ Select a set of features to retain (and remove other features)
        Args:
            features (list(str)): Selected set of features
        Returns:
            0 for success
        """
        if 'Weather Type' not in features:
            features.append('Weather Type')
        self.data = self.data[:,[self._getFIdx(f) for f in features]]
        self.featureNames = self.featureNames[[self._getFIdx(f) for f in features]]
        return 0

    def discard(self):
        """ Discard observations with null data
        Returns:
            0 for success
        """
        for f in self.featureNames:
            self.data = self.data[self.data[:,self._getFIdx(f)] != '-99999']
        return

    def delete(self, feature):
        """ Delete a feature and assoicated data
        Args:
            feature (str): name of feature to delete
        Returns:
            0 for success
        """
        if (self._isFIdx(feature)):
            self.data = np.delete(self.data, self._getFIdx(feature), axis=1)
            self.featureNames = np.delete(self.featureNames, self._getFIdx(feature))
        return 0

    def export(self, fname):
        """ Export object to pickle file
        Args:
            fname (str): export file name
        Returns:
            0 for success
        """

        # discard any data with null feature values
        self.discard()

        # set target as last column
        self.target = self.getFeatureData('Weather Type')

        # remove non-exportable features
        for n in ['Station ID', 'Station Name', 'Date', 'Weather Type']:
            if self._isFIdx(n):
                self.delete(n)

        # convert all data to float
        self.data = self.data.astype(float)

        # export to file
        pickle.dump(self, open(fname, 'wb'))

        return 0

    ## STATS UTILITIES

    def getObservations(self, stationId='', obsDate='', obsTime='', features=[]):
        """ Provide observation data for a chosen feature filtered by station, date, time
        Args:
            stationId (str): Station ID
            obsDate (str): Observation date
            obsTime (str): Observation time
            features (list(str)): List of chosen features
        Returns:
            stats (list): List of observation data
        """

        # filter data
        stats = self.data
        if (stationId):
            stats = stats[stats[:,self._getFIdx('Station ID')] == stationId]
        if (obsDate):
            stats = stats[stats[:,self._getFIdx('Date')] == obsDate]
        if (obsTime):
            stats = stats[stats[:,self._getFIdx('Time since midnight')] == obsTime]

        # return features
        if (features):
            features = [self._getFIdx(f) for f in features]
            return stats[:,features]
        else:
            return stats

    def findStations(self, coords=[], offset=[], minThreshold=10, maxThreshold=100):
        """ Find the nearet observation station to a given location
        Args:
            coords (list(str1, str2)): Latitude and Longitude of location
            offset (list(str1, str2)): Magnitude (km) and Direction (deg) offset to apply to location
            minThreshold (int): Minimum acceptable distance from chosen location
            maxThreshold (int): Maximum acceptable distance from chosen location
        Returns:
            stations (list): List of nearby stations
        """

        nearStations = []

        # check for supplied Latitude and Longitude
        if not (coords[0] and coords[1]):
            return 0

        # calculate new coords with offset
        if (offset):
            if not (offset[0] and offset[1]):
                return 0
            coords = self._getNewCoords(coords, offset)

        # iterate through weather stations
        for s in self.stationData:

            # get distance between point and station
            distance = self._getDistance([float(coords[0]), float(coords[1])], \
                [float(s[2]), float(s[3])] )

            # add if within threshold
            if ((distance > minThreshold) and (distance < maxThreshold)):
                nearStations.append([s[0], s[1], s[2], s[3], distance])

        return sorted(nearStations, key=lambda x: (x[4]))

    def setRelTime(self):
        """ Define new feature to track observation time relative to start of sample
        Returns:
            0 if success
        """
        obsRelTime = [self._getRelTime(o) for o in self.data]
        self.append('Relative Time', obsRelTime)
        return 0


   ## PRIVATE SET METHODS

    def _setTargetNames(self):
        """ Set target names based on data stream
        Returns:
            0 if success
        """

        # full target names
        if (self.dataStream == 0):
            self.targetNames = np.array(['Clear Night', 'Sunny Day', 'Partly cloudy (night)', 'Partly cloudy (day)',\
             'Not used', 'Mist', 'Fog', 'Cloudy', 'Overcast', 'Light rain shower (night)', \
             'Light rain shower (day)', 'Drizzle', 'Light rain', 'Heavy rain shower (night)', \
             'Heavy rain shower (day)', 'Heavy rain', 'Sleet shower (night)', 'Sleet shower (day)', \
             'Sleet', 'Hail shower (night)', 'Hail shower (day)', 'Hail', 'Light snow shower (night)', \
             'Light snow shower (day)', 'Light snow', 'Heavy snow shower (night)', 'Heavy snow shower (day)', \
             'Heavy snow', 'Thunder shower', 'Thunder shower (night)', 'Thunder'])

        # main target names
        elif (self.dataStream == 1):
            self.targetNames =  np.array(['Clear', 'Partly Cloudy', 'Mist', 'Fog', 'Cloudy', \
             'Overcast', 'Rain', 'Sleet', 'Hail', 'Snow', 'Thunder'])

        # basic target names
        elif (self.dataStream == 2):
            self.targetNames = np.array(['Clear', 'Cloudy', 'Precipitation'])

        return 0

    def _setFeatureNames(self):
        """ Set feature names
        Returns:
            0 if success
        """
        self.featureNames = np.array(['Station ID', 'Station Name', 'Elevation', 'Latitude', 'Longitude', 'Date', \
            'Time since midnight', 'Gust', 'Temperature', 'Visibilty', 'Wind Direction', \
            'Wind Speed', 'Pressure', 'Pressure Trend', 'Dew Point', 'Humidity', 'Weather Type'])
        return 0

    def _setStationData(self):
        """ Set station data
        LIMITATION:
          Old version of numpy on desktop PCs which does not accept axis \
          argument in np.unique(). Use workaround to reduce array
        Returns:
            0 if success
        """

        self.stationData = self.data[:,[self._getFIdx(f) for f in \
             'Station ID', 'Station Name', 'Latitude', 'Longitude']]
        # self.stationData = np.unique(self.stationData, axis=0)
        self.stationData = self._unique_rows(self.stationData)

        return 0

    ## PRIVATE DATA MANIPULATION METHODS

    def _load(self):
        """ Load data from file
        Returns:
            0 if success, -1 if file cannot be read
        """

        # number of non-data header details at top of data file
        header = 1

        # open file
        weatherData = []
        with open(self.wfile) as myfile:
            if (self.lines > 0):
                weatherData = [next(myfile) for x in xrange(self.lines + header)]
            else:
                weatherData = myfile.readlines()

        # get data stream from first line
        streamHeader = weatherData.pop(0).rstrip()
        if (streamHeader == 'FULL'):
            self.dataStream = 0
        elif (streamHeader == 'ADVANCED'):
            self.dataStream = 1
        elif (streamHeader == 'BASIC'):
            self.dataStream = 2
        else:
            print "Error: unecognised data stream from file %s" % (self.wfile)
            return -1

        # read data
        inputData = []
        for line in weatherData:
             entries = line.split()
             inputData.append(entries)

        # copy all into np array
        self.data = np.array(inputData)

        return 0

    def _getFIdx(self, featureName):
        """ Get Feature Index in data numpy array
        Args:
            featureName (str): Name of feature
        Returns:
            index
        """
        return np.where(self.featureNames == featureName)[0][0]

    def _isFIdx(self, featureName):
        """ Look up if feature name is indexed in data numpy array
        Args:
            featureName (str): Name of feature
        Returns:
            1 if success, 0 if not found
        """
        return 1 if (featureName in self.featureNames) else 0

    ## PRIVATE STATS UTILITIES

    def _getDistance(self, source, dest):
        """ Get the distance as crow flies between two coordinates
        Args:
            source (float): Longitude and Latitude of source point
            source (float): Longitude and Latitude of destination point
        Returns:
            distance (float): distance betwen points
        """

        lat1 = source[0]
        lat2 = dest[0]
        lon1 = source[1]
        lon2 = dest[1]

        # Formula from https://www.movable-type.co.uk/scripts/latlong.html
        R = 6370000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        deltaPhi = math.radians(lat2-lat1)
        deltalmb = math.radians(lon2-lon1)
        a = math.sin(deltaPhi/2) * math.sin(deltaPhi/2) + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(deltalmb/2) * math.sin(deltalmb/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a));
        d = (R * c)/1000.

        return d

    def _getNewCoords(self, coords, offset):
        """ Calculate new coordinates after applying offset
        Args:
            coords (list(str1, str2)): Latitude and Longitude of location
            offset (list(str1, str2)): Magnitude (km) and Direction (deg) offset to apply to location
        BUG?:
            direction seems to be opposite from what I expect, made correction of 360-x
        LIMITATION:
            Due E (or W) gives slightly different results for latitude (e.g. 50N over 200km is 49.96N)
        Returns:
            coords (list(float, float)): New coordinates
        """

        oldlat = math.radians(float(coords[0]))
        oldlon = math.radians(float(coords[1]))
        magnitude = float(offset[0]) / 6370.
        direction = math.radians(360.-float(offset[1]))

        # Calculate lat/lon given radial and distnace (http://www.edwilliams.org/avform.htm#LL)
        lat = math.asin(math.sin(oldlat) * math.cos(magnitude) + math.cos(oldlat) \
            * math.sin(magnitude) * math.cos(direction))
        lon = (oldlon - math.asin(math.sin(direction) * math.sin(magnitude) / math.cos(lat)) \
            + math.pi) % (2 * math.pi) - math.pi

        # print coords, offset, oldlat, oldlon, magnitude, direction, math.degrees(lat), math.degrees(lon)
        return (math.degrees(lat), math.degrees(lon))

    # Workaround on earlier numpy versions from https://github.com/numpy/numpy/issues/2871
    def _unique_rows(self, A, return_index=False, return_inverse=False):
        """
        Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
        where B is the unique rows of A and I and J satisfy
        A = B[J,:] and B = A[I,:]

        Returns I if return_index is True
        Returns J if return_inverse is True
        """
        A = np.require(A, requirements='C')
        assert A.ndim == 2, "array must be 2-dim'l"

        B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
                   return_index=return_index,
                   return_inverse=return_inverse)

        if return_index or return_inverse:
            return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
                + B[1:]
        else:
            return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')

    def _getRelTime(self, obsData):
        """ Calculate the time relative to set sample start time for a given data point
        Args:
            obsData (list): Observation data for single time point
        Returns:
            relTime (str): Time relative to set sample start time (hours)
        """

        # get unix time for start of data sample (midnight) as ref point
        dt = datetime.datetime(self.obsStart[0], self.obsStart[1], self.obsStart[2], 0, 0)
        startOfDay = int(time.mktime(dt.timetuple()))

        # strip date string
        dateString = [x.strip() for x in obsData[self._getFIdx('Date')].split('-')]

        # get unix time for start of observation date
        obsStartOfDay = int(time.mktime(datetime.datetime( \
            int(dateString[0]), int(dateString[1]), int(dateString[2]), 0, 0).timetuple()))

        # calculate relative time (hours)
        relTime = int((obsStartOfDay + (int(obsData[self._getFIdx('Time since midnight')])*60) \
            - startOfDay)/3600.)

        return str(relTime)

    def setSelectedFeatures(self,features):
        self.selectedFeatures = features
        print(self.selectedFeatures)

    def getSelectedFeatures(self):
        #del self.selectedFeatures[-1]
        return self.selectedFeatures[:-1]
