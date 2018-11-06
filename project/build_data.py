"""Build_Data

This script allows the user to build the csv data file which will be loaded in to SVM

This script requires that `cv` (OpenCV),`csv` and `os` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * build_data - builds the datafile
    * main - the main function of the script
"""
import os
import cv2
import csv
from feature_extraction import FeatureExtraction

class BuildData():
    def build_data(self, path):

        """Builds the Datafile

        Parameters
        ----------
        path : str
            The location of the images

        Returns
        -------
        None
        """
        rootdir = path
        for subdir, dirs, files in os.walk(rootdir): # for every sub directory of path 
            for file in files:   #for every file
                filepath = subdir+os.sep+file 
                if filepath.endswith('.png'):
                    image = cv2.imread(filepath) #read image
                    self = FeatureExtraction() 
                    face = self.viola_jones(image) #find faces
                    fd = self.calc_hogs(face) #calculate hog desc
                    print (filepath) #print filepath for image
                    if len(fd) != 0: #If face found
                        with open('./assets/data.csv', mode ='a') as csv_file:
                            writer = csv.writer(csv_file)
                            data = [subdir.replace(rootdir+os.sep,'')] #Write Y and x
                            fs=fd[0]
                            for feature in fs:
                                data.append(feature)
                            writer.writerow(data)
        print("COMPLETE DATA BUILD")

if __name__ == '__main__': 
    builder = BuildData()
    builder.build_data('./assets/dataset')