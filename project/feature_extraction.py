"""Feature Extraction

This script allows the user to identify faces in an image using haarcascades and calcualte
Histogram of Oriented Gradients feature vector

This script requires that `cv2` (OpenCV),`matplotlib` and `sklearn` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * viola_jones - Uses haar cascades to identify faces in an image
    * calc_hog - calcualte HOG feature vector for region of interest
    * calc_hogs - helper function to handle multiple faces
    * main - the main function of the script
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import img_as_float



class FeatureExtraction():


    def viola_jones(self,image):
        """Runs Viola-Jones Algorirthm to detect faces

        Parameters
        ----------
        image : str
            Path of the image

        Returns
        -------
        faces : []
            Array Dict of the regions of interest found
        """

        height = 56
        width = 56
        scaled = np.zeros((height, width), dtype=np.float)
        #output = image.copy() #Make copy of input image to display as output
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #make grayscale version of image
        face_cascade = cv2.CascadeClassifier('./project/assets/haarcascade_frontalface_default.xml') #load cascade classifier
        extract_faces = face_cascade.detectMultiScale(gray, 1.3, 5) #extract faces
        #print (str(len(extract_faces))+" Faces Found")
        faces = [] 
        for (x,y,w,h) in extract_faces: #extract ROI
            face = {}
            #cv2.rectangle(output,(x,y),(x+w,y+h),(255,255,0),2) #show found faces on output

            face['roi']=[x,y,w,h] 
            roi = image[y:y + h, x:x + w] #ROI bounds
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            scaled = cv2.resize(roi_gray, (height, width))
            face['scaled'] = scaled
            faces.append(face)              #store in faces dict
  
        return faces

    def calc_hog(self,image):
        """Runs Histogram of Oriented Gradients Algorithm to generate feature
            vector of ROI

        Parameters
        ----------
        image : str
            Path of the image

        Returns
        -------
        fd : []
            Returns coefficients of the classifier
        """       


        image = img_as_float(image)  # convert unit8 tofloat64 ... dtype
        orientations = 9 #number of orientation bins
        cellSize = (4, 4)  # size of cell (pixels)
        blockSize = (3, 3)  # cells_per_block
        blockNorm = 'L1-sqrt'  # {'L1', 'L1-sqrt', 'L2', 'L2-Hys'}
        visualize = True  # Also return an image of the HOG.
        transformSqrt = False
        featureVector = True
        fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=cellSize,
                    cells_per_block=blockSize, block_norm = blockNorm, visualize = visualize, transform_sqrt=transformSqrt, feature_vector=featureVector)
            #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))  
            # Rescale histogram for better display
        return fd

    def calc_hogs(self, faces):
        """Helper Function for HOG - handles multiple faces

        Parameters
        ----------
        faces : []
            Array of regions of interest found

        Returns
        -------
        fds : []
            Returns coefficients of the classifier
        """    


        fds=[] 
        for face in faces: #for every face found
            fd =self.calc_hog(face['scaled']) #calc hog on roi
            fds.append(fd)
        return fds

    def main(self):
        image = cv2.imread('./project/assets/harry.jpg')
        faces = self.viola_jones(image)
        fds = self.calc_hogs(faces)
        #print(fds[1].shape)
        #cv2.imshow('highlight face',image)
        #cv2.waitKey(0)



if __name__ == '__main__': FeatureExtraction().main()