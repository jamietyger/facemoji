"""Facemoji

This script allows the user to replace a face in an image with an
emoji representing the facial expression.

This tool accepts image files (.jpg, .png)

This script requires that `cv2` OpenCV be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * predict - Predicts the facial expression and applies emoji
    * main - the main function of the script
"""


import pickle
from project.feature_extraction import FeatureExtraction
import cv2
import sys


class classify():
    """Finds faces in images

    Parameters
    ----------
    image : str
        The file location of the image
    Returns
    -------
    image
        a image with the faces replaced with emojis
    """   
    def predict(self, image):
        # Load emoji images
        happy = cv2.imread('./project/assets/emoji/happy.png',-1) 
        angry = cv2.imread('./project/assets/emoji/angry.png',-1)
        neutral = cv2.imread('./project/assets/emoji/neutral.png',-1)
        suprised = cv2.imread('./project/assets/emoji/suprised.png',-1)

        #Load model
        loaded_model = pickle.load(open('./project/assets/modelname', 'rb'))

        self = FeatureExtraction()   
        faces = self.viola_jones(image) #Apply Viola Jones - Haar Cascade to find faces
        fds = self.calc_hogs(faces)   #Calculate HOG features for faces
        i=0 #Face Counter
        for face in faces:     #For every face found
            [x,y,w,h] = face['roi']    #ROI - bounding details
            #cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2) #show found faces on output

            result =loaded_model.predict(fds[i].reshape(1,-1))  #Make prediction based on HOG feature

            #result_proba = loaded_model.predict_proba(fds[i].reshape(1,-1))
            #print (result)
            #print (result_proba)

            # Assign Emoji to prediction    
            if result[0]=='neutral':
                emoji = neutral
            elif result[0]=='happy':
                emoji = happy
            elif result[0]=='angry':
                emoji = angry
            elif result[0]=='suprise':
                emoji = suprised
            emoji = cv2.resize(emoji, (w, h)) #Resize Emoji to ROI

            y1, y2 = y, y+emoji.shape[0]
            x1, x2 = x, x+emoji.shape[1]

            #Adjust alpha Channel to make background of png transparent
            alpha_s = emoji[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            #Write Emoji to image
            for c in range(0, 3):
                image[y1:y2, x1:x2, c] = (alpha_s * emoji[:, :, c] +alpha_l * image[y1:y2, x1:x2, c])
            i+=1 #Increment Face counter

            return image

if __name__ == '__main__': 
    filename = sys.argv[1] #Get File Name from Args
    image = cv2.imread(filename)
    classifier = classify()     #Call Classifier
    classifier.predict(image)   
    cv2.imshow('Result',image)  #Display Result
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit() 