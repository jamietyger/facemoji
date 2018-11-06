"""Build_SVM

This script allows the user to build the SVM classifier required to
identify facial expression

This script requires that `pandas`,`sklearn` and `pickle` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * predict - returns the column headers of the file
    * main - the main function of the script
"""


import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold,train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import svm
import pickle




class BuildSVM():


    def build(self,path):
        """Builds Support Vector Machine

        Parameters
        ----------
        datafile : str
            Path of the csv

        Returns
        -------
        None
        """
        data = pd.read_csv(path, header=None) #Load Data File
        #data[0]=data[0].map({'angry':0, 'happy':1})
        y = data[0]  #Assign first column as Y values
        X = data.drop(0,axis=1) #Assign the rest as Features

        #Grid Search Parameters
        param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},]
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.33, stratify=y, random_state=42)

        # choose svm type: SVC - Support Vector Classification(based on libsvm)
        svc = svm.SVC(probability=True)

       
        print('Initiating Grid Search...')
        clf = GridSearchCV(svc, param_grid)

        print('Cross Validation...')
        skf = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
        scores = cross_val_score(clf, X, y, cv=skf, n_jobs=-1)
        print ("Mean score:", scores.mean()) 
        print("2 STDs:",scores.std()*2)

        filename = 'modelname'

        print('Training Model...')
        clf = clf.fit(X_train, y_train)
        pickle.dump(clf, open('./assets/'+filename, 'wb'))


        print ('Best score for classifier:', clf.best_score_)
        print ('Best C:', clf.best_estimator_.C)
        print ('Best Kernel:', clf.best_estimator_.kernel)
        print ('Best Gamma:', clf.best_estimator_.gamma)
        print ('SVM Best Estimator:', clf.best_estimator_)
        print ('SVM Grid Scores: \n', clf.cv_results_)


        y_pred = clf.predict(X_test)



        print("Accuracy:",accuracy_score(y_test,y_pred), normalize=True)

        print (classification_report(y_test, y_pred,target_names=['angry','happy','neutral','surprise']))

        print('Confusion Matrix:')
        confusion_matrix(y_test, y_pred, labels = [1,2,3,4])


if __name__ == '__main__': 

    builder = BuildSVM()     #Build SVM
    builder.build('./assets/data.csv')