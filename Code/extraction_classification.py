#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy as np
import math
import cv2
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from joblib import dump, load
import pickle

from tempfile import mkdtemp
savedir = mkdtemp()
import os


def from255To1(mask):  #makes the mask binary, it is originally made of 0 and 255
    (lineNumber,columnNumber)=mask.shape
    mask = (mask/np.max(mask)).astype('float')
    return mask

def compare(initialMask,transformedMask): #Compare two masks pixel by pixel, returns the number of pixels which are different between both
    (lineNumber,columnNumber)=initialMask.shape

    counter = np.sum(np.abs(initialMask - transformedMask )).astype(int)
    return counter#/area
##ASSYMETRIE

def centroid(thresh):  #returns the centroid of the nevus
    M = cv2.moments(thresh)

# calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return(cX,cY)

def distance(x1,y1,x2,y2):#retuns the distance between two points (x1,y1) and (x2,y2)
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def angle(xc,yc,x,y): #returns the angle between the segments [(xc,yc),(x,y)] and [(xc,yc),(x,yc)]
    cos=distance(x,yc,xc,yc)/distance(xc,yc,x,y)
    return math.acos(cos)

def maxDistancePoint(mask):  #returns the furthest point from the centroid of the nevus
    (xc,yc)=centroid(mask)
    (lineNumber,columnNumber)=mask.shape
    xMax=0
    yMax=0
    distanceMax=0
    dist=0.0

    for i in range(lineNumber):
        for j in range(columnNumber):
            if(mask[i][j]==1):
                dist=distance(xc,yc,i,j)
                if(dist>distanceMax):
                    xMax=j
                    yMax=i
                    distanceMax=dist
    return(xMax,yMax)



def baseCoord(mask): #returns the orthogonal base whose first vector is oriented according to the furthest point from the centroid
    (xc,yc)=centroid(mask)
    (x1,y1)=maxDistancePoint(mask)
    vector1=(x1-xc,y1-yc)
    xa=vector1[0]
    ya=vector1[1]
    x1=200
    vector2=(x1,-x1*(xa/ya))
    return(vector1,vector2)



def rotate_image(mat, angle,xc,yc):  #rotate the image arround the centroid
    height, width = mat.shape
    image_center = (xc, yc)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    rotated_mat = cv2.warpAffine(mat.astype(float), rotation_mat, (width,height) )
    return rotated_mat

def assymmetry(mask):  #compare the original mask and the one turned 90 degrees
    (xc,yc)=centroid(mask)
    rotatedMask=rotate_image(mask,180,xc,yc)
    return compare(mask,rotatedMask)

def mirror_principalAxis(mask): #returns the mirror-mask according to the principal axis (axis which represents the maximal variability)
    (xc,yc)=centroid(mask)
    
    # find the principal axes
    #find the coordinates of the ones in the mask
    y_inds, x_inds = np.nonzero(mask>0)
    y_inds = y_inds - yc
    x_inds = x_inds - xc
    y_carre = (y_inds**2).sum()
    x_carre = (x_inds**2).sum()
    y_x = (np.multiply(y_inds,x_inds)).sum()

    A = np.zeros((2,2))
    A[0,0] = y_carre
    A[0,1] = y_x
    A[1,0] = y_x
    A[1,1] = x_carre
    w,v = np.linalg.eig(A)
    
    if abs(w[0])>abs(w[1]):
        theta = angle(xc,yc,xc+v[1,0]*800,yc+v[0,0]*800)
    else:
        theta = angle(xc,yc,xc+v[1,1]*800,yc+v[0,1]*800)
    mask=rotate_image(mask,-theta*180/math.pi,xc,yc) 
    mask=np.flipud(mask)
    mask=rotate_image(mask,theta*180/math.pi,xc,yc)
   
    return mask

def applyMask(img,mask): #returns the segmented nevus according to the mask
    (x,y)=mask.shape
    for i in range (x):
        for j in range (y):
            if mask[i][j]==0:
                img[i][j]=[0,0,0]
    return img

#%%ABCD

def ABCD(imageName,maskName): #returns the different features of a nevus thanks to the adresses of the picture and the mask
    image=cv2.imread(imageName,1)
    mask = cv2.imread(maskName,0)

    (x,y)=mask.shape
    list=[0,0,0,0,0]

    #assymetry

    newDim = max(x,y)
    enlargedMask= np.zeros((2*newDim,2*newDim))
    enlargedMask[0:x,0:y] = mask
    enlargedMask = np.roll(enlargedMask,(newDim-x//2-1,newDim-y//2-1),(0,1))
    enlargedMask=from255To1(enlargedMask)
    list[0]=assymmetry(enlargedMask)
    
    list[1]=compare(enlargedMask,mirror_principalAxis(enlargedMask))

    #similarToEllipse
    ret,thresh = cv2.threshold(mask,127,255,0)
    im, contours,hierarchy = cv2.findContours(thresh, 1, 2) #contours,hierarchy
    cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)
    ellipseMask=np.zeros((x,y))
    cv2.ellipse(ellipseMask,ellipse,255,-1)
    list[2]=compare(from255To1(mask),from255To1(ellipseMask))

    #perimeter/area
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    list[3]=perimeter/area*1000

    #color
    binaryMask=from255To1(mask).astype(np.uint8)
    
    hist = (cv2.calcHist([image], [0], binaryMask, [256], [0, 255])+cv2.calcHist([image], [1], binaryMask, [256], [0, 255])+cv2.calcHist([image], [2], binaryMask, [256], [0, 255]))/3
   
    list[4]=np.var(hist)

    return np.array(list).reshape(1,-1)

def ABCDbis(imageName,mask): #returns the different features of a nevus thanks to the picture adress and the mask itself
    image=cv2.imread(imageName,1)

    (x,y)=mask.shape
    list=[0,0,0,0,0]

    #assymetry

    newDim = max(x,y)
    enlargedMask= np.zeros((2*newDim,2*newDim))
    enlargedMask[0:x,0:y] = mask
    enlargedMask = np.roll(enlargedMask,(newDim-x//2-1,newDim-y//2-1),(0,1))
    enlargedMask=from255To1(enlargedMask)
    list[0]=assymmetry(enlargedMask)
    
    list[1]=compare(enlargedMask,mirror_principalAxis(enlargedMask))

    #similarToEllipse
    ret,thresh = cv2.threshold(mask,127,255,0)
    im1, contours,hierarchy = cv2.findContours(thresh, 1, 2) #contours,hierarchy
    cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)
    ellipseMask=np.zeros((x,y))
    cv2.ellipse(ellipseMask,ellipse,255,-1)
    list[2]=compare(from255To1(mask),from255To1(ellipseMask))

    #perimeter/area
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    list[3]=perimeter/area*1000

    #colour
    binaryMask=from255To1(mask).astype(np.uint8)
    
    hist = (cv2.calcHist([image], [0], binaryMask, [256], [0, 255])+cv2.calcHist([image], [1], binaryMask, [256], [0, 255])+cv2.calcHist([image], [2], binaryMask, [256], [0, 255]))/3
   
    list[4]=np.var(hist)

    return np.array(list).reshape(1,-1)

#%%TEST CLASSIFICATION - PRE PROCESSING

def preProcessing(address): #returns the diagnostic of the pictures which are in the database

    df = pd.read_excel(address)
    df.as_matrix()
    # Getting the data from excel into a python data stream

    features1 = np.array(df)
    # Data inside a matrix --> ground truth about the nature of  spots

    features = []
    for k in range(len(features1)):
        if features1[k][2]==0:
            features.append(features1[k])
            #get rid of seborrheic keratosis lines

    features = np.array(features)
    features = features[::,:2]
    # 0 means benign, 1 means malignant
    return features

#%%TEST CLASSIFICATION - LDA
excelAddress = 'Ground_truth_ISIC_1.xlsx'
trainingSetLength = 5

diagnostic=preProcessing(excelAddress)
(length,columnNumber)=diagnostic.shape
extractedFeatures=np.zeros((length,5))

for i in range(trainingSetLength):#previous version range(lenght)

    imageAddress='../data/ISIC-2017_Training_Data/ISIC-2017_Training_Data/'+diagnostic[i,0]+'.jpg'
    maskAddress='../data/ISIC-2017_Training_Part1_GroundTruth/'+diagnostic[i,0]+'_segmentation.png'
    extractedFeatures[i]=ABCD(imageAddress,maskAddress)

y = diagnostic[:trainingSetLength,1:] # target values (i.e. expected output for X)

for i in range (len(y)):
    y[i]=int(y[i])
y=np.transpose(y).astype('int')
    
trainingSet=extractedFeatures[:trainingSetLength]
    
lda = LinearDiscriminantAnalysis()
    
lda.fit(trainingSet, y[0])
# letting the algorithm know which sample in X belongs to which class labelled in y
    
# save the params to disk
lda_params = lda.get_params()
params_lda = 'params_lda.sav'

# save the model to disk
filename_lda = 'lda_model.sav'
    
    
#testSet=extractedFeatures[trainingSetLength:trainingSetLength+10]
#prediction=lda.predict(testSet)

pickle.dump(lda, open(filename_lda, 'wb'))
pickle.dump(lda_params, open(params_lda, 'wb'))
    
#%%TEST CLASSIFICATION - QDA
excelAddress = 'Ground_truth_ISIC_1.xlsx'
trainingSetLength = 5

diagnostic=preProcessing(excelAddress)
(length,columnNumber)=diagnostic.shape
extractedFeatures=np.zeros((length,5))

for i in range(trainingSetLength):#previous version range(lenght)
    
    imageAddress='../data/ISIC-2017_Training_Data/ISIC-2017_Training_Data/'+diagnostic[i,0]+'.jpg'
    maskAddress='../data/ISIC-2017_Training_Part1_GroundTruth/'+diagnostic[i,0]+'_segmentation.png'
    extractedFeatures[i]=ABCD(imageAddress,maskAddress)

y = diagnostic[:trainingSetLength,1:] # target values (i.e. expected output for X)

for i in range (len(y)):
    y[i]=int(y[i])
y=np.transpose(y).astype('int')

trainingSet=extractedFeatures[:trainingSetLength]
qda = QuadraticDiscriminantAnalysis()

qda.fit(trainingSet, y[0])
# letting the algorithm know which sample in X belongs to which class labelled in y
        
# save the params to disk
qda_params = qda.get_params()
params_qda = 'params_qda.sav'
    
# save the model to disk
filename_qda = 'qda_model.sav'
pickle.dump(qda, open(filename_qda, 'wb'))
    
#testSet=extractedFeatures[trainingSetLength:trainingSetLength+10]
#prediction=qda.predict(testSet)

pickle.dump(qda_params, open(params_qda, 'wb'))
#%%TEST CLASSIFICATION - Naive Bayes
excelAddress = 'Ground_truth_ISIC_1.xlsx'
trainingSetLength = 5

diagnostic=preProcessing(excelAddress)
(length,columnNumber)=diagnostic.shape
extractedFeatures=np.zeros((length,5))

for i in range(trainingSetLength):#previous version range(lenght)

    imageAddress='../data/ISIC-2017_Training_Data/ISIC-2017_Training_Data/'+diagnostic[i,0]+'.jpg'
    maskAddress='../data/ISIC-2017_Training_Part1_GroundTruth/'+diagnostic[i,0]+'_segmentation.png'
    extractedFeatures[i]=ABCD(imageAddress,maskAddress)

y = diagnostic[:trainingSetLength,1:] # target values (i.e. expected output for X)

for i in range (len(y)):
    y[i]=int(y[i])
y=np.transpose(y).astype('int')

trainingSet=extractedFeatures[:trainingSetLength]
bys = GaussianNB()

bys.fit(trainingSet, y[0])
# letting the algorithm know which sample in X belongs to which class labelled in y
    
# save the params to disk
bys_params = bys.get_params()
params_bys = 'params_bys.sav'
    
# save the model to disk
filename_bys = 'bys_model.sav'
pickle.dump(bys, open(filename_bys, 'wb'))
    
#testSet=extractedFeatures[trainingSetLength:trainingSetLength+10]
#prediction=bys.predict(testSet)

pickle.dump(bys_params, open(params_bys, 'wb'))
#%%TEST CLASSIFICATION - kNN
excelAddress = 'Ground_truth_ISIC_1.xlsx'
trainingSetLength = 5

diagnostic=preProcessing(excelAddress)
(length,columnNumber)=diagnostic.shape
extractedFeatures=np.zeros((length,5))

for i in range(trainingSetLength):#previous version range(lenght)

    imageAddress='../data/ISIC-2017_Training_Data/ISIC-2017_Training_Data/'+diagnostic[i,0]+'.jpg'
    maskAddress='../data/ISIC-2017_Training_Part1_GroundTruth/'+diagnostic[i,0]+'_segmentation.png'
    extractedFeatures[i]=ABCD(imageAddress,maskAddress)

y = diagnostic[:trainingSetLength,1:] # target values (i.e. expected output for X)

for i in range (len(y)):
    y[i]=int(y[i])
y=np.transpose(y).astype('int')

trainingSet=extractedFeatures[:trainingSetLength]
neigh = KNeighborsClassifier(n_neighbors=1)

neigh.fit(trainingSet, y[0])
# letting the algorithm know which sample in X belongs to which class labelled in y
    
# save the params to disk
neigh_params = neigh.get_params()
params_neigh = 'params_neigh.sav'
    
# save the model to disk
filename_neigh = 'neigh_model.sav'
pickle.dump(neigh, open(filename_neigh, 'wb'))
    
#testSet=extractedFeatures[trainingSetLength:trainingSetLength+10]
#prediction=lda.predict(testSet)

pickle.dump(neigh_params, open(params_neigh, 'wb'))
#%%TEST CLASSIFICATION - Logistic Regression
excelAddress = 'Ground_truth_ISIC_1.xlsx'
trainingSetLength = 5

diagnostic=preProcessing(excelAddress)
(length,columnNumber)=diagnostic.shape
extractedFeatures=np.zeros((length,5))

for i in range(trainingSetLength):#previous version range(lenght)

    imageAddress='../data/ISIC-2017_Training_Data/ISIC-2017_Training_Data/'+diagnostic[i,0]+'.jpg'
    maskAddress='../data/ISIC-2017_Training_Part1_GroundTruth/'+diagnostic[i,0]+'_segmentation.png'
    extractedFeatures[i]=ABCD(imageAddress,maskAddress)

y = diagnostic[:trainingSetLength,1:] # target values (i.e. expected output for X)

for i in range (len(y)):
    y[i]=int(y[i])
y=np.transpose(y).astype('int')

trainingSet=extractedFeatures[:trainingSetLength]
lgr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

lgr.fit(trainingSet, y[0])
# letting the algorithm know which sample in X belongs to which class labelled in y
    
# save the params to disk
lgr_params = lgr.get_params()
params_lgr = 'params_lgr.sav'
    
# save the model to disk
filename_lgr = 'lgr_model.sav'
pickle.dump(lgr, open(filename_lgr, 'wb'))
    
#testSet=extractedFeatures[trainingSetLength:trainingSetLength+10]
#prediction=lda.predict(testSet)
   
pickle.dump(lgr_params, open(params_lgr, 'wb'))  

#%%TEST CLASSIFICATION - Diagnostic

def diagnostic(features): #returns the diagnostic (0 for a normal nevus and 1 for a melanoma) according to different classification methods
    diagnostics=[0,0,0,0]
    
    lda = pickle.load(open(filename_lda, 'rb'))
    paramsLda=pickle.load(open(params_lda,'rb'))
    lda.set_params(**paramsLda)
    diagnostics[0]=lda.predict(features).tolist()[0]
    
    qda = pickle.load(open(filename_qda, 'rb'))
    paramsQda=pickle.load(open(params_qda,'rb'))
    qda.set_params(**paramsQda)
    diagnostics[1]=qda.predict(features).tolist()[0]
    
    bys = pickle.load(open(filename_bys, 'rb'))
    paramsBys=pickle.load(open(params_bys,'rb'))
    bys.set_params(**paramsBys)
    diagnostics[2]=bys.predict(features).tolist()[0]
    
    lgr = pickle.load(open(filename_lgr, 'rb'))
    paramsLr=pickle.load(open(params_lgr,'rb'))
    lgr.set_params(**paramsLr)
    diagnostics[3]=lgr.predict(features).tolist()[0]
    
    return diagnostics


address = 'Ground_truth_ISIC_1.xlsx'
#print(linearDiscriminantAnalysis(address,10))


#%% How to find current working directory (where pickle save files are stored)

import os 
dir_path = os.path.dirname(os.path.realpath(filename_lda))


