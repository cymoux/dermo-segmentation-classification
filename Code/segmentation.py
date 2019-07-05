#!/usr/bin/env python
# -*- coding: utf-8 -*-



import matplotlib.pyplot as plt
import numpy as np
import cv2
from math import sqrt
import copy
import scipy.spatial
import math


def ColorSpaceTransformation(img_in):
    #We keep this transformation for the moment : the extraction of the blue channel 
    img_BaW = img_in[:,:,0]
    
    return img_BaW


#return a blurred image using a parameter for the blur amount
def noiseRemoval(img):
    blur = 5
    kernel = np.ones((blur, blur), np.float32) / (blur**2)
    dst = cv2.filter2D(img, -1, kernel)

    return(dst)

#Map the values of the intensity from the current min and max to 0 and 255 with a certain percentage of saturation
def intensityAdjust(imgBaW):

    minOut = 0
    maxOut = 255

    imgFlat = imgBaW.flatten()
    imgFlat.sort()

    # percentage is a number of piwel to saturate atthe low and high ends
    percentage = 50
    # the current max and min values
    minIn = imgFlat[percentage]
    maxIn = imgFlat[imgFlat.shape[0] - percentage]

    G = (imgBaW-minIn)
    scalaire = (maxOut-minOut)/(maxIn - minIn + 0.01)
    G = G*scalaire + minOut
    G = G.astype(int)
    G[imgBaW>maxIn] = 255
    G[imgBaW<minIn] = 0
    return G

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

#This function is useful to treat blacks frames
def circle(im, r):
    (h, l) = im.shape[0], im.shape[1]

    mask = create_circular_mask(h, l, center=None, radius=300)
    mask = np.logical_not(mask)

    img2 = im* mask
    nb = np.count_nonzero(mask)
    s = np.sum(np.sum(img2))

    return s/nb

    
#This function is useful to treat blacks frames
def rayon(im):
  
    L = len(im[0])
    l = len(im)

    rmin = l/4
    rmax = (1/2)*sqrt(l**2 + L**2)
    nb_pas = 10
    pas = (rmax - rmin) / nb_pas
    
    Lrayon = []
    Lmoyenne = []
    
    r = rmin 
    for i in range (nb_pas) : 
        Lrayon.append(r)
        Lmoyenne.append(circle(im,r))
        r += pas
    return [Lrayon, Lmoyenne]
    
#This function is useful to treat blacks frames
def minMoyenne(image):

    Lrayon, Lmoyenne = rayon(image)
    
    Mmin = 256
    indice = -1
    
    for i in range (len(Lrayon)):
        if Lmoyenne[i] < Mmin : 
            Mmin = Lmoyenne[i]
            indice = i     
    return Lrayon[indice - 4]

#This function is useful to treat blacks frames
def masqueBinaire(im):
    rayon = minMoyenne(im)
    
    m,n = im.shape
    m2 = int(np.floor(float(m)/ 2.0))
    n2 = int(np.floor(float(n)/ 2.0))
    mask = copy.copy(im)
        
    x, y = np.meshgrid(range( -n2, n2+np.mod(n,2)), range(-m2,m2+np.mod(m,2)))
    mask = (( np.power(x,2) + np.power(y,2) ) < rayon**2).astype(int)    
    return mask
    
#Otsu implementation
def otsu(im):
    
    mask = copy.copy(im)
    
    mask = masqueBinaire(im)
    
    pxl = im[mask>0]
    pxl_hist,_ = np.histogram(pxl,range(0,256))
    
    totalSigma = np.zeros(256)         
    for k in range (0,256): 
        omega_1 = pxl_hist[0:k].sum().astype(float)
        omega_2 = pxl_hist[k:-1].sum().astype(float)
        mu_1 = np.mean(pxl[pxl<=k])
        if (np.isnan(mu_1)):
            mu_1 = 0
        mu_2 = np.mean(pxl[pxl>k])
        if (np.isnan(mu_2)):
            mu_2 = 0
        totalSigma[k] = omega_1 * omega_2 * ( (mu_1 - mu_2)**2)
        
    seuil = np.argmax(totalSigma)

    mask_out = np.logical_and( mask>0 , im < seuil).astype(int)

    return mask_out

#the connectedComponent function takes a two dimensionnal numpy array corresponding to a segmentation mask
def connectedComponentsRemoval(imgArg):
    #calculation of the connected components using openCV
    img = np.array(imgArg, dtype=np.uint8)

    #connected components data
    (retval, comp, stats, centroid) = cv2.connectedComponentsWithStats(img)

    # we get the connected components areas
    areas = (stats[:, 4])

    maskOut = np.copy(imgArg)
    maskOut = np.array(maskOut, dtype=np.uint8)

    #we sort the areas to get the the 2 maximum areas (the Area Of Iterrest and the rest of the Skin)
    sortedAreas = np.sort(areas)

    #if there is only 2 zones:, the mask is ok
    if (retval <= 2):
        return (imgArg)
    else:
        #for each connected component, if the area is inferior to the second bigge, we fill it with black
        for i in range(0, retval):
            if (areas[i] < sortedAreas[-2]):
                maskOut[comp == i] = 0

    return(maskOut)

def MorphologicalFilling (im):
    kernel = np.ones((45,45), np.uint8)
    closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    return closing

# This algorithm takes in parameter an image of beauty spot and returns the mask of the hair
def hairMask(im):
    M_colors = []
    
    for color in range(3): # The closing is made on the three channels (red, green, blue) to detect hair
        channel = im[:,:,color] 

        # The closing is done with a rectangular kernel
        SE = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)) # --> Tester sur : vertical, puis un peu diagonal, puis beaucoup plus, jusqu'à horizontal
        
        # Let's obtain thin, a gray-scale image which show the thin elements of the initial image
        thin = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, SE) - channel
        
        # temp is the matrix of the same size as thin. It is actually  the mask of the hair through the used channel
        temp = np.zeros_like(thin) 
        temp[thin > 15] = 1 
        
        M_colors.append(temp)
    
    # Let's finally obtain the mask, which is the union of the three masks calculated above
    mask = np.logical_or(M_colors[0], M_colors[1], M_colors[2]) 
    
    # Finally, closedMask is the image with less discontinuity
    SE4 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    closedMask = cv2.dilate(mask, SE4, 2) 
    
    return closedMask

# This algorithm takes in parameter the mask of hairMask and removes the noise. It removes the elements with few neighbours
def hairDetection(mask): 
    
    #Let's find all the connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    #the following line removes the background of the image. It is actually a component that won't be useful for the rest of the algorithm
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # min_size is the minimum size of particles that we want to keep (that is to say, the number of pixels)
    min_size = 180

    #img2 will become the mask without noise
    img2 = np.zeros((output.shape))

    #for every component in the image, let's keep it only if there are more than min_size elements in it.
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2

#takes an image RGB array and a mask corresponding to the hair
#Aims at removing the hair by inpainting them with interpolation
#here : we replace the pixel by the value interpolated from 2 pixels on the sides or above and below
# the distance of the piwels is a parameter
def hairInterpolation1(origin, detect):
    (h,l)= detect.shape

    new = origin.copy()

    #the indices of the hair points
    indices = np.nonzero(detect)

    nbPoints = (indices[0].shape[0])

    #the distance of the 2 points of interpolation
    cSquare = 30
    #going through all the hair points
    for i in range (0, nbPoints):
        x = indices[0][i]
        y = indices[1][i]

        # A and B are the 2 points used to interpolate
        neighbourhood = False
        #a loop to test if the interpolation points are valid (to be improved)
        while  (not neighbourhood):
            xA,yA = min (int(x /cSquare)*cSquare,  h-1), y
            xB,yB = min(xA + cSquare, h-1), y

            if (  detect[xA][yA]  or detect[xB][yB] ):
                xA, yA = x, min ( int( (y /cSquare)*cSquare), l-1 )
                xB, yB = x, min((yA + cSquare), l-1)
            neighbourhood = True

        if all(x >= 0 for x in (xA, yA, xB, yB)) and all(x < h for x in (xA, xB)) and all(x < l for x in (yA, yB)):
            for k in range (0,3):
                #we interpolate
                u = ((x-xA + y-yA)/float(cSquare))
                new[x][y][k] = int(origin[xA][yA][k]*u + (1-u)*origin[xB][yB] [k]  )

    #plt.show()
    return (new)

#For each hair mask pixel, we calculate the direction of the belonging hair, using linear regression
#and replace the pixel by interpolating it with two pixels chosen on a perpendicular direction
def hairInterpolationNormalMethod(origin, detect):
    (h, l) = detect.shape
    #detect = np.logical_not(detect)

    new = origin.copy()
    indices = np.nonzero(detect)
    nbPoints = (indices[0].shape[0])
    #size of the zone where the linreg is computed
    zoneSize = 5
    #half of the distance between the interpolation pixels
    interpolationMaxSize = 20

    #we take the skeleton of the  mask in order to calculate the directions of the hair
    skeleton = skimage.morphology.skeletonize(detect)

    for i in range (0,int(nbPoints/1)):
        x = indices[0][i]
        y = indices[1][i]
        #print (str(i) + "/" + str(nbPoints))

        #limits of the zone of linear regression
        left_size = min (zoneSize, y)
        right_size = min (zoneSize, l-y)
        up_size = min(zoneSize, x)
        down_size = min(zoneSize, h-x)
        subArraySkeleton = skeleton[x-up_size:x+down_size, y-left_size:y+right_size]
        subBones = np.nonzero(subArraySkeleton)
        nBones = subBones[0].shape[0]

        if (nBones != 0):
            #linear regression
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(subBones[0], subBones[1])
            if (  not math.isnan(slope) ):
                normalX, normalY = (slope/float(np.sqrt(slope**2 + 1))), (-1/(float(np.sqrt(slope**2 + 1))))

                xA = (x + int(normalX * interpolationMaxSize))
                yA = y + int(normalY * interpolationMaxSize)

                xB = (x - int(normalX * interpolationMaxSize))
                yB = (y - int(normalY * interpolationMaxSize) )

                #we interpolate linearly
                if all(x >= 0 for x in (xA, yA,xB,yB) ) and all(x < h for x in (xA, xB)) and all(x < l for x in (yA, yB)) :
                    u = (np.sqrt((xA-x)**2 + (yA-y)**2))/float(np.sqrt(  (xA-xB)**2 + (yA-yB)**2) + 0.01 )
                    for k in range(0, 3):
                        new[x][y][k] = int(new[xA][yA][k] * u + (1 - u) * new[xB][yB][k])

            #else:
                #print("no slope")
        #else :
            #print("no skeleton around the point")

    return(new)

def main(src):
    img = cv2.imread(src)
    imMask = hairMask(img)
    imDetection = hairDetection(imMask)
    imInterpolation = hairInterpolation1(img, imDetection)
    imgBaW = ColorSpaceTransformation(imInterpolation)
    imNoNoise = noiseRemoval(imgBaW)
    imContrast = intensityAdjust(imNoNoise)
    imgBinaire = otsu(imContrast)
    imgFiltree = connectedComponentsRemoval(imgBinaire)
    imgFinale = MorphologicalFilling(imgFiltree)
    return imgFinale