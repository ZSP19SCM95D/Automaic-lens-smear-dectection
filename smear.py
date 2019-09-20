import cv2
import os
import numpy as np
import sys
import glob
from random import randint

def data_read_process(path):
    # initialize the parameter for the mean of all pixel values of all images
    mean = np.zeros((500,500,3),np.float)
    
    # save all the image list in the list
    data= glob.glob(path+"/*.jpg")
    print("the total image number is "+ str(len(data)))
    
    for image in data:
        # read every image data in the file folder and resize them to the same size as mean pic
        imagec = cv2.imread(image)
        imagec_temp = cv2.resize(imagec,(500,500))
        # smooth the image using the median filter
        imagec_blur = cv2.medianBlur(imagec_temp,5)
        imagec_mat = np.array(imagec_blur,dtype=np.float)
        mean += imagec_mat】
    
    # Find mean value for all images by pixel
    mean = mean/len(data)
    cv2.imwrite("Mean_"+path.split('/')[1]+".jpg", mean)
    
#     mean = np.array(np.round(mean),dtype=np.uint8)
    mean = np.array(np.round(mean),dtype=np.uint8)
    return mean,data
    
def masksaving(path,mean,data):

    # convert mean image to grayscale image
    gray = cv2.cvtColor(mean, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('gray',gray)
    cv2.waitKey(0)

    # apply threshold function 
#     thres = cv2.threshold(path, dst, 177, 200, cv.THRESH_BINARY)
    
    thres = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 85, 11)
    
    cv2.imshow('threshold image',thres)
    cv2.waitKey(0)
    
    #Inverts every bit of an array and create the mask so the smear area is in white
    mask = cv2.bitwise_not(thres)

    cv2.imwrite("Mask_"+path.split('/')[1]+".jpg",mask)
    
    # detect the smear on random image
    Ram = data[randint(0,len(data))]
    readImage = cv2.imread(Ram)
    Ram_resize = cv2.resize(readImage,(500,500))
    cv2.imshow('Apply Mask', Ram_resize)
    cv2.waitKey(0)
    
    return mask,Ram_resize
    

def LensSmearDetection(path,mask,Ram_resize):
    
#**
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # locate the contour on the random image
        if( cv2.contourArea(contours[0])<maxArea and cv2.contourArea(contours[0])>minArea):
            # draw contours
            smear_con = cv2.drawContours(Ram_resize,contours,-1,(0,255,255),2)

            cv2.imwrite("final_"+path.split('/')[1]+".jpg",Ram_resize)
            cv2.imshow("smear_con_Result",smear_con)
            cv2.waitKey(0)
            return True
    return False


if __name__ == "__main__":
    
    #set the minimum and maximum radius
    R_min = 10
    R_max = 12
    minArea = 3.14 * R_min**2
    maxArea = 3.14 * R_max**2
    
    #detect if the directory is valid or not
    args = sys.argv[1:]
    if not args[0]:
        sys.exit()
    print(" The Directory of source picture has been found. \n The smear dectection is in progress...")
    
    # process the data and find mean value for all images by pixel and write the image.
    mean,data = data_read_process(args[0])
    
    # save mask to the disk and detect smear on random images
    mask,Ram_resize = masksaving(args[0],mean,data)
    
    if(LensSmearDetection(args[0],mask,Ram_resize)):
        print ("Smear has been detected for "+args[0]+" source.")
    else:
        print("No Smear in "+ args[0])
