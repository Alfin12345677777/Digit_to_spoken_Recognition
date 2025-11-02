import cv2
import numpy as np        
import matplotlib.pyplot as plt
from scipy import ndimage
import math
from keras.models import load_model
import pyttsx3

# Voice digit settings 
engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

engine.setProperty('rate', 100)

# loading pre trained model
model = load_model('Final_Model.h5')

def predict_digit(img):
    test_image = img.reshape(-1,28,28,1)
    return np.argmax(model.predict(test_image))


#pitting label
def put_label(t_img,label,x,y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    l_x = int(x) - 10
    l_y = int(y) + 10
    cv2.rectangle(t_img,(l_x,l_y+5),(l_x+35,l_y-35),(0,255,0),-1) 
    cv2.putText(t_img,str(label),(l_x,l_y), font,1.5,(255,0,0),1,cv2.LINE_AA)
    return t_img

# refining each digit
#The image_refiner function takes a grayscale image as input and resizes it to a fixed size of 28x28 pixels. 
#It also adds padding to the image to ensure that it has the same size as the input image.

#This function is useful for preprocessing images before feeding them into a machine learning model.
def image_refiner(gray):
    org_size = 22
    img_size = 28
    rows,cols = gray.shape
    
    if rows > cols:
        factor = org_size/rows
        rows = org_size
        cols = int(round(cols*factor))        
    else:
        factor = org_size/cols
        cols = org_size
        rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))
    
    #get padding 
    colsPadding = (int(math.ceil((img_size-cols)/2.0)),int(math.floor((img_size-cols)/2.0)))
    rowsPadding = (int(math.ceil((img_size-rows)/2.0)),int(math.floor((img_size-rows)/2.0)))
    
    #apply apdding 
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    return gray


def get_output_image(path):
    # The function get_output_image takes an image path as input and reads the image using OpenCV's imread function.
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception("Could not read image at path: " + path)
    
    #The image is converted to grayscale and thresholded to obtain a binary image.

    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    if ret is None:
        raise Exception("Could not threshold image")

    # Contours are detected in the binary image using findContours.
    if cv2.__version__.startswith('3.'):
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        im2 = thresh

    for j, cnt in enumerate(contours):
        # Calculate the convex hull
        hull = cv2.convexHull(cnt)
        if hull is None:
            raise Exception("Could not calculate convex hull")

        # Check if the contour is convex
        k = cv2.isContourConvex(cnt)

        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue

        # Check if the contour is the outermost contour
        if hierarchy[0][j][3] != -1:
            # Draw a rectangle around the digit
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the digit
            roi = img[y:y + h, x:x + w]

            # Process the digit
            roi = cv2.bitwise_not(roi)
            roi = image_refiner(roi)

            # Threshold the digit
            th, fnl = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

            # Predict the digit
            pred = predict_digit(fnl)
            print(pred)

            digit_to_text(pred)

            # Place the label on the digit
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            img = put_label(img, pred, int(x), int(y))

    return img

def digit_to_text(pred):
    digit_map = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine"
    }
    
    text = digit_map.get(pred, "")
    if text:
        engine.say(text)
        engine.runAndWait()