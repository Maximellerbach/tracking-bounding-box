import collections
import math
import time
import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import Birch

#####################################

def is_crossing(line1, line2):
    px, py, x, y = line1
    lpx, lpy, lx, ly = line2

    I1 = [min(py,y), max(py,y)]
    I2 = [min(lpy,ly), max(lpy,ly)]

    if I1[0]>I2[1] or I2[0]>I1[1]:
        return (False, 0)

    if px-x == 0:
        A1 = 0
    else:
        A1 = (py-y)/(px-x)

    A2 = (lpy-ly)/(lpx-lx)

    if (A1 == A2):
        return (False, 0)

    b1 = py-A1*px
    b2 = lpy-A1*lpx
    X = (b2 - b1) / (A1 - A2)

    if ((X < max(min(px,x), min(lpx,lx))) or (X > min( max(px,x), max(lpx,lx)))):
        return (False, 0)

    else:
        if py>y:
            direction = -1
        else:
            direction = 1 

        return (True, direction)

def auto_canny(image, sigma=0.33):
    # calcul de la luminosité mediane de l'image
    v = np.median(image)

    # calcul des paramètres à partir de la valeur mediane
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

        
    return edged


def camera_settings(shape, fov = 90, h = 3):

    y,x = shape

    length = (math.tan(fov/2*math.pi/180)*h)*2
    
    ratio = x/length #pixels per meters
    
    min_w = ratio*0.2
    min_h = ratio*1.4
    min_box = min_w*min_h

    max_w = ratio*1
    max_h = ratio*2
    max_box = max_w*max_h

    print(ratio, min_box, max_box)
    return (min_box, max_box)


#####################################

cap = cv2.VideoCapture('walking.mp4')


#initialisation
_, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

min_box , max_box = camera_settings(img.shape, fov = 120, h=9)


points = []
next_points = []
heat_points = [[],[]]
colors = []

algo = 0
nb_av = 0
counter = 0

crossing_line = (0, int(img.shape[0]/3), img.shape[1], int(img.shape[0]/3))
n = 30 # nombre d'image traité avant d'analyser le mouvement
i = 0
tot_i = 0


while(True):
    i += 1
    tot_i += 1

    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    can = auto_canny(cv2.blur(img,(3,3)), 0.2)
    can = cv2.resize(can, (can.shape[1]//2, can.shape[0]//2))
    
    _, cnts, hie = cv2.findContours(can, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:

        x,y,w,h = cv2.boundingRect(cnt)


        if min_box<w*h<max_box:
            cx, cy = int((x+(x+w))/2), int((y+(y+h))/2)
            points.append([cx,cy,i])

            if i == n:
                next_points.append([cx,cy,0])

            heat_points[0].append(cx)
            heat_points[1].append(cy)
            
            #cv2.rectangle(can, (x,y), (x+w, y+h), 255)

            nb_av+=1
    


    if i == n and len(points)>2:
        algo = Birch(n_clusters=round(nb_av/i), threshold=1).fit(points)
        label = algo.labels_


        min_it = np.full((int(nb_av/i)+1, 3), n)
        max_it = np.full((int(nb_av/i)+1, 3), 0)
        track = np.zeros((can.shape[0], can.shape[1], 3))

        for p in range(len(points)):
            lab = label[p]
            x,y,it = points[p]

            if min_it[lab][2] >= i:
                min_it[lab] = [x,y,it]

            if max_it[lab][2] <= i:
                max_it[lab] = [x,y,it]


        for lab in range(len(min_it)):
            px, py, _ = min_it[lab]
            x, y, _ = max_it[lab]

            crossing, direction =  is_crossing((px,py,x,y), crossing_line)

            if crossing:
                counter += direction

        #heatmap, xedges, yedges = np.histogram2d(heat_points[0], heat_points[1], bins=(16,9))
        #plt.imsave('log\\heatmap\\'+str(time.time())+'.png', heatmap.T)

        np.save('log\\points\\'+str(time.time())+'.npy', np.array([[min_it],[max_it]]))
        

        points = next_points
        next_points = []
        i = 0
        nb_av = 0



    #cv2.putText(can, str(counter), (0,can.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
    #cv2.imshow('can', can)
    #cv2.imshow('img', img)

    cv2.waitKey(1)
