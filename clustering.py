import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import *
import collections
import math


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


cap = cv2.VideoCapture('walking.mp4')



#initialisation
_, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
memory = img

min_box = 800
max_box = 7500
points = []
next_points = []
colors = []
algo = 0
nb_av = 0
counter = 0

crossing_line = (0, int(img.shape[0]/3), img.shape[1], int(img.shape[0]/3))
n = 20 # nombre d'image traité avant d'analyser le mouvement
i = 0

while(True):
    i+=1

    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    '''
    #difference entre l'image x0 et l'image x1 (detecter le mouvement)
    new_img = (img-memory*0.5)
    new_img[new_img<0]=0

    new_img = np.uint8(new_img)
    '''

    can = auto_canny(img, 0.2)
    can = cv2.resize(can, (len(can[0])//2, len(can)//2))
    
    _, cnts, hie = cv2.findContours(can, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:

        x,y,w,h = cv2.boundingRect(cnt)


        if min_box<w*h<max_box:
            cv2.rectangle(can, (x,y), (x+w, y+h), 255)

            points.append([int((x+(x+w))/2),int((y+(y+h))/2),i])
            if i == n:
                next_points.append([int((x+(x+w))/2),int((y+(y+h))/2),0])

            nb_av+=1
    


    if i == n and len(points)>2:
        track = np.zeros((can.shape[0], can.shape[1], 3))

        algo = Birch(n_clusters=int(nb_av/i)+1).fit(points)
        label = algo.labels_

        colors = []

        ctr = collections.Counter(np.sort(label))
        for lab in range(len(ctr)):
            b = np.random.randint(0, 255)
            g = np.random.randint(0, 255)
            r = np.random.randint(0, 255)
            colors.append((b,g,r))



        min_it = np.full((int(nb_av/i)+1, 3), n*(i+1))
        max_it = np.full((int(nb_av/i)+1, 3), 0)

        for p in range(len(points)):
            lab = label[p]
            x,y,it = points[p]

            if min_it[lab][2] >= it:
                min_it[lab] = [x,y,it]

            if max_it[lab][2] <= it:
                max_it[lab] = [x,y,it]


            track[y][x]= colors[lab]



        for lab in range(len(min_it)):
            px, py, _ = min_it[lab]
            x, y, _ = max_it[lab]

            cv2.line(track, (px,py), (x,y), color = colors[lab])
            cv2.line(track, (crossing_line[0], crossing_line[1]), (crossing_line[2], crossing_line[3]), color = (255,255,255))

            crossing, direction =  is_crossing((px,py,x,y), crossing_line)
            if crossing:
                counter += direction
                if direction>0:
                    print('someone has entered')
                else:
                    print('someone has exited')


        cv2.imshow('track', track/255)

        points = next_points
        next_points = []
        i = 0
        nb_av = 0


    cv2.putText(can, str(counter), (0,can.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))

    cv2.imshow('can', can)
    cv2.imshow('img', img)


    cv2.waitKey(1)

    memory = img