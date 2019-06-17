import numpy as np

import cv2
import matplotlib as plt
from glob import glob


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


img = np.zeros((1080,1920,3))
crossing_line = (0, int(img.shape[0]/3), img.shape[1], int(img.shape[0]/3))

p_dos = glob('log\\points\\*')
p_dos.sort()

counter = 0

while(True):
    for p in p_dos:
        min_it, max_it = np.load(p)
        track = np.zeros((img.shape[0]//2, img.shape[1]//2, 3))
        
        colors = []
        for lab in range(len(min_it[0])):
            b = np.random.random()
            g = np.random.random()
            r = np.random.random()
            colors.append((b,g,r))


        for i in range(len(min_it[0])):
            px, py, _ = min_it[0][i]
            x, y, _ = max_it[0][i]

            cv2.line(track, (px,py), (x,y), color = colors[i])
            crossing, direction =  is_crossing((px,py,x,y), crossing_line)

            if crossing:
                counter += direction

        cv2.line(track, (crossing_line[0], crossing_line[1]), (crossing_line[2], crossing_line[3]), color = (255,255,255))
        cv2.putText(track, str(counter), (0,track.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
        cv2.imshow('track', track)
        cv2.waitKey(660)