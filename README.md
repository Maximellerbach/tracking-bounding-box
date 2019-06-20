# bounding-box-tracking
used sklearn and OpenCV to track bounding box in a sequence of frame.
I detect people's bounding box using Opencv then I add their coordinates to an array of points,
every 20 images or so, I use clusering to determine if the box has moved and where did it moved

/!\ this project is still under developpement 

![img](gifs/img.gif)

input image

![can](gifs/can.gif)

canny image + box visalization

![track](gifs/track.gif)

clustered coordinates
