# Active Contour - What is it?
This project implements a segmentation algorithmus: a snake algorithm. 
A snake consists of connected points (e.g. a circle). 
The goal is that the snake moves towards the object until 
it completely segements it once.
To segment a object the 'snake' has to minimize the following three terms: 
- Continuity term: to keep the points of the snake together.
- Curvature term:  to smooth the snake/follow the contours.
- Image term:      attract the snake towards the edges of the object.

The Cup.jpg is an own photography.
 
# How to run?
Open computer_vision_snake.ipynb in jupyter notebook and press 'run all'
Otherwise you can use google colab for example. But make sure, all images are uploaded ass well

# Implemented with jupyter notebook version:
6.1.4
