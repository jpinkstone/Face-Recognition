# Face-Recognition
A test program to train and detect faces of people. This is a personal project I completed myself.

The images folder, pickle file, and yml were left out for privacy. 

This program uses OpenCV to train a dataset with pictures of people and uses that to detect a face and which person it thinks it is.

## Code Segments
 - The face-train.py file uses an uploaded folder "images" that contains directories with faces inside. The directories must by named with the person's name. The program trains the faces into a yml file for use in the main program.
 - The main.py program uses the trained face data from the yml file to detect if a face is present in the frame, and then decides which person from the trained dataset it is. The program uses a frontal face haar cascade and eye haar cascade to detect face fronts and eyes. A box is drawn around the face and eyes if they are present, and the name of the person with a confidence percentage is also displayed.
