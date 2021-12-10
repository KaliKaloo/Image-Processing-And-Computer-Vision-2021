# Image-Processing-And-Computer-Vision-2021

Source files can be compilied and run using the piece of code below:  
```
g++ 'file_name'.cpp /usr/lib64/libopencv_core.so.2.4/usr/lib64/libopencv_highgui.so.2.4
```
```
./a.out 'insert image filepath'    
```

Below are the source files and their outputs:    
**Subtask 1 face detection:** gt_face.cpp  
**outout:** Images are located in groundTruths_faces folder. Draws detected faces in green and ground truths in red.

**Subtask 2 no-entry detection:** gt_NoEntry.cpp  
**outout:** Images are located in groundTruths_NoEntry folder. Draws detected signs in green and ground truths in red.

**Subtask 3 Shape detection:** gt_Hough_NoEntry.cpp  
**outout:** Images are located in groundTruths_NoEntry_Hough folder. There are subfolders for each image. Within those sub-folders, there is:
- circles detected
- signs detected by viola-jones
- filtered VJ detections using the circles
- x & y direction output
- hough space
- gradient magnitude
- gradient magnitude detection
- thresholded gradient magnitude detection

**Subtask 4 face detection:** subtask4.cpp  
**outout:** Images are located in subtask4 folder. Folder format is the same as in subtask 3.

