# Code_YOLO
In this folder you will find all the code for the object detection module.


## Using pretrained YOLO
to run object detection with webcam using pretrained general YOLO, run the following from this directory : 
```
python food_detection.py --model yolov8s.pt --camera 0 --conf 0.25
```
Note that this model is restricted to items that can be found in a fridge.
As a consequence, the only detectable items are the following : 
```
 {39: 'bottle', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake'}
```


## Fine tuned Model
CURRENTLY NOT WORKING <br>
Use a model fined tuned for object detection on fridge content.

to run object detection with webcam
```
python food_detection.py --model best_yolov8s_food_cleaned_dataset_1.pt --camera 0 --conf 0.25
```
