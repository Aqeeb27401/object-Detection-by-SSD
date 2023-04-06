# object-Detection-by-SSD
FYP(30%) Code 
# import the dependencies
import cv2
import matplotlib.pyplot as plt

# load the files
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

#  load the model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# empty list of python
classLabels = []
# loading the labels
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    #classlables.append(fpt.read())
# we have to define the input size 320/320 because our model only supports this format
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
cap = cv2.VideoCapture() # you can leave this blank if you want to use your webcam for object detection.
# Check if the video is playing correctly or not
if not cap. isOpened():
    cap = cv2. VideoCapture (0)
if not cap. isOpened():
    raise IOError("Cannot open camera")

font_scale = 2
font = cv2. FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)


    if (len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex. flatten(), confidence.flatten(), bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame,boxes, (255, 0, 0), 3)
                cv2. putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes [1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3 )
    cv2.imshow('object Detection Project', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
