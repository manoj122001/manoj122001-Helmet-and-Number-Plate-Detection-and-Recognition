import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
from datetime import datetime

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = load_model('helmet-nonhelmet_cnn.h5')
print('model loaded!!!')

cap = cv2.VideoCapture('test.mp4')
COLORS = [(0, 255, 0), (0, 0, 255)]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter('output.avi', fourcc, 5, (888, 500))

# Create a folder to store frames
output_folder = 'frames_without_helmet'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi, dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi / 255.0
        return int(model.predict(helmet_roi)[0][0])
    except Exception as e:
        print(f"Error in helmet_or_nohelmet: {e}")
        return 0

ret = True
frame_id = 0

while ret:
    ret, img = cap.read()
    img = imutils.resize(img, height=500)

    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    classIds = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = [int(c) for c in COLORS[classIds[i]]]
            if classIds[i] == 0:  # bike
                helmet_roi = img[max(0, y):max(0, y) + max(0, h)//4, max(0, x):max(0, x) + max(0, w)]
                c = helmet_or_nohelmet(helmet_roi)
                cv2.putText(img, ['helmet', 'no-helmet'][c], (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                # Save the entire frame when a bike rider is detected without a helmet
                if c == 1:
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    frame_filename = f"{output_folder}/frame_{timestamp}_{frame_id}.jpg"
                    cv2.imwrite(frame_filename, img)

    writer.write(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        break

    frame_id += 1

writer.release()
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
