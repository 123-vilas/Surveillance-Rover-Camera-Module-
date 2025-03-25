from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# YOLO Configuration
whT = 224  # Input image size (must be multiple of 32)
confThreshold = 0.5  # Confidence threshold for filtering detections
nmsThreshold = 0.4  # Non-Maximum Suppression threshold

# Load class names
with open('YOLOv3/coco.names') as f:
    classes = f.read().strip().split('\n')

# Load YOLO model
net = cv2.dnn.readNetFromDarknet('YOLOv3/yolov3.cfg', 'YOLOv3/yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def drawPred(classId, conf, left, top, right, bottom, frame):
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = f"{classes[classId]}: {conf:.2f}"
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])

    cv2.rectangle(frame, (left, top - labelSize[1] - 5),
                  (left + labelSize[0] + 5, top + baseLine - 5), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def postprocess(frame, outputs):
    frameHeight, frameWidth = frame.shape[:2]
    classIds, confidences, boxes = [], [], []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                center_x, center_y, width, height = (
                    int(detection[0] * frameWidth),
                    int(detection[1] * frameHeight),
                    int(detection[2] * frameWidth),
                    int(detection[3] * frameHeight),
                )
                left, top = int(center_x - width / 2), int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    if indices is None or len(indices) == 0:
        return  

    for i in indices.flatten():
        left, top, width, height = boxes[i]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)

def generate():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        blob = cv2.dnn.blobFromImage(frame, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outputs = net.forward(getOutputsNames(net))

        postprocess(frame, outputs)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# import numpy as np
# import cv2

# # YOLO Configuration
# whT = 224  # Input image size (must be multiple of 32)
# confThreshold = 0.5  # Confidence threshold for filtering detections
# nmsThreshold = 0.4  # Non-Maximum Suppression threshold

# # Load class names
# with open('YOLOv3/coco.names') as f:
#     classes = f.read().strip().split('\n')

# # Load YOLO model
# net = cv2.dnn.readNetFromDarknet('YOLOv3/yolov3.cfg', 'YOLOv3/yolov3.weights')
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# # Get output layer names
# def getOutputsNames(net):
#     layer_names = net.getLayerNames()
#     return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# # Draw bounding box
# def drawPred(classId, conf, left, top, right, bottom):
#     cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
#     label = f"{classes[classId]}: {conf:.2f}"
    
#     labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#     top = max(top, labelSize[1])
    
#     cv2.rectangle(frame, (left, top - labelSize[1] - 5), 
#                   (left + labelSize[0] + 5, top + baseLine - 5), (255, 255, 255), cv2.FILLED)
#     cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# # Process detections
# def postprocess(frame, outputs):
#     frameHeight, frameWidth = frame.shape[:2]
#     classIds, confidences, boxes = [], [], []

#     for out in outputs:
#         for detection in out:
#             scores = detection[5:]
#             classId = np.argmax(scores)
#             confidence = scores[classId]

#             if confidence > confThreshold:
#                 center_x, center_y, width, height = (
#                     int(detection[0] * frameWidth),
#                     int(detection[1] * frameHeight),
#                     int(detection[2] * frameWidth),
#                     int(detection[3] * frameHeight),
#                 )
#                 left, top = int(center_x - width / 2), int(center_y - height / 2)
#                 classIds.append(classId)
#                 confidences.append(float(confidence))
#                 boxes.append([left, top, width, height])

#     indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
#     if indices is None or len(indices) == 0:
#         return  

#     for i in indices.flatten():
#         left, top, width, height = boxes[i]
#         drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

# # Start video capture
# cap = cv2.VideoCapture(0)

# while True:
#     success, frame = cap.read()
#     if not success:
#         print("Failed to capture frame")
#         break

#     # Preprocess image for YOLO
#     blob = cv2.dnn.blobFromImage(frame, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
#     net.setInput(blob)
#     outputs = net.forward(getOutputsNames(net))

#     postprocess(frame, outputs)
#     cv2.imshow('YOLO Object Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

