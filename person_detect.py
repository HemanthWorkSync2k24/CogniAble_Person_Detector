import cv2
import numpy as np

# Load pre-trained YOLO model for person detection
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()

# Get the output layers
unconnected_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_layers]

# Load COCO names file to label detected objects
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
cap = cv2.VideoCapture("video_20.mp4") #Sample video 

# Print video properties
print("Frame width: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Frame height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frame rate (FPS): ", cap.get(cv2.CAP_PROP_FPS))
print("Total frames: ", cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get the frame rate of the input video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Try changing this to 'MJPG' if necessary
out = cv2.VideoWriter('video_20op.avi', fourcc, fps, (frame_width, frame_height))  #Sample video output file name

# Prepare a unique ID counter
id_counter = 0
person_dict = {}

# Function to detect and assign unique IDs
def detect_and_assign_id(frame, id_counter, person_dict):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if classes[class_id] == "person" and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Assign a unique ID if not already in person_dict
            if i not in person_dict:
                id_counter += 1
                person_dict[i] = id_counter

            # Display the bounding box and ID on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {person_dict[i]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return id_counter, person_dict

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    id_counter, person_dict = detect_and_assign_id(frame, id_counter, person_dict)

    # Write the frame to the video file
    out.write(frame)

# Release everything when the job is finished
cap.release()
out.release()
#cv2.destroyAllWindows()
