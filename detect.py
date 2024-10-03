import cv2
import numpy as np
import os
import time

start_time = time.time()

# Load YOLOv4 network
net = cv2.dnn.readNet("/home/nayan/Taksh/LicensePlate/yolov4/lp_detector_yolo_training_best.weights", 
                       "/home/nayan/Taksh/LicensePlate/yolov4/lp_detector_yolo_training.cfg")

# Load the class names
with open("/home/nayan/Taksh/LicensePlate/yolov4/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
end_time = time.time()

print(f"took {end_time-start_time}s to Load YoloV4tiny")
# Define the folder containing images

start_time = time.time()
input_folder = "/home/nayan/Taksh/LicensePlate/License/input/"
output_folder = "/home/nayan/Taksh/LicensePlate/License/output/"

# Get all image files in the folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]


# Iterate through each image file
for image_file in image_files:
    # Load the input image
    img_path = os.path.join(input_folder, image_file)
    img = cv2.imread(img_path)  # Load the image
    height, width, channels = img.shape

    # Prepare the image for YOLOv4
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Run the forward pass and get the network outputs
    detections = net.forward(output_layers)

    # Initialize lists to hold detection data
    class_ids = []
    confidences = []
    boxes = []

    # Process the detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Set a confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  + 10
                h = int(detection[3] * height) + 10

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform Non-Maximum Suppression to remove overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and labels on the image
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # confidence = str(round(confidences[i], 2))
            # color = (0, 255, 0)
            # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(img, f"{label} {confidence}", (x, y - 10), font, 1, color, 2)

            # Crop the detected object (license plate)
            cropped_img = img[y:y + h, x:x + w]
            cropped_output_path = os.path.join(output_folder, f"cropped_{i}_{image_file}")
            cv2.imwrite(cropped_output_path, cropped_img)
            

    # Show the image with detections
    # cv2.imshow("YOLOv4 Detection", img)
    # cv2.waitKey(0)  # Wait for a key press to proceed to the next image
    # cv2.destroyAllWindows()

    # Optionally, save the result image
    # output_path = os.path.join(output_folder, f"output_{image_file}")
    # cv2.imwrite(output_path, img)
end_time = time.time()

print(f"took {end_time-start_time}s to Crop 10 Vehicle Images")