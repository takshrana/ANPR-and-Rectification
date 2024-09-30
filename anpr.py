import cv2
import asyncio
from fastanpr import FastANPR
import os
import numpy as np
import psutil
import time

# Initialize FastANPR model
fast_anpr = FastANPR()

# Define paths for the input and output
input_file_path = "path/to/image"  # Specify the image file

def measure_cpu(func):
    # Get the initial CPU usage
    initial_cpu = psutil.cpu_percent(interval=None)
    
    start_time = time.time()  # Start time
    func()                    # Execute the function
    end_time = time.time()    # End time
    
    # Get the CPU usage after the function
    final_cpu = psutil.cpu_percent(interval=None)
    
    # Calculate elapsed time and CPU usage
    elapsed_time = end_time - start_time
    cpu_usage = final_cpu - initial_cpu
    
    print(f"{func.__name__} took {elapsed_time:.4f} seconds to execute")
    print(f"CPU consumption for {func.__name__}: {cpu_usage:.2f}%")

def sort_corners(corners):
    sum_coords = np.sum(corners, axis=1)
    diff_coords = np.diff(corners, axis=1)

    top_left = corners[np.argmin(sum_coords)]
    bottom_right = corners[np.argmax(sum_coords)]
    top_right = corners[np.argmin(diff_coords)]
    bottom_left = corners[np.argmax(diff_coords)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

def classify_box(rot_rect):
    width = rot_rect[1][0]
    height = rot_rect[1][1]
    aspect_ratio = max(width, height) / min(width, height)
    
    # Define threshold for classifying as long or short width
    if aspect_ratio > 2.5:
        return "long-width"
    else:
        return "short-width"
    
async def cropped_anpr(img):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run FastANPR on the image
    number_plates = await fast_anpr.run(image_rgb)  # Await the coroutine
    
    # Find the plate with the highest confidence score
    highest_confidence = -1
    best_plate = None
    
    for plate_list in number_plates:
        for plate in plate_list:
            if plate.rec_conf is not None and plate.rec_conf > highest_confidence:
                highest_confidence = plate.rec_conf
                best_plate = plate
        if best_plate is None:
            best_plate = plate_list[0]
    
    # If a best plate is found, crop the image based on its bounding box
    if best_plate is not None and best_plate.det_box is not None:
        x1, y1, x2, y2 = best_plate.det_box
        x1 -= 2  # Expand x1 to the left by 2 pixels
        y1 -= 2  # Expand y1 upward by 2 pixels
        x2 += 2  # Expand x2 to the right by 2 pixels
        y2 += 2  # Expand y2 downward by 2 pixels
        
        # Ensure the coordinates are within the image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        # Crop the image using the adjusted bounding box
        cropped_plate = img[y1:y2, x1:x2]

    return cropped_plate

def upscale_image(image):
    original_height, original_width = image.shape[:2]
        
    # Step 3: Define the scaling factor
    scale_factor = 3 
    # Change this to your desired scaling factor
        
    # Step 4: Calculate new dimensions
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)
    
    # Step 5: Upscale the image using INTER_LINEAR interpolation
    upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return upscaled_image

def generate_convex_hull(image):
    # Step 1: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply CLAHE to enhance the contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)

    # Step 3: Perform binarization (thresholding)
    _, binary_image = cv2.threshold(clahe_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Create a mask for the borders
    border_mask = np.zeros_like(binary_image)
    border_mask[15:-15, 15:-15] = 1  # Create a border mask that excludes the outer 15 pixels

    # Step 5: Suppress light structures connected to the borders
    suppressed_image = cv2.bitwise_and(binary_image, binary_image, mask=border_mask.astype(np.uint8))

    # Step 6: Find contours from the suppressed binary image
    contours, _ = cv2.findContours(suppressed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Create a mask for the license plate area
    license_plate_mask = np.zeros_like(gray_image)

    # Step 8: Filter contours based on area to retain the license plate
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Adjust this threshold based on your license plate size
            cv2.drawContours(license_plate_mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Step 9: Perform morphological closing (dilation followed by erosion)
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size
    closed_mask = cv2.morphologyEx(license_plate_mask, cv2.MORPH_CLOSE, kernel)

    # Step 10: Perform morphological opening (erosion followed by dilation)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

    # Step 11: Find contours from the opened mask to compute the convex hull
    hulls_image = np.zeros_like(opened_mask)  # Create an empty image to draw the convex hull
    contours_hull = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Draw the convex hull for each contour
    for contour in contours_hull:
        hull = cv2.convexHull(contour)
        cv2.drawContours(hulls_image, [hull], -1, (255), thickness=cv2.FILLED)

    return hulls_image

def affine_transformation(hulls_image, upscaled_image):
    gray = hulls_image.copy()
    gray = 255 - gray

    # Blur image
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # cv2.imshow("blur",  blur)
    # cv2.waitKey(0)

    # Adaptive threshold on gray image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 2)
    thresh = 255 - thresh

    # Apply morphology
    kernel = np.ones((5, 5), np.uint8)
    rect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel)

    # Thin using erosion
    rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)

    # Find the largest contour
    contours = cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    big_contour = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
            big_contour = c

    # Get rotated rectangle from the largest contour
    rot_rect = cv2.minAreaRect(big_contour)
    box = cv2.boxPoints(rot_rect)
    box = np.intp(box)

    # Sort the corner points
    sorted_box = sort_corners(box)

    # Classify the box
    box_type = classify_box(rot_rect)
    print(f"Detected box type: {box_type}")
    
    # Draw rotated rectangle on a copy of the image
    rot_bbox = upscaled_image.copy()
    cv2.drawContours(rot_bbox, [np.intp(sorted_box)], 0, (0, 0, 255), 2)
    # cv2.imshow("Gray", rot_bbox)

    # Define the destination points for affine transformation
    if box_type == "long-width":
        width, height = 700, 150  # Larger width for long rectangles
    else:
        width, height = 500, 300  # Symmetric or taller for short rectangles
    dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    # print("dstPoints" ,dst_pts)
    # print("Detected box ", sorted_box)

    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(sorted_box, dst_pts)
    # print(width, height)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(upscaled_image, M, (width, height))
    
    return warped

# Define an async function to run the FastANPR model on a single image
def main():
    # Load the specific image
    img = cv2.imread(input_file_path)
    filename = os.path.basename(input_file_path)

    if img is None:
        print(f"Image not found at path: {input_file_path}")
        return
    
    # start_time = time.time()
    # print("function started after ", main_start-start_time)

    cropped_plate = asyncio.run(cropped_anpr(img))

    # end_time = time.time()

    # print(end_time-start_time, "s")
        # Optionally, print a message
        # cv2.imshow("Output", cropped_plate)
        # cv2.waitKey(0)

    upscaled_image = upscale_image(cropped_plate)

    hulls_image = generate_convex_hull(upscaled_image)

    warped = affine_transformation(hulls_image, upscaled_image)

    cv2.imshow("Affine Transformed LP", warped)
    cv2.waitKey(0)


main()
