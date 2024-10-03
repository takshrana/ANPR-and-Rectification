import cv2
import os
import time
import numpy as np

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

# Define the path to the folder containing images
input_folder = '/home/nayan/Taksh/LicensePlate/License/output'
output_folder = '/home/nayan/Taksh/LicensePlate/License/yolov4'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

start_time = time.time()
# Loop through each image in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        # Construct full file path
        file_path = os.path.join(input_folder, filename)
        
        # Load the image
        image = cv2.imread(file_path)
        
        if image is None:
            print(f"Error loading image: {file_path}")
            continue

        # Step 2: Get the original dimensions
        original_height, original_width = image.shape[:2]
        
        # Step 3: Define the scaling factor
        scale_factor = 4 # Change this to your desired scaling factor
        
        # Step 4: Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Step 5: Upscale the image using INTER_LINEAR interpolation
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Step 1: Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply CLAHE to enhance the contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray_image)

        # Step 3: Perform binarization (thresholding)
        _, binary_image = cv2.threshold(clahe_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4: Create a mask for the borders
        border_mask = np.zeros_like(binary_image)
        border_mask[20:-20, 20: -20] = 1  # Create a border mask that excludes the outer 5 pixels

        # Step 5: Suppress light structures connected to the borders
        suppressed_image = cv2.bitwise_and(binary_image, binary_image, mask=border_mask.astype(np.uint8))

        # Step 6: Find contours from the suppressed binary image
        contours, _ = cv2.findContours(suppressed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 7: Create a mask for the license plate area
        license_plate_mask = np.zeros_like(gray_image)

        # Step 8: Filter contours based on area to retain the license plate
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:  # Adjust this threshold based on your license plate size
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

        gray = hulls_image.copy()
        gray = 255 - gray

        # Blur image
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

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
            # print(f"Detected box type: {box_type}")

            # Draw rotated rectangle on a copy of the image
            rot_bbox = image.copy()
            cv2.drawContours(rot_bbox, [np.intp(sorted_box)], 0, (0, 0, 255), 2)

            cv2.imshow("Gray", rot_bbox)
            # Define the destination points for affine transformation
            if box_type == "long-width":
                width, height = 700, 150  # Larger width for long rectangles
            else:
                width, height = 500, 300  # Symmetric or taller for short rectangles

            dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

            # Calculate the perspective transformation matrix
            M = cv2.getPerspectiveTransform(sorted_box, dst_pts)

            # Apply the perspective transformation
            warped = cv2.warpPerspective(image, M, (width, height))

        # # Step 12: Perform Dilation to smoothen edges
        # dilated_hulls_image = cv2.dilate(hulls_image, kernel, iterations=1)

        # # Step 13: Perform Erosion to refine the edges
        # eroded_hulls_image = cv2.erode(dilated_hulls_image, kernel, iterations=1)

        # # Step 14: Extract the boundary of the convex hull
        # boundary_image = np.zeros_like(image)  # Create an empty image for the boundary
        # for contour in contours_hull:
        #     hull = cv2.convexHull(contour)
        #     cv2.drawContours(boundary_image, [hull], -1, (0, 0, 255), thickness=2)  # Draw red boundary

        # # Step 15: Combine the original image with the boundary image
        # combined_image = cv2.addWeighted(image, 1, boundary_image, 1, 0)

        # Save the binarized image to the output folder
            
        output_path = os.path.join(output_folder, f'warped_{filename}')
        cv2.imwrite(output_path, warped)
        # print(f"Processed and saved: {output_path}")

print(f"Process completed in {time.time() - start_time}s")