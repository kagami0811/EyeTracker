import cv2
import numpy as np
import random
import math
import tkinter as tk
import os
from tkinter import filedialog
import matplotlib.pyplot as plt
import argparse
import ctypes
import tkinter.ttk as ttk
from PIL import Image, ImageOps, ImageTk
import copy

TK_WINDOW_WIDTH = 1400
TK_WINDOW_HEIGHT = 800

IMAGE_W = 600
IMAGE_H = 600

DEBUG = False
FRAME_COUNT = 0

# default
default_th_dict = {"thresholded_image_strict_value":5, "thresholded_image_medium_value":15,"thresholded_image_relaxed_value":25, "eye_size_value": 250, "search_w_value":100, "search_h_value":100 }


def show_next_frame(thresholded_image_strict_value, thresholded_image_medium_value,thresholded_image_relaxed_value,eye_size_value, search_w_value, search_h_value, cap, canvas):
    global FRAME_COUNT
    FRAME_COUNT += 1
    print(f"FRAME {FRAME_COUNT}")
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_COUNT)
        ret, original_image = cap.read()
        if ret == False:
            FRAME_COUNT += 1
        if ret == True:
            break
    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = crop_to_aspect_ratio(original_image)
    
    original_rgb_pil = Image.fromarray(original_image)
    # canvas.photo = ImageTk.PhotoImage(original_rgb_pil)
    # canvas_create = canvas.create_image(0,0,anchor='nw', image=canvas.photo)
    # canvas.itemconfig(canvas_create, image= canvas.photo)
    # replace_image(canvas=canvas, img_pil=original_rgb_pil)
    bottun_click(thresholded_image_strict_value, thresholded_image_medium_value,thresholded_image_relaxed_value, eye_size_value, search_w_value, search_h_value, original_image, canvas)

    

def bottun_click(thresholded_image_strict_value, thresholded_image_medium_value,thresholded_image_relaxed_value, eye_size_value, search_w_value, search_h_value, original_image, canvas):
    
    th_dict = copy.deepcopy(default_th_dict)
    thresholded_image_strict_value = thresholded_image_strict_value.get()
    thresholded_image_medium_value = thresholded_image_medium_value.get()
    thresholded_image_relaxed_value = thresholded_image_relaxed_value.get()
    eye_size_value = eye_size_value.get()
    search_w_value = search_w_value.get()
    search_h_value = search_h_value.get()
    
    th_dict["thresholded_image_strict_value"] = int(thresholded_image_strict_value)
    th_dict["thresholded_image_medium_value"] = int(thresholded_image_medium_value)
    th_dict["thresholded_image_relaxed_value"] = int(thresholded_image_relaxed_value)
    th_dict["eye_size_value"] = int(eye_size_value)
    th_dict["search_h_value"] = int(search_h_value)
    th_dict["search_w_value"] = int(search_w_value)



    assert len(th_dict.keys()) == len(default_th_dict.keys())
    
    darkest_point = get_darkest_area(original_image, search_w=th_dict["search_w_value"], search_h=th_dict["search_h_value"])

    # Convert to grayscale to handle pixel value operations
    gray_frame = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    # apply thresholding operations at different levels
    # at least one should give us a good ellipse segment
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, th_dict["thresholded_image_strict_value"])#lite
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, th_dict["eye_size_value"])

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, th_dict["thresholded_image_medium_value"])#medium
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, th_dict["eye_size_value"])
    
    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, th_dict["thresholded_image_relaxed_value"])#heavy
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, th_dict["eye_size_value"])
    
    #take the three images thresholded at different levels and process them
    final_rotated_rect, test_frame = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, original_image, gray_frame, darkest_point, False, False, th_dict)
    
    img_pil = Image.fromarray(test_frame)
    replace_image(canvas=canvas, img_pil=img_pil)
    
    print(f"th_dict {th_dict}")                  


def replace_image(canvas, img_pil):
    canvas.photo = ImageTk.PhotoImage(img_pil)
    canvas_create = canvas.create_image(0,0,anchor='nw', image=canvas.photo)
    canvas.itemconfig(canvas_create, image= canvas.photo)

# Crop the image to maintain a specific aspect ratio (width:height) before resizing. 
def crop_to_aspect_ratio(image, width=IMAGE_W, height=IMAGE_H):
    
    # Calculate current aspect ratio
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset+new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset+new_height, :]

    return cv2.resize(cropped_img, (width, height))

#apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    # Calculate the threshold as the sum of the two input values
    threshold = darkestPixelValue + addedThreshold
    # Apply the binary threshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded_image

#Finds a square area of dark pixels in the image
#@param I input image (converted to grayscale during search process)
#@return a point within the pupil region
def get_darkest_area(image, search_w, search_h):

    ignoreBounds = 20 #don't search the boundaries of the image for ignoreBounds pixels
    imageSkipSize = 10 #only check the darkness of a block for every Nth x and y pixel (sparse sampling)
    searchArea = 20 #the size of the block to search
    internalSkipSize = 5 #skip every Nth x and y pixel in the local search area (sparse sampling)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 中心からsearch_w, seach_hの範囲でdarkest_pointを探す
    h, w = gray.shape
    center_x = w //2
    center_y = h //2
    min_x = max(0, center_x - search_w)
    max_x = min(w, center_x + search_w)
    min_y = max(0, center_y - search_h)
    max_y = min(h, center_y + search_h)


    min_sum = float('inf')
    darkest_point = None

    # Loop over the image with spacing defined by imageSkipSize, ignoring the boundaries
    for y in range(min_x, max_x,imageSkipSize):
        for x in range(min_y, max_y, imageSkipSize):
            # Calculate sum of pixel values in the search area, skipping pixels based on internalSkipSize
            current_sum = np.int64(0)
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1

            # Update the darkest point if the current block is darker
            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)  # Center of the block

    return darkest_point

#mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    # Create a mask initialized to black
    mask = np.zeros_like(image)

    # Calculate the top-left corner of the square
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)

    # Calculate the bottom-right corner of the square
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)

    # Set the square area in the mask to white
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)
    #DEBUG
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow("img", masked_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return masked_image
   
def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    # Holds the candidate points
    all_contours = np.concatenate(contours[0], axis=0)

    # Set spacing based on size of contours
    spacing = int(len(all_contours)/25)  # Spacing between sampled points

    # Temporary array for result
    filtered_points = []
    
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
    # Create an image of the same size as the original image
    point_image = image.copy()
    
    skip = 0
    
    # Loop through each point in the all_contours array
    for i in range(0, len(all_contours), 1):
    
        # Get three points: current point, previous point, and next point
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        # Calculate vectors between points
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            # Calculate angles between vectors
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        
        # Calculate vector from current point to centroid
        vec_to_centroid = centroid - current_point
        
        # Check if angle is oriented towards centroid
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  # Convert angle to radians
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

#returns the largest contour that is not extremely long or tall
#contours is the list of contours, pixel_thresh is the max pixels to filter, and ratio_thresh is the max ratio
def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh, mask_image, debug_counter_images, debug_elipse_images,debug_ellipse_on_eye_images, frame):
    max_area = 0
    largest_contour = None

    #DEBUG
    # print(len(contours))
    draw_counter_image = copy.deepcopy(mask_image)
    draw_counter_image = cv2.drawContours(cv2.cvtColor(draw_counter_image,cv2.COLOR_GRAY2BGR ), contours, -1, (0,0,255), 10)
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow("img", draw_counter_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    debug_counter_images.append(draw_counter_image)


    draw_elipse_image = copy.deepcopy(mask_image)
    draw_elipse_image = cv2.cvtColor(draw_elipse_image,cv2.COLOR_GRAY2BGR )
    draw_elipse_on_eye_image = copy.deepcopy(frame)
    
    for i in range(len(contours)):
        counter = contours[i]
        # Fit an ellipse to the contour
        if len(counter) < 5:
            continue
        ellipse = cv2.fitEllipse(counter)

        # Draw the ellipse on the mask with white color (255)
        
        
        cv2.ellipse(draw_elipse_image, ellipse, (255, 255, 0), 3)
        cv2.ellipse(draw_elipse_on_eye_image, ellipse, (255, 255, 0), 3)
        
    debug_elipse_images.append(draw_elipse_image)
    debug_ellipse_on_eye_images.append(draw_elipse_on_eye_image)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            width = min(w, h)

            # Calculate the length-to-width ratio and width-to-length ratio
            length_to_width_ratio = length / width
            width_to_length_ratio = width / length

            # Pick the higher of the two ratios
            current_ratio = max(length_to_width_ratio, width_to_length_ratio)

            # Check if highest ratio is within the acceptable threshold
            if current_ratio <= ratio_thresh:
                # Update the largest contour if the current one is bigger
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    # Return a list with only the largest contour, or an empty list if no contour was found
    if largest_contour is not None:
        return [largest_contour], debug_counter_images, debug_elipse_images, debug_ellipse_on_eye_images
    else:
        return [], debug_counter_images, debug_elipse_images, debug_ellipse_on_eye_images

#Fits an ellipse to the optimized contours and draws it on the image.
def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        # Ensure the data is in the correct shape (n, 1, 2) for cv2.fitEllipse
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        # Draw the ellipse
        cv2.ellipse(image, ellipse, color, 2)  # Draw with green color and thickness of 2

        return image
    else:
        print("Not enough points to fit an ellipse.")
        return image

#checks how many pixels in the contour fall under a slightly thickened ellipse
#also returns that number of pixels divided by the total pixels on the contour border
#assists with checking ellipse goodness    
def check_contour_pixels(contour, image_shape, debug_mode_on):
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        return [0, 0]  # Not enough points to fit an ellipse
    
    # Create an empty mask for the contour
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask, filling it
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    # Fit an ellipse to the contour and create a mask for the ellipse
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    # Draw the ellipse with a specific thickness
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) #capture more for absolute
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) #capture fewer for ratio

    # Calculate the overlap of the contour mask and the thickened ellipse mask
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    # Count the number of non-zero (white) pixels in the overlap
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)#compute with thicker border
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)#compute with thicker border
    
    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = np.sum(contour_mask > 0)
    
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

#outside of this method, select the ellipse with the highest percentage of pixels under the ellipse 
#TODO for efficiency, work with downscaled or cropped images
def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] #covered pixels, edge straightness stdev, skewedness   
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        print("length of contour was 0")
        return 0  # Not enough points to fit an ellipse
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Create a mask with the same dimensions as the binary image, initialized to zero (black)
    mask = np.zeros_like(binary_image)
    
    # Draw the ellipse on the mask with white color (255)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    # Calculate the number of pixels within the ellipse
    ellipse_area = np.sum(mask == 255)
    
    # Calculate the number of white pixels within the ellipse
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    # Calculate the percentage of covered white pixels within the ellipse
    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness  # Avoid division by zero if the ellipse area is somehow zero
    
    #percentage of covered pixels to number of pixels under area
    ellipse_goodness[0] = covered_pixels / ellipse_area
    
    #skew of the ellipse (less skewed is better?) - may not need this
    axes_lengths = ellipse[1]  # This is a tuple (minor_axis_length, major_axis_length)
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness

def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window, th_dict):
  
    final_rotated_rect = ((0,0),(0,0),0)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict] #holds images
    name_array = ["relaxed", "medium", "strict"] #for naming windows
    final_image = image_array[0] #holds return array
    final_contours = [] #holds final contours
    ellipse_reduced_contours = [] #holds an array of the best contour points from the fitting process
    goodness = 0 #goodness value for best ellipse
    best_array = 0 
    kernel_size = 5  # Size of the kernel (5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray_copy1 = gray_frame.copy()
    gray_copy2 = gray_frame.copy()
    gray_copy3 = gray_frame.copy()
    gray_copies = [gray_copy1, gray_copy2, gray_copy3]
    final_goodness = 0

    debug_counter_images = []
    debug_elipse_images = []
    debug_ellipse_on_eye_images = []
    
    #iterate through binary images and see which fits the ellipse best
    for i in range(1,4):
        # Dilate the binary image
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)#medium
        # dilated_image = image_array[i-1]
        
        # Find contours
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw contours
        contour_img2 = np.zeros_like(dilated_image)
        reduced_contours, debug_counter_images, debug_elipse_images, debug_ellipse_on_eye_images = filter_contours_by_area_and_return_largest(contours, 1000, 3, mask_image=dilated_image, debug_counter_images=debug_counter_images, debug_elipse_images=debug_elipse_images, debug_ellipse_on_eye_images=debug_ellipse_on_eye_images, frame=frame) # th

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            #gray_copy = gray_frame.copy()
            #cv2.drawContours(gray_copies[i-1], reduced_contours, -1, (255), 1)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            if debug_mode_on: #show contours 
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])
                
            #in total pixels, first element is pixel total, next is ratio
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)                 
            
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)  # Draw with specified color and thickness of 2
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            
            final_goodness = current_goodness[0]*total_pixels[0]*total_pixels[0]*total_pixels[1]
            
            #show intermediary images with text output
            if debug_mode_on:
                cv2.putText(gray_copies[i-1], "%filled:     " + str(current_goodness[0])[:5] + " (percentage of filled contour pixels inside ellipse)", (10,30), font, .55, (255,255,255), 1) #%filled
                cv2.putText(gray_copies[i-1], "abs. pix:   " + str(total_pixels[0]) + " (total pixels under fit ellipse)", (10,50), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "pix ratio:  " + str(total_pixels[1]) + " (total pix under fit ellipse / contour border pix)", (10,70), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "final:     " + str(final_goodness) + " (filled*ratio)", (10,90), font, .55, (255,255,255), 1) #skewedness
                cv2.imshow(name_array[i-1] + " threshold", image_array[i-1])
                cv2.imshow(name_array[i-1], gray_copies[i-1])

        if final_goodness > 0 and final_goodness > goodness: 
            goodness = final_goodness
            ellipse_reduced_contours = total_pixels[2]
            best_image = image_array[i-1]
            final_contours = reduced_contours
            final_image = dilated_image
    
    if debug_mode_on:
        cv2.imshow("Reduced contours of best thresholded image", ellipse_reduced_contours)

    test_frame = frame.copy()
    
    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    
    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0] > 5):
        #cv2.drawContours(test_frame, final_contours, -1, (255, 255, 255), 1)
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse
        cv2.ellipse(test_frame, ellipse, (55, 255, 0), 2)
        #cv2.circle(test_frame, darkest_point, 3, (255, 125, 125), -1)
        center_x, center_y = map(int, ellipse[0])
        cv2.circle(test_frame, (center_x, center_y), 3, (255, 255, 0), -1)
        # cv2.putText(test_frame, "SPACE = play/pause", (10,410), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #space
        # cv2.putText(test_frame, "Q      = quit", (10,430), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #quit
        # cv2.putText(test_frame, "D      = show debug", (10,450), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #debug

    if render_cv_window:
        cv2.imshow('best_thresholded_image_contours_on_frame', test_frame)
    
    # Create an empty image to draw contours
    contour_img3 = np.zeros_like(image_array[i-1])
    
    if len(final_contours[0]) >= 5:
        contour = np.array(final_contours[0], dtype=np.int32).reshape((-1, 1, 2)) #format for cv2.fitEllipse
        ellipse = cv2.fitEllipse(contour) # Fit ellipse
        cv2.ellipse(gray_frame, ellipse, (255,255,255), 2)  # Draw with white color and thickness of 2

    #DEBUG
    if DEBUG:
        fig =  plt.figure(figsize=(30, 40))
        plt.title(f"frame {FRAME_COUNT} {th_dict}")
        ax1 = fig.add_subplot(4, 3, 1)
        debug_image = debug_counter_images[0]
        debug_image = Image.fromarray(np.uint8(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)))
        ax1.imshow(debug_image)
        ax1.axis("off")
        ax2 = fig.add_subplot(4, 3, 2)
        debug_image = debug_counter_images[1]
        debug_image = Image.fromarray(np.uint8(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)))
        ax2.imshow(debug_image)
        ax2.axis("off")
        ax3 = fig.add_subplot(4, 3, 3)
        debug_image = debug_counter_images[2]
        debug_image = Image.fromarray(np.uint8(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)))
        ax3.imshow(debug_image)
        ax3.axis("off")

        ax4 = fig.add_subplot(4, 3, 4)
        debug_image = debug_elipse_images[0]
        debug_image = Image.fromarray(np.uint8(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)))
        ax4.imshow(debug_image)
        ax4.axis("off")
        ax5 = fig.add_subplot(4, 3, 5)
        debug_image = debug_elipse_images[1]
        debug_image = Image.fromarray(np.uint8(cv2.cvtColor(debug_image,cv2.COLOR_BGR2RGB)))
        ax5.imshow(debug_image)
        ax5.axis("off")
        ax6 = fig.add_subplot(4, 3, 6)
        debug_image = debug_elipse_images[2]
        debug_image = Image.fromarray(np.uint8(cv2.cvtColor(debug_image,cv2.COLOR_BGR2RGB)))
        ax6.imshow(debug_image)
        ax6.axis("off")
        
        ax7 = fig.add_subplot(4, 3, 7)
        debug_image = debug_ellipse_on_eye_images[0]
        debug_image = Image.fromarray(np.uint8(cv2.cvtColor(debug_image,cv2.COLOR_BGR2RGB)))
        ax7.imshow(debug_image)
        ax7.axis("off")
        # debug_image = debug_elipse_images[0]
        # debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        # debug_image[np.where(debug_image==np.array([255,255,255]))] = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)[np.where(debug_image==np.array([255,255,255]))]
        # ax7.imshow(debug_image)
        ax8 = fig.add_subplot(4, 3, 8)
        debug_image = debug_ellipse_on_eye_images[1]
        debug_image = Image.fromarray(np.uint8(cv2.cvtColor(debug_image,cv2.COLOR_BGR2RGB)))
        ax8.imshow(debug_image)
        ax8.axis("off")
        ax9 = fig.add_subplot(4, 3, 9)
        debug_image = debug_ellipse_on_eye_images[2]
        debug_image = Image.fromarray(np.uint8(cv2.cvtColor(debug_image,cv2.COLOR_BGR2RGB)))
        ax9.imshow(debug_image)
        ax9.axis("off")
        ax10 = fig.add_subplot(4, 3, 10)
        ax10.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax10.axis("off")
        ax11 = fig.add_subplot(4, 3, 11)
        ax11.imshow(test_frame)
        ax11.axis("off")
        
        plt.tight_layout()
        plt.savefig(f"debug_frame{FRAME_COUNT}.jpg")
        plt.close()

    
    #process_frames now returns a rotated rectangle for the ellipse for easy access
    return final_rotated_rect, test_frame


# Finds the pupil in an individual frame and returns the center point
# def process_frame(frame):

#     # Crop and resize frame
#     frame = crop_to_aspect_ratio(frame)

#     #find the darkest point
#     darkest_point = get_darkest_area(frame)

#     # Convert to grayscale to handle pixel value operations
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
#     # apply thresholding operations at different levels
#     # at least one should give us a good ellipse segment
#     thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
#     thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

#     thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
#     thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
    
#     thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
#     thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
    
#     #take the three images thresholded at different levels and process them
#     final_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, False, False)
    
#     return final_rotated_rect

# Loads a video and finds the pupil in each frame
def process_video(video_path, input_method, gui=True):
    global FRAME_COUNT

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    # out = cv2.VideoWriter('C:/Storage/Source Videos/output_video.mp4', fourcc, 30.0, (640, 480))  # Output video filename, codec, frame rate, and frame size

    if input_method == 1:
        cap = cv2.VideoCapture(video_path)
    elif input_method == 2:
        cap = cv2.VideoCapture(00, cv2.CAP_DSHOW)  # Camera input
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    else:
        print("Invalid video source.")
        return

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    debug_mode_on = False
    
    temp_center = (0,0)
    
    if gui:
    
        #https://watlab-blog.com/2020/07/18/tkinter-frame-pack-grid/
        #https://office54.net/python/tkinter/python-tkinter-button
        # https://qiita.com/yutaka_m/items/f3bb883a5ffc860fcfca
        # rootメインウィンドウの設定
        root = tk.Tk()
        root.title("tkinter application")
        root.geometry(f"{TK_WINDOW_WIDTH}x{TK_WINDOW_HEIGHT}")
        
        # sub window
        canvas_frame = tk.Frame(root, height=IMAGE_H, width=IMAGE_W)
        th_frame = tk.Frame(root, height=100, width=600)
        th_frame = tk.Frame(root, height=100, width=600)
        eye_size_frame = tk.Frame(root, height=100, width=600)
        search_area_frame = tk.Frame(root, height=100, width=600)

        
        apply_frame = tk.Frame(root, height=100, width=600)
        next_frame = tk.Frame(root, height=100, width=600)

        canvas_frame.place(relx=0.05, rely=0.05)
        th_frame.place(relx=0.50, rely=0.1)
        eye_size_frame.place(relx=0.65, rely=0.1)
        search_area_frame.place(relx=0.85, rely=0.1)
        apply_frame.place(relx=0.50, rely=0.8)
        next_frame.place(relx=0.50, rely=0.85)
        
        # canvas frame label
        image_label = tk.Label(
            canvas_frame, text="Movie", bg="white", relief=tk.RIDGE
            )
        image_label.grid(row=0, column=0, sticky=tk.W + tk.E)
        # canvas frame (image)
        canvas = tk.Canvas(root, bg="#deb887", width=IMAGE_W, height=IMAGE_H)
        canvas.grid(row=1, column=0)
      
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_COUNT)
            ret, original_image = cap.read()
            if ret == False:
                FRAME_COUNT += 1
            if ret == True:
                break
        
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = crop_to_aspect_ratio(original_image)
        
        original_rgb_pil = Image.fromarray(original_image)
        # canvas.photo = ImageTk.PhotoImage(original_rgb_pil)
        # canvas_create = canvas.create_image(0,0,anchor='nw', image=canvas.photo)
        # canvas.itemconfig(canvas_create, image= canvas.photo)
        replace_image(canvas=canvas, img_pil=original_rgb_pil)
        
        # 画像の閾値処理の受け取り
        thresholded_image_strict_value = tk.StringVar()
        thresholded_image_strict_entry = tk.Entry(th_frame, bd=5, textvariable=thresholded_image_strict_value)
        thresholded_image_strict_entry.grid(row=1, column=0,pady=10)
        thresholded_image_strict_label = tk.Label(th_frame, bg="lightblue", text="thresholded_image_strict")
        thresholded_image_strict_label.grid(row=0, column=0)
        
        thresholded_image_medium_value = tk.StringVar()
        thresholded_image_medium_entry = tk.Entry(th_frame, bd=5, textvariable=thresholded_image_medium_value)
        thresholded_image_medium_entry.grid(row=4, column=0,pady=10)
        thresholded_image_medium_label = tk.Label(th_frame, bg="lightblue", text="thresholded_image_medium")
        thresholded_image_medium_label.grid(row=3, column=0)
        
        thresholded_image_relaxed_value = tk.StringVar()
        thresholded_image_relaxed_entry = tk.Entry(th_frame, bd=5, textvariable=thresholded_image_relaxed_value)
        thresholded_image_relaxed_entry.grid(row=7, column=0,pady=10)
        thresholded_image_relaxed_label = tk.Label(th_frame, bg="lightblue", text="thresholded_image_relaxed")
        thresholded_image_relaxed_label.grid(row=6, column=0)
        
        
        # 目の大きさ
        eye_size_value = tk.StringVar()
        eye_size_entry = tk.Entry(eye_size_frame, bd=5, textvariable=eye_size_value)
        eye_size_entry.grid(row=1, column=0,pady=10)
        eye_size_label = tk.Label(eye_size_frame, bg="orange", text="eye size")
        eye_size_label.grid(row=0, column=0)

        # darket pointの探索範囲
        search_w_value = tk.StringVar()
        search_w_entry= tk.Entry(search_area_frame, bd=5, textvariable=search_w_value)
        search_w_entry.grid(row=1, column=0,pady=10)
        search_w_label = tk.Label(search_area_frame, bg="green", text="searh width")
        search_w_label.grid(row=0, column=0)

        search_h_value = tk.StringVar()
        search_h_entry= tk.Entry(search_area_frame, bd=5, textvariable=search_h_value)
        search_h_entry.grid(row=3, column=0,pady=10)
        search_h_label = tk.Label(search_area_frame, bg="green", text="searh height")
        search_h_label.grid(row=2, column=0)


                
        # apply 
        button = tk.Button(apply_frame, text = "Apply",command = lambda:bottun_click(thresholded_image_strict_value=thresholded_image_strict_value,thresholded_image_medium_value=thresholded_image_medium_value,thresholded_image_relaxed_value=thresholded_image_relaxed_value,  eye_size_value=eye_size_value,  search_h_value=search_h_value, search_w_value=search_w_value, original_image=original_image, canvas=canvas))
        button.grid(row=0, column=0, pady=10)
        
        next_buttun = tk.Button(next_frame, text="Next", command=lambda:show_next_frame(thresholded_image_strict_value=thresholded_image_strict_value,thresholded_image_medium_value=thresholded_image_medium_value, thresholded_image_relaxed_value=thresholded_image_relaxed_value,eye_size_value=eye_size_value,  search_h_value=search_h_value, search_w_value=search_w_value,  cap=cap, canvas=canvas))
        next_buttun.grid(row=0, column=0)
        
        root.mainloop()
        
        
        cap.release()

    else:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame[:,500:] = 255
            # Crop and resize frame
            frame = crop_to_aspect_ratio(frame)

            #find the darkest point
            darkest_point = get_darkest_area(frame)

            if debug_mode_on:
                darkest_image = frame.copy()
                cv2.circle(darkest_image, darkest_point, 10, (0, 0, 255), -1)
                cv2.imshow('Darkest image patch', darkest_image)

            # Convert to grayscale to handle pixel value operations
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
            
            # apply thresholding operations at different levels
            # at least one should give us a good ellipse segment
            thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
            thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

            thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
            thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
            
            thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
            thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
            
            #take the three images thresholded at different levels and process them
            pupil_rotated_rect, test_image = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, True)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('d') and debug_mode_on == False:  # Press 'q' to start debug mode
                debug_mode_on = True
            elif key == ord('d') and debug_mode_on == True:
                debug_mode_on = False
                cv2.destroyAllWindows()
            if key == ord('q'):  # Press 'q' to quit
                # out.release()
                break   
            elif key == ord(' '):  # Press spacebar to start/stop
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # Press spacebar again to resume
                        break
                    elif key == ord('q'):  # Press 'q' to quit
                        break

        cap.release()
        # out.release()
        cv2.destroyAllWindows()

#Prompts the user to select a video file if the hardcoded path is not found
#This is just for my debugging convenience :)
def select_video(video_path):
    # root = tk.Tk()
    # root.withdraw()  # Hide the main window
    # video_path = 'C:/Google Drive/Eye Tracking/fulleyetest.mp4'
    if not os.path.exists(video_path):
        print("No file found at hardcoded path. Please select a video file.")
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi")])
        if not video_path:
            print("No file selected. Exiting.")
            return
            
    #second parameter is 1 for video 2 for webcam
    process_video(video_path, 1)


# unset SESSION_MANAGER
# python OrloskyPupilDetector.py /home/demo/Downloads/241208cut2.mov
if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('video_path', help='video_path')   
    args = parser.parse_args()    
    
    video_path = args.video_path
    select_video(video_path=video_path)


