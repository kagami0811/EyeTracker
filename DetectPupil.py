import os
import pycalib
import cv2
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from starburst import StarBurst
import math
import torch
import torch.nn.functional as F
import copy
import random
from kalman import KalmanTracker
from tqdm import tqdm

import sys


random.seed(314)

MAX_RANSAC_TIME = 5000
print(sys.argv)
THETA_NUM = int(sys.argv[3]) # 50
# ELLIPSE_POINT_NUM = 40

ELLIPSE_POINT_NUM = int(sys.argv[1])

DIFF_TH = 20
TH = 30

OMEGA_NUM = 200
RADISU_NUM  = 30
BEST_CHOOSE = 10

DRAW_CENTER_SIZE = 100

USE_MSE = True
USE_PIKEL = True
REFINE = sys.argv[2]
if REFINE == "True":
    REFINE = True
else:
    REFINE = False
MAX_INFERENCE_FRMAE = 100
RANSAC_MSE_TH = 10 # ransacの正常値を決める閾値


def blur_image(original_image, type, kernel_size=3):
    if type=="median":
        blured_image = cv2.medianBlur(original_image, ksize=3)
    if type == "gaussian":
        blured_image = cv2.GaussianBlur(original_image, ksize=(kernel_size, kernel_size), sigmaX=0)
    
    return blured_image

def make_sobel_image(sobel_kernel, blured_image):

  
    sobel_x = cv2.Sobel(blured_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)               # 水平方向の勾配
    sobel_y = cv2.Sobel(blured_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)               # 垂直方向の勾配
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    # equalize  = cv2.equalizeHist(sobel_combined)
    sobel_combined_normalize = sobel_combined.astype(np.float32)
    sobel_combined_normalize = (sobel_combined_normalize - np.min(sobel_combined_normalize)) / (np.max(sobel_combined_normalize) - np.min(sobel_combined_normalize))

    return sobel_combined




def get_points(gray, start, theta):
    h, w = gray.shape
    max_l = math.sqrt((h - start[1])**2 + (w - start[0])**2)
    lengths = np.linspace(0, max_l, RADISU_NUM)
    ray_section = lengths * np.exp(1j * theta)
    points = np.concatenate([ray_section.real[:, None], ray_section.imag[:, None]], axis=1) 
    ray_points = points + np.array(start)[None, ...]
    
    return ray_points

def get_values(gray, ray_points):

    h, w = gray.shape
    ray_points_ = copy.deepcopy(ray_points)
    ray_points_[:, 0] = (ray_points[:, 0] - ( w//2)) / (w //2)
    ray_points_[:, 1] = (ray_points[:, 1] - ( h // 2)) / ( h //2)
    ray_points_ = ray_points_.reshape(-1,2).astype(np.float32)
    input = torch.from_numpy(gray.astype(np.float32)).reshape(1,1,h,w)
    grid = torch.from_numpy(ray_points_).reshape(1,1,-1,2)
    #https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    output = F.grid_sample(input=input, grid=grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    output = output.detach().numpy().copy().reshape(-1)
    

    return output


def take_differencial(original_points):
    rotated_points = np.roll(original_points, shift=1)
    diff = original_points - rotated_points
    diff[0] = 0
    return diff


def ransac(picked_points_all_direction, gray, sobel, ellipse_point_num=ELLIPSE_POINT_NUM):
    H, W = gray.shape
    assert len(picked_points_all_direction) >= 5
    good_elipses = [] # ransacで基準を満たすものの
    fit_mse_scores = [] # ransacで基準を満たすものの候補のmse
    surrounding_values = [] # ransacで基準を満たす楕円の中でsobelを積分した値
    fitted_points = [] # ransacに使った点
    
    assert len(picked_points_all_direction) >= 5

    indexes = np.arange(len(picked_points_all_direction)).tolist()

    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
   
    
    for _ in range(MAX_RANSAC_TIME):
        pick_index = random.sample(indexes, ellipse_point_num)
        points = picked_points_all_direction[pick_index]
        points = points.astype(np.float32)
       
        ellipse = cv2.fitEllipse(points)
        (cx, cy), (w, h), deg = ellipse
        a = max(h, w)
        b = min(h, w)
        f = (a - b) / a
        if f > 0.3:
            continue
        if a > 1 *(W // 2):
            continue
        # error 

        # good_elipses.append(ellipse)
        # fitted_points.append(points)
        
        #描画して確認
        # image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # image = cv2.ellipse(image,ellipse,(255,255,0),2)
        omegas = np.linspace(0, 2*np.pi, OMEGA_NUM)
        w_ = w //2
        h_ = h//2
        deg_ = deg * np.pi/180
        ellipse_x = (w_ * np.cos(omegas) * np.cos(deg_)) - (h_ * np.sin(omegas) * np.sin(deg_)) + cx
        ellipse_y = (w_ * np.cos(omegas) * np.sin(deg_)) + (h_ * np.sin(omegas) * np.cos(deg_)) + cy

        ellpise_surrounding = np.concatenate([ellipse_x[:,None], ellipse_y[:,None]], axis=1)
        
        # 全データサンプルを用いて誤差の小さいもので再度　fitting
        normal_value_points = []
        for point in picked_points_all_direction:
            distance =  np.linalg.norm((ellpise_surrounding-point), axis=1)
            min_dist = np.min(distance)
            if min_dist <= RANSAC_MSE_TH:
                normal_value_points.append(point)
        
        normal_value_points = np.array(normal_value_points).astype(np.float32)
        if len(normal_value_points) < 5:
            continue
        ellipse = cv2.fitEllipse(normal_value_points)
        (cx, cy), (w, h), deg = ellipse
        a = max(h, w)
        b = min(h, w)
        f = (a - b) / a
        if f > 0.3:
            continue
        if a > 1 *(W // 2):
            continue
        
        good_elipses.append(ellipse)
        fitted_points.append(normal_value_points)
        
        omegas = np.linspace(0, 2*np.pi, OMEGA_NUM)
        w_ = w //2
        h_ = h//2
        deg_ = deg * np.pi/180
        ellipse_x = (w_ * np.cos(omegas) * np.cos(deg_)) - (h_ * np.sin(omegas) * np.sin(deg_)) + cx
        ellipse_y = (w_ * np.cos(omegas) * np.sin(deg_)) + (h_ * np.sin(omegas) * np.cos(deg_)) + cy

        ellpise_surrounding = np.concatenate([ellipse_x[:,None], ellipse_y[:,None]], axis=1)
        
        
        # fit した楕円とのmseを計算
        min_distance_list = []
        # for point in points:
        for point in normal_value_points:
            distance = np.linalg.norm((ellpise_surrounding-point), axis=1)
            min_dist = np.min(distance)
            min_distance_list.append(min_dist)
            # image = cv2.circle(image, (int(point[0]), int(point[1])), 2,(0,0,255),-1)
            # cv2.imwrite("debug_ellipse.jpg", image)
        mse = np.mean(min_distance_list)
        fit_mse_scores.append(mse)

        # sobel画像のピクセル値を計算(白い方が良い)

        ray_points_ = copy.deepcopy(ellpise_surrounding)
        ray_points_[:, 0] = (ellpise_surrounding[:, 0] - ( W//2)) / (W //2)
        ray_points_[:, 1] = (ellpise_surrounding[:, 1] - ( H // 2)) / ( H //2)
        ray_points_ = ray_points_.reshape(-1,2).astype(np.float32)
        input = torch.from_numpy(sobel.astype(np.float32)).reshape(1,1,H,W)
        grid = torch.from_numpy(ray_points_).reshape(1,1,-1,2)
        #https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        output = F.grid_sample(input=input, grid=grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        output = output.detach().numpy().copy().reshape(-1)
        surrounding_values.append(np.mean(output))
        

        # for plot_x, plot_y in zip(ellipse_x, ellipse_y):

        #     image = cv2.circle(image, (int(plot_x), int(plot_y)), 2,(0,0,255),-1)
        # image = cv2.circle(image, (int(cx), int(cy)), 2,(255,0,255),-1)
        # cv2.imwrite("debug_ellipse.jpg", image)
        # import pdb; pdb.set_trace()

    min_mse_candidate_indexes = np.argsort(fit_mse_scores)[:BEST_CHOOSE]
    # print(fit_mse_scores[min_mse_candidate_indexes[0]],fit_mse_scores[min_mse_candidate_indexes[1]],fit_mse_scores[min_mse_candidate_indexes[2]] )

    high_value_candidate_indexes = np.argsort(surrounding_values)[::-1][:BEST_CHOOSE]
    # print(surrounding_values[high_value_candidate_indexes[0]],surrounding_values[high_value_candidate_indexes[1]],surrounding_values[high_value_candidate_indexes[2]])
    if USE_PIKEL and USE_MSE:
        good_candidates = list(set((list(min_mse_candidate_indexes) + list(high_value_candidate_indexes))))
    elif USE_MSE:
        good_candidates = list(min_mse_candidate_indexes)
    elif USE_PIKEL:
        good_candidates = list(high_value_candidate_indexes)

    # print(len(good_candidates))

    final_candidates = []
    final_candidate_center = []
    final_candidate_major = []
    final_candidate_minor = []
    final_fitted_points = []
    for candidate_ind in good_candidates:
        draw_ellipse = good_elipses[candidate_ind]
        (cx, cy), (w, h), deg = draw_ellipse
        final_candidate_center.append([cx, cy])
        final_candidate_major.append(w)
        final_candidate_minor.append(h)
        final_candidates.append(draw_ellipse)
        final_points = fitted_points[candidate_ind]
        final_fitted_points.append(final_points)
        image = cv2.ellipse(image,draw_ellipse,(255,255,0),1)
    
    # cv2.imwrite("debug_ellipse.jpg", image)
   
    # print(final_candidates)
    # import pdb; pdb.set_trace()
    final_candidate_center = np.array(final_candidate_center)
    final_candidate_major = np.array(final_candidate_major)
    final_candidate_minor = np.array(final_candidate_minor)
    return final_candidates, final_candidate_center, final_candidate_major, final_candidate_minor, image, final_fitted_points
    


# def fit_ellipse_ransac(picked_points_all_direction, gray):
#     H, W = gray.shape
#     assert len(picked_points_all_direction) >= 5
#     ellipse_candidates = []
#     fitted_points = []
#     centers = []
#     indexes = np.arange(len(picked_points_all_direction)).tolist()
#     for _ in range(MAX_RANSAC_TIME):
#         pick_index = random.sample(indexes, ELLIPSE_POINT_NUM)
#         points = picked_points_all_direction[pick_index]
#         points = points.astype(np.float32)
       
#         ellipse = cv2.fitEllipse(points)
#         # print(ellipse)
#         (cx, cy), (h, w), deg = ellipse
#         a = max(h, w)
#         b = min(h, w)
#         f = (a - b) / a
#         if f > 0.3:
#             continue
#         if a > W // 2:
#             continue
        
#         ellipse_candidates.append(ellipse)
#         fitted_points.append(points)
#         centers.append(np.array([cx, cy]))
    
#     return ellipse_candidates, fitted_points, np.array(centers)


def select_best(final_candidate_center, final_candidate_major, final_candidate_minor,final_ellipse_candidates, final_candidate_fitted_points, pupil_center,pupil_major, pupil_minor, original_image):

    dist_center = np.linalg.norm(final_candidate_center-pupil_center[None,:], axis=1)
    dist_major = np.abs(final_candidate_major-pupil_major)
    dist_minor = np.abs(final_candidate_minor-pupil_minor)

    dist_sum = dist_center + dist_major + dist_minor
    best_ind = np.argmin(dist_sum)
    observe_center = final_candidate_center[best_ind]
    observe_major = final_candidate_major[best_ind]
    observe_minor = final_candidate_minor[best_ind]
    observe_ellipse = final_ellipse_candidates[best_ind]
    observe_points = final_candidate_fitted_points[best_ind]
    

    # image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    result_image= copy.deepcopy(original_image)
   
    result_image =  cv2.ellipse(result_image,observe_ellipse,(255,255,0),2)
    result_image = cv2.circle(result_image, (int(observe_center[0]), int(observe_center[1])), 5,(0,255,0),-1) # green

    

    for point in observe_points:
        result_image = cv2.circle(result_image, (int(point[0]), int(point[1])), 3,(0,0,255),-1)
   


    return observe_center, observe_major, observe_minor, observe_ellipse, result_image, observe_points



def refine_ransac(candidate_points, gray, sobel, pupil_center, pupil_major, pupil_minor):
    final_ellipse_candidates, final_candidate_center, final_candidate_major, final_candidate_minor, image, final_candidate_fitted_points = ransac(picked_points_all_direction=candidate_points, gray=gray, sobel=sobel, ellipse_point_num=int(ELLIPSE_POINT_NUM*0.7))

    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    for point in candidate_points:
        result_image = cv2.circle(result_image, (int(point[0]), int(point[1])), 2,(0,255,255),-1)


    observe_center, observe_major, observe_minor, observe_ellipse, result_image, observe_points = select_best(final_candidate_center=final_candidate_center, final_candidate_major=final_candidate_major, final_candidate_minor=final_candidate_minor, final_ellipse_candidates=final_ellipse_candidates, final_candidate_fitted_points=final_candidate_fitted_points, pupil_center=pupil_center, pupil_major=pupil_major, pupil_minor=pupil_minor, original_image=result_image)

    return observe_center, observe_major, observe_minor, observe_ellipse, result_image, observe_points
    



def refine(candidate_points, gray, sobel):

    ellipse = cv2.fitEllipse(candidate_points)
    # print(ellipse)
    (cx, cy), (w, h), deg = ellipse
    a = max(h, w)
    b = min(h, w)
    f = (a - b) / a

    omegas = np.linspace(0, 2*np.pi, OMEGA_NUM)
    w_ = w //2
    h_ = h//2
    deg_ = deg * np.pi/180
    ellipse_x = (w_ * np.cos(omegas) * np.cos(deg_)) - (h_ * np.sin(omegas) * np.sin(deg_)) + cx
    ellipse_y = (w_ * np.cos(omegas) * np.sin(deg_)) + (h_ * np.sin(omegas) * np.cos(deg_)) + cy

    ellpise_surrounding = np.concatenate([ellipse_x[:,None], ellipse_y[:,None]], axis=1)
    # fit した楕円とのmseを計算
    min_distance_list = []
    for point in candidate_points:
        distance = np.linalg.norm((ellpise_surrounding-point), axis=1)
        min_dist = np.min(distance)
        min_distance_list.append(min_dist)
        # image = cv2.circle(image, (int(point[0]), int(point[1])), 2,(0,0,255),-1)
        # cv2.imwrite("debug_ellipse.jpg", image)
    median = np.median(min_distance_list)
    good_fitting_points = candidate_points[np.where(min_distance_list < 2 * median)[0]]
    
    assert len(good_fitting_points) > 5
    refined_ellipse =  cv2.fitEllipse(good_fitting_points)
    (cx, cy), (w, h), deg = refined_ellipse

    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image = cv2.ellipse(image,refined_ellipse,(255,255,0),2)
    for point in candidate_points:
       
        image = cv2.circle(image, (int(point[0]), int(point[1])), 2,(0,255,255),-1) # yellow
    for point in good_fitting_points:
        image = cv2.circle(image, (int(point[0]), int(point[1])), 3,(0,0,255),-1) # red
    image = cv2.circle(image, (int(cx), int(cy)), 5,(0,255,0),-1) # green

    

    cv2.imwrite("debug_refine.jpg", image)
   
    return [cx, cy], w, h, refined_ellipse, image, good_fitting_points


# ffmpeg -i 241208cut2_crop_trim22m_b01_c18movresult_0107_normal/result_50.mp4 -vf crop=w=800:h=560:x=0:y=0 241208cut2_crop_trim22m_b01_c18movresult_0107_normal/result_50_crop.mp4

# ffmpeg -i 241208cut2_crop_trim22m_b01_c18movrevise_ransac_result_0107_normal_ellipse20_100/result_100.mp4 -vf crop=w=800:h=560:x=0:y=0 241208cut2_crop_trim22m_b01_c18movrevise_ransac_result_0107_normal_ellipse20_100/result_20_100_crop.mp4

if __name__ == "__main__":
    plt.rcParams["font.size"] = 20
    video_path = "241208cut2_crop_trim22m_b01_c18.mov"
    if REFINE:
        subname = "refine"
    else:
        subname = "normal"
    save_image_folder_path = os.path.basename(video_path).replace(".","")  + f"revise_ransac_result_0107_{subname}_ellipse{ELLIPSE_POINT_NUM}_{THETA_NUM}"
    os.makedirs(save_image_folder_path, exist_ok=True)
    
    



    # video_path = "241208cut2_crop.mov"
    
    pupil_center_points_list = [[240, 230]]
    pupil_major_minor_list= [[125, 135]]

    pupil_center_tracker = KalmanTracker(result=pupil_center_points_list[0])
    pupil_major_minor_tracker = KalmanTracker(result=pupil_major_minor_list[0])
    
    cap = cv2.VideoCapture(video_path)
    total_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if ret == True:
            

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (600, 600))
    
            original_image = gray[120:520,60:440]
            # gaussian_kernel_size = 3
            # blured_image = cv2.GaussianBlur(original_image, ksize=(gaussian_kernel_size, gaussian_kernel_size), sigmaX=0)
            kernel_size = 3
            blured_image = blur_image(original_image=original_image, type="median", kernel_size=kernel_size)
            sobel_combined = make_sobel_image(sobel_kernel=-1, blured_image=blured_image)
            erode_kernel = np.ones((3, 3), np.uint8)
            erode = cv2.erode(sobel_combined, erode_kernel, iterations=1)
            dilate = cv2.dilate(erode, erode_kernel, iterations=3)

            # pupil_center = pupil_center_points_list[-1]
            
            # if frame_count != 0:
            #     initial_pupil_center_point = pupil_center_points_list[0]
            #     dist = np.linalg.norm(np.array(initial_pupil_center_point)[None,...] - np.array(pupil_center)[None,...], ord=2, axis=1)[0]
            #     center_dist_th = math.sqrt((original_image.shape[0]//2)**2 + (original_image.shape[1]//2)**2) // 7
            #     # print(frame_count, dist, center_dist_th)
            #     if center_dist_th < dist:
            #         pupil_center = pupil_center_points_list[0]
            #         print(dist, center_dist_th)

            pupil_center = pupil_center_tracker.predict()
            pupil_major,pupil_minor = pupil_major_minor_tracker.predict()
           
            print(pupil_center, pupil_major, pupil_minor)
            thetas = np.linspace(0, 2*np.pi, THETA_NUM)
            picked_points_all_direction = []
            
            # fig =  plt.figure(figsize=(24, 24))
            # ax1= fig.add_subplot(1,1,1)
            # for theta in thetas:
            #     ray_points = get_points(gray=dilate, start=darkest_point, theta=theta)
            #     value = get_values(gray=dilate, ray_points=ray_points)
            #     diff_points = take_differencial(original_points=value)
            #     length = np.arange(len(value))
                # ax1.scatter(length, diff_points)
                # ax1.scatter(length, value)
                # plt.savefig("debug_plot.jpg")
                # plt.show()
            # plt.close()
                
                
                
            
            for theta in thetas:
                ray_points = get_points(gray=dilate, start=pupil_center, theta=theta)
                value_points = get_values(gray=dilate, ray_points=ray_points)
                diff_points = take_differencial(original_points=value_points)
                
                picked_points = ray_points[np.abs(diff_points) >=DIFF_TH]
                # picked_points = ray_points[diff_points >=diff_th]
                
                if len(picked_points) > 0:
                    picked_points_all_direction.append(picked_points[0])
                
                
    
    
                
                
            
            picked_points_all_direction = np.array(picked_points_all_direction)

            final_ellipse_candidates, final_candidate_center, final_candidate_major, final_candidate_minor, candidate_draw_image, final_candidate_fitted_points = ransac(picked_points_all_direction=picked_points_all_direction, gray=original_image, sobel=sobel_combined)

            
            #候補がないとき
            if len(final_candidate_center) == 0:
                print(frame_count)

                fig =  plt.figure(figsize=(24, 24))
                ax1 = fig.add_subplot(3, 3, 1)
                ax1.imshow(original_image, cmap="gray")
                ax1.axis("off")
                ax1.set_title(f"Frame {frame_count} original image")
                ax2 = fig.add_subplot(3, 3, 2)
                ax2.imshow(blured_image, cmap="gray")
                ax2.axis("off")
                ax2.set_title(f"Blurred image ")
                ax3 = fig.add_subplot(3, 3, 3)
                # result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                # ax3.imshow(result_image)
                ax3.axis("off")
                ax3.set_title(f"Detection Result")
                ax4 = fig.add_subplot(3, 3, 4)
                ax4.imshow(sobel_combined, cmap="gray")
                ax4.axis("off")
                ax4.set_title(f"Schor(Sobel) filtered image")
            

                ax6 = fig.add_subplot(3, 3, 5)
                ax6.imshow(cv2.flip(sobel_combined, 0), cmap="gray")
                # ax6.imshow(sobel_combined, cmap="gray")

                for theta in thetas:
                    ray_points = get_points(gray=dilate, start=pupil_center, theta=theta)
                    value_points = get_values(gray=dilate, ray_points=ray_points)
                    diff_points = take_differencial(original_points=value_points)
                    picked_points = ray_points[np.abs(diff_points) >=DIFF_TH] 
                    # picked_points = ray_points[np.abs(diff_points) >=diff_th] 
                    draw_picked_points = picked_points[(picked_points[:,0] > 0) & (picked_points[:,0]<original_image.shape[1]) & (picked_points[:,1] < original_image.shape[0]) & (picked_points[:, 1] > 0)]
                    if len(picked_points) > 0:
                        ax6.scatter(draw_picked_points[:, 0],(original_image.shape[0]- draw_picked_points[:, 1]), color="yellow")
                        ax6.scatter(picked_points[0, 0], (original_image.shape[0] - picked_points[0, 1]), color="red")
                ax6.scatter(pupil_center[0],(original_image.shape[0] - pupil_center[1]) ,c="magenta", s=DRAW_CENTER_SIZE)

                #     if len(picked_points) > 0:
                #         ax6.scatter(draw_picked_points[:, 0], draw_picked_points[:, 1], color="yellow")
                #         ax6.scatter(picked_points[0, 0],  picked_points[0, 1], color="red")
                # ax6.scatter(pupil_center[0],pupil_center[1] ,c="magenta", s=DRAW_CENTER_SIZE)
                ax6.set_xlim([0, original_image.shape[1]])
                ax6.set_ylim([0, original_image.shape[0]])

                ax6.axis("off")      
                ax6.set_title(f"Differentiate in radical direction")     

                plt.tight_layout()
                save_path = os.path.join(save_image_folder_path, f"debug_frame{frame_count:04d}_median_kernel{kernel_size}_fault.jpg")
                plt.savefig(save_path)
                plt.close()
                frame_count += 1
                if frame_count >= total_frame_num:
                    break 
                if frame_count >= MAX_INFERENCE_FRMAE:
                    break

                continue


            # bestのものを選ぶ
            dist_center = np.linalg.norm(final_candidate_center-pupil_center[None,:], axis=1)
            dist_major = np.abs(final_candidate_major-pupil_major)
            dist_minor = np.abs(final_candidate_minor-pupil_minor)

            dist_sum = dist_center + dist_major + dist_minor
            best_ind = np.argmin(dist_sum)
            observe_center = final_candidate_center[best_ind]
            observe_major = final_candidate_major[best_ind]
            observe_minor = final_candidate_minor[best_ind]
            observe_ellipse = final_ellipse_candidates[best_ind]
            observe_points = final_candidate_fitted_points[best_ind]

            # # refine bestのもので誤差の大きいものだけ除く
            # if REFINE:
            #     observe_center, observe_major, observe_minor, observe_ellipse, result_image,observe_points = refine(candidate_points=observe_points, gray=original_image, sobel=sobel_combined)

            # else:

            #     image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            #     result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #     result_image =  cv2.ellipse(result_image,observe_ellipse,(255,255,0),2)
            #     for point in observe_points:
            #         result_image = cv2.circle(result_image, (int(point[0]), int(point[1])), 2,(255,0,255),-1)
            #     result_image = cv2.circle(result_image, (int(observe_center[0]), int(observe_center[1])), 5, (0,255,0),-1)

            

            if REFINE:
                observe_center, observe_major, observe_minor, observe_ellipse, result_image, observe_points = refine_ransac(candidate_points=observe_points, gray=original_image, sobel=sobel_combined, pupil_center=pupil_center, pupil_major=pupil_major, pupil_minor=pupil_minor)
            # cv2.imwrite("debug_result.jpg", result_image)


            else:
                image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result_image =  cv2.ellipse(result_image,observe_ellipse,(255,255,0),2)
                for point in observe_points:
                    result_image = cv2.circle(result_image, (int(point[0]), int(point[1])), 3,(0,0,255),-1)
                result_image = cv2.circle(result_image, (int(observe_center[0]), int(observe_center[1])), 5, (0,255,0),-1)

            #observe とpredictが離れすぎているときはアップデートしない
            prediction_dist = np.linalg.norm(observe_center - pupil_center)
            if prediction_dist < 40:
                # update
                pupil_center_tracker.update(new_posx=observe_center[0], new_posy=observe_center[1])
                pupil_major_minor_tracker.update(new_posx=observe_major, new_posy=observe_minor)

                pupil_center_points_list.append(observe_center)
            else:
                print(frame_count, prediction_dist)


            fig =  plt.figure(figsize=(24, 24))
            ax1 = fig.add_subplot(3, 3, 1)
            ax1.imshow(original_image, cmap="gray")
            ax1.axis("off")
            ax1.set_title(f"Frame {frame_count} original image")
            ax2 = fig.add_subplot(3, 3, 2)
            ax2.imshow(blured_image, cmap="gray")
            ax2.axis("off")
            ax2.set_title(f"Blurred image ")
            ax3 = fig.add_subplot(3, 3, 3)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            ax3.imshow(result_image)
            ax3.axis("off")
            ax3.set_title(f"Detection Result")
            ax4 = fig.add_subplot(3, 3, 4)
            ax4.imshow(sobel_combined, cmap="gray")
            ax4.axis("off")
            ax4.set_title(f"Schor(Sobel) filtered image")
           

            ax6 = fig.add_subplot(3, 3, 5)
            ax6.imshow(cv2.flip(sobel_combined, 0), cmap="gray")
            for theta in thetas:
                ray_points = get_points(gray=dilate, start=pupil_center, theta=theta)
                value_points = get_values(gray=dilate, ray_points=ray_points)
                diff_points = take_differencial(original_points=value_points)
                picked_points = ray_points[np.abs(diff_points) >=DIFF_TH] 
                # picked_points = ray_points[np.abs(diff_points) >=diff_th] 
                draw_picked_points = picked_points[(picked_points[:,0] > 0) & (picked_points[:,0]<original_image.shape[1]) & (picked_points[:,1] < original_image.shape[0]) & (picked_points[:, 1] > 0)]
                if len(picked_points) > 0:
                    ax6.scatter(draw_picked_points[:, 0],(original_image.shape[0]- draw_picked_points[:, 1]), color="yellow")
                    ax6.scatter(picked_points[0, 0], (original_image.shape[0] - picked_points[0, 1]), color="red")
            ax6.scatter(pupil_center[0],(original_image.shape[0] - pupil_center[1]) ,c="magenta", s=DRAW_CENTER_SIZE)

            #     if len(picked_points) > 0:
            #         ax6.scatter(draw_picked_points[:, 0], draw_picked_points[:, 1], color="yellow")
            #         ax6.scatter(picked_points[0, 0],  picked_points[0, 1], color="red")
            # ax6.scatter(pupil_center[0],pupil_center[1] ,c="magenta", s=DRAW_CENTER_SIZE)

            ax6.set_xlim([0, original_image.shape[1]])
            ax6.set_ylim([0, original_image.shape[0]])

            ax6.axis("off")      
            ax6.set_title(f"Differentiate in radical direction")     

            ax7 = fig.add_subplot(3, 3, 6)
            candidate_draw_image = cv2.cvtColor(candidate_draw_image, cv2.COLOR_BGR2RGB)
            ax7.imshow(candidate_draw_image)
            ax7.axis("off")
            ax7.set_title(f"Candidates")     
            plt.tight_layout()
            
            save_path = os.path.join(save_image_folder_path, f"debug_frame{frame_count:04d}_median_kernel{kernel_size}.jpg")
            plt.savefig(save_path)
            # plt.show()
            plt.close()

            frame_count +=1
            if frame_count >= total_frame_num:
                break 
            if frame_count >= MAX_INFERENCE_FRMAE:
                break
          


            continue
   
    
            

            ellipse_candidates, fitted_points, centers = fit_ellipse_ransac(picked_points_all_direction=picked_points_all_direction, gray=sobel_combined)




            

            best = np.argmin(np.linalg.norm((centers - np.array(pupil_center)[None, ...]), axis=1))
            image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            # print(len(ellipse_candidates))
            # for fitted, ellipse in zip(fitted_points, ellipse_candidates):
            #     # image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            #     image = cv2.ellipse(image,ellipse,(0,255,0),1)
            #     fitted = fitted.astype(np.int64)
            #     # for i in range(len(fitted)):
            #     #     image = cv2.circle(image, fitted[i], 5,(0,0,255),-1)
            #     continue
            image = cv2.ellipse(image,ellipse_candidates[best],(255,255,0),2)
            
            result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.circle(image, fitted_points[best], 5,(0,0,255),-1)
            # cv2.imwrite("ellipse_candidate.jpg", image)
            
            all_candidate_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            for fitted, ellipse in zip(fitted_points, ellipse_candidates):
                # image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                all_candidate_image = cv2.ellipse(all_candidate_image,ellipse,(255,255,0),1)
                fitted = fitted.astype(np.int64)
                # for i in range(len(fitted)):
                #     all_candidate_image = cv2.circle(all_candidate_image, fitted[i], 5,(0,0,255),-1)
            
            all_candidate_image = cv2.cvtColor(all_candidate_image, cv2.COLOR_BGR2RGB)
            
            next_darkest_point = ellipse_candidates[best][0]
            pupil_center_points_list.append(next_darkest_point)

            fig =  plt.figure(figsize=(24, 24))
            ax1 = fig.add_subplot(3, 3, 1)
            ax1.imshow(original_image, cmap="gray")
            ax1.axis("off")
            ax1.set_title(f"Frame {frame_count} original image")
            ax2 = fig.add_subplot(3, 3, 2)
            ax2.imshow(blured_image, cmap="gray")
            ax2.axis("off")
            ax2.set_title(f"Blurred image ")
            ax3 = fig.add_subplot(3, 3, 3)
            ax3.imshow(result_image)
            ax3.scatter(next_darkest_point[0], next_darkest_point[1] ,c="magenta", s=DRAW_CENTER_SIZE)
            ax3.axis("off")
            ax3.set_title(f"Detection Result")
            ax4 = fig.add_subplot(3, 3, 4)
            ax4.imshow(sobel_combined, cmap="gray")
            ax4.axis("off")
            ax4.set_title(f"Schor(Sobel) filtered image")
            # ax7 = fig.add_subplot(3, 3, 7)
            # ax7.imshow(blured_image, cmap="gray")
            # ax7.scatter(darkest_point[0], darkest_point[1], c="red")
            # for theta in thetas:
            #     ray_points = get_points(gray=sobel_combined, start=darkest_point, theta=theta)
            #     ax7.scatter(ray_points[:, 0], ray_points[:, 1], color="cyan")
            #     value_points = get_values(gray=dilate, ray_points=ray_points)
            
            # ax5 = fig.add_subplot(3, 3, 5)
            # ax5.imshow(sobel_combined, cmap="gray")
            # ax5.scatter(darkest_point[0], darkest_point[1], c="red")
            # ax4.set_title(f"pixcel image")
            # for theta in thetas:
            #     ray_points = get_points(gray=dilate, start=darkest_point, theta=theta)
            #     value_points = get_values(gray=dilate, ray_points=ray_points)
            #     picked_points = ray_points[value_points >=th]
            #     ax5.scatter(picked_points[:, 0], picked_points[:, 1], color="magenta")
            # ax5.axis("off")

            ax6 = fig.add_subplot(3, 3, 5)
            ax6.imshow(cv2.flip(sobel_combined, 0), cmap="gray")
            for theta in thetas:
                ray_points = get_points(gray=dilate, start=pupil_center, theta=theta)
                value_points = get_values(gray=dilate, ray_points=ray_points)
                diff_points = take_differencial(original_points=value_points)
                picked_points = ray_points[np.abs(diff_points) >=DIFF_TH] 
                # picked_points = ray_points[np.abs(diff_points) >=diff_th] 
                draw_picked_points = picked_points[(picked_points[:,0] > 0) & (picked_points[:,0]<original_image.shape[1]) & (picked_points[:,1] < original_image.shape[0]) & (picked_points[:, 1] > 0)]
                if len(picked_points) > 0:
                    ax6.scatter(draw_picked_points[:, 0],(original_image.shape[0]- draw_picked_points[:, 1]), color="yellow")
                    ax6.scatter(picked_points[0, 0], (original_image.shape[0] - picked_points[0, 1]), color="red")
            ax6.scatter(pupil_center[0],(original_image.shape[0] - pupil_center[1]) ,c="magenta", s=DRAW_CENTER_SIZE)
            ax6.set_xlim([0, original_image.shape[1]])
            ax6.set_ylim([0, original_image.shape[0]])

            ax6.axis("off")      
            ax6.set_title(f"Differentiate in radical direction")     

            ax7 = fig.add_subplot(3, 3, 6)
            ax7.imshow(all_candidate_image)
            ax7.axis("off")
            ax7.set_title(f"Candidates")     
            plt.tight_layout()
            
            save_path = os.path.join(save_image_folder_path, f"debug_frame{frame_count:04d}_median_kernel{kernel_size}.jpg")
            plt.savefig(save_path)
            # plt.show()
            plt.close()
            
            print(total_frame_num, frame_count)
            exit()
     
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.imshow("img", frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
     
    #     key =cv2.waitKey(10)
    #     if key == 27:
    #         break
            frame_count += 1
          
    cap.release()


    # video
    frame_rate = 5
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    size = (800, 800) 
    
    video_file_path = os.path.join(save_image_folder_path, f"result_{THETA_NUM}.mp4")
    writer = cv2.VideoWriter(video_file_path, fmt, frame_rate, size)
    picture_paths = sorted(glob.glob(os.path.join(save_image_folder_path, "*.jpg")))
    for path in picture_paths:
        frame = cv2.imread(path)
        frame = cv2.resize(frame, dsize=size)
        writer.write(frame)
    writer.release()  
    

    exit()
# test
if __name__ == "__main__":
    video_path = "241208cut2_crop_trim22m_b01_c18.mov"
    # video_path = "241208cut2_crop.mov"
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if ret == True:
            

    
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.imshow("img", frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
    #     key =cv2.waitKey(10)
    #     if key == 27:
            break
    cap.release()     

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (600, 600))
    
    original_image = gray[120:520,60:440]
    # https://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    gaussian_kernel_size = 3
    blured_image = cv2.GaussianBlur(original_image, ksize=(gaussian_kernel_size, gaussian_kernel_size), sigmaX=0)
    # blured_image =  cv2.medianBlur(original_image, gaussian_kernel_size)
    # blured_image = original_image
    # blured_image = cv2.equalizeHist(blured_image)
    # sobel_kernel = -1
    # sobel_x = cv2.Sobel(blured_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)               # 水平方向の勾配
    # sobel_y = cv2.Sobel(blured_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)               # 垂直方向の勾配
    # sobel_x = cv2.convertScaleAbs(sobel_x)
    # sobel_y = cv2.convertScaleAbs(sobel_y)
    # sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    # # equalize  = cv2.equalizeHist(sobel_combined)
    # sobel_combined_normalize = sobel_combined.astype(np.float32)
    # sobel_combined_normalize = (sobel_combined_normalize - np.min(sobel_combined_normalize)) / (np.max(sobel_combined_normalize) - np.min(sobel_combined_normalize))
    sobel_combined = make_sobel_image(sobel_kernel=-1, blured_image=blured_image)
    sobel_combined_3 = make_sobel_image(sobel_kernel=3, blured_image=blured_image)
    sobel_combined_5 = make_sobel_image(sobel_kernel=5, blured_image=blured_image)

    # laplacian = cv2.Laplacian(blured_image,cv2.CV_64F )
    erode_kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(sobel_combined, erode_kernel, iterations=1)
    dilate = cv2.dilate(erode, erode_kernel, iterations=3)
    
    
    # cany = cv2.Canny(blured_image, 0, 0)
    # result = hough_ellipse(sobel_combined)
    # result.sort(order='accumulator')
    # print(len(result))
    
    
    # # 画像の中心から
    # h, w = original_image.shape
    # center_x, center_y = w // 2, h // 2
    # search = 120
    # search_min_x = max(0, center_x - search)
    # search_min_y = max(0, center_y - search)
    # search_max_x = min(w, center_x + search)
    # search_max_y = min(h, center_y + search)
    
    # search_image = original_image[search_min_y:search_max_y, search_min_x:search_max_x]
    # darkest_point = np.unravel_index(np.argmin(search_image), search_image.shape)
    pupil_center = [240, 230]
    thetas = np.linspace(0, 2*np.pi, 30)
    picked_points_all_direction = []


    fig =  plt.figure(figsize=(24, 24))
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.imshow(original_image, cmap="gray")
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.imshow(blured_image, cmap="gray")
    # ax3 = fig.add_subplot(3, 3, 3)
    # ax3.imshow(original_image, cmap="gray")
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.imshow(sobel_combined_3, cmap="gray")
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.imshow(sobel_combined_5, cmap="gray")
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.imshow(sobel_combined, cmap="gray")
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.imshow(blured_image, cmap="gray")
    ax7.scatter(pupil_center[0], pupil_center[1], c="red")
    for theta in thetas:
        ray_points = get_points(gray=sobel_combined, start=pupil_center, theta=theta)
        ax7.scatter(ray_points[:, 0], ray_points[:, 1], color="cyan")
        value_points = get_values(gray=dilate, ray_points=ray_points)
    
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(sobel_combined, cmap="gray")
    ax8.scatter(pupil_center[0], pupil_center[1], c="red")
    for theta in thetas:
        ray_points = get_points(gray=dilate, start=pupil_center, theta=theta)
        value_points = get_values(gray=dilate, ray_points=ray_points)
        TH = 30
        picked_points = ray_points[value_points >=TH]
        ax8.scatter(picked_points[:, 0], picked_points[:, 1], color="magenta")
    

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.imshow(sobel_combined, cmap="gray")
    ax9.scatter(pupil_center[0], pupil_center[1], c="red")
    for theta in thetas:
        ray_points = get_points(gray=dilate, start=pupil_center, theta=theta)
        value_points = get_values(gray=dilate, ray_points=ray_points)
        diff_points = take_differencial(original_points=value_points)
        DIFF_TH = 30
        picked_points = ray_points[np.abs(diff_points) >=DIFF_TH]
        ax9.scatter(picked_points[:, 0], picked_points[:, 1], color="yellow")
        ax9.scatter(picked_points[0, 0], picked_points[0, 1], color="red")
        picked_points_all_direction.append(picked_points[0])

    

    # plt.show()
    plt.savefig("debug.jpg")
    plt.close()

    picked_points_all_direction = np.array(picked_points_all_direction)
    ellipse_candidates, fitted_points, centers = fit_ellipse_ransac(picked_points_all_direction=picked_points_all_direction, gray=sobel_combined)

    best = np.argmin(np.linalg.norm((centers - np.array(pupil_center)[None, ...]), axis=1))
    image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    # print(len(ellipse_candidates))
    for fitted, ellipse in zip(fitted_points, ellipse_candidates):
        # image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        image = cv2.ellipse(image,ellipse,(0,255,0),1)
        fitted = fitted.astype(np.int64)
        # for i in range(len(fitted)):
        #     image = cv2.circle(image, fitted[i], 5,(0,0,255),-1)
        continue
    image = cv2.ellipse(image,ellipse_candidates[best],(255,255,0),2)
    # image = cv2.circle(image, fitted_points[best], 5,(0,0,255),-1)
    cv2.imwrite("ellipse_candidate.jpg", image)

        # import pdb; pdb.set_trace()
    # fig =  plt.figure(figsize=(24, 24))
    # ax1= fig.add_subplot(1,1,1)
    # for theta in thetas:
    #     ray_points = get_points(gray=sobel_combined, start=darkest_point, theta=theta)
    #     value = get_values(gray=sobel_combined, ray_points=ray_points)
    #     diff_points = take_differencial(original_points=value_points)
    #     length = np.arange(len(value))
    #     ax1.scatter(length, diff_points)
    #     plt.savefig("debug_plot.jpg")
    #     import pdb; pdb.set_trace()
    # plt.close()