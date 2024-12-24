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
from tqdm import tqdm

random.seed(314)

MAX_RANSAC_TIME = 1000
ELLIPSE_POINT_NUM = 10
DIFF_TH = 20
TH = 30
THETA_NUM = 50
RADISU_NUM  = 30

DRAW_CENTER_SIZE = 100


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


def fit_ellipse_ransac(picked_points_all_direction, gray):
    H, W = gray.shape
    assert len(picked_points_all_direction) >= 5
    ellipse_candidates = []
    fitted_points = []
    centers = []
    indexes = np.arange(len(picked_points_all_direction)).tolist()
    for _ in range(MAX_RANSAC_TIME):
        pick_index = random.sample(indexes, ELLIPSE_POINT_NUM)
        points = picked_points_all_direction[pick_index]
        points = points.astype(np.float32)
       
        ellipse = cv2.fitEllipse(points)
        # print(ellipse)
        (cx, cy), (h, w), deg = ellipse
        a = max(h, w)
        b = min(h, w)
        f = (a - b) / a
        if f > 0.3:
            continue
        if a > W // 2:
            continue
    
        
        ellipse_candidates.append(ellipse)
        fitted_points.append(points)
        centers.append(np.array([cx, cy]))
    
    return ellipse_candidates, fitted_points, np.array(centers)


if __name__ == "__main__":
    plt.rcParams["font.size"] = 20
    video_path = "241208cut2_crop_trim22m_b01_c18.mov"
    save_image_folder_path = os.path.basename(video_path).replace(".","")  + "result"
    os.makedirs(save_image_folder_path, exist_ok=True)
    
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

    # video_path = "241208cut2_crop.mov"
    darkest_points_list = [[240, 230]]
    cap = cv2.VideoCapture(video_path)
    total_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if ret == True:
            

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (600, 600))
    
            original_image = gray[120:520,60:440]
            gaussian_kernel_size = 3
            blured_image = cv2.GaussianBlur(original_image, ksize=(gaussian_kernel_size, gaussian_kernel_size), sigmaX=0)
            sobel_combined = make_sobel_image(sobel_kernel=-1, blured_image=blured_image)
            erode_kernel = np.ones((3, 3), np.uint8)
            erode = cv2.erode(sobel_combined, erode_kernel, iterations=1)
            dilate = cv2.dilate(erode, erode_kernel, iterations=3)

            darkest_point = darkest_points_list[-1]
            
            if frame_count != 0:
                initial_darkest_point = darkest_points_list[0]
                dist = np.linalg.norm(np.array(initial_darkest_point)[None,...] - np.array(darkest_point)[None,...], ord=2, axis=1)[0]
                center_dist_th = math.sqrt((original_image.shape[0]//2)**2 + (original_image.shape[1]//2)**2) // 7
                # print(frame_count, dist, center_dist_th)
                if center_dist_th < dist:
                    darkest_point = darkest_points_list[0]
                    print(dist, center_dist_th)
         
            
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
                ray_points = get_points(gray=dilate, start=darkest_point, theta=theta)
                value_points = get_values(gray=dilate, ray_points=ray_points)
                diff_points = take_differencial(original_points=value_points)
                
                picked_points = ray_points[np.abs(diff_points) >=DIFF_TH]
                # picked_points = ray_points[diff_points >=diff_th]
                
                if len(picked_points) > 0:
                    picked_points_all_direction.append(picked_points[0])
                
                
    
    
                
                
            
            picked_points_all_direction = np.array(picked_points_all_direction)
            ellipse_candidates, fitted_points, centers = fit_ellipse_ransac(picked_points_all_direction=picked_points_all_direction, gray=sobel_combined)

            best = np.argmin(np.linalg.norm((centers - np.array(darkest_point)[None, ...]), axis=1))
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
            darkest_points_list.append(next_darkest_point)

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
                ray_points = get_points(gray=dilate, start=darkest_point, theta=theta)
                value_points = get_values(gray=dilate, ray_points=ray_points)
                diff_points = take_differencial(original_points=value_points)
                picked_points = ray_points[np.abs(diff_points) >=DIFF_TH] 
                # picked_points = ray_points[np.abs(diff_points) >=diff_th] 
                draw_picked_points = picked_points[(picked_points[:,0] > 0) & (picked_points[:,0]<original_image.shape[1]) & (picked_points[:,1] < original_image.shape[0]) & (picked_points[:, 1] > 0)]
                if len(picked_points) > 0:
                    ax6.scatter(draw_picked_points[:, 0],(original_image.shape[0]- draw_picked_points[:, 1]), color="yellow")
                    ax6.scatter(picked_points[0, 0], (original_image.shape[0] - picked_points[0, 1]), color="red")
            ax6.scatter(darkest_point[0],(original_image.shape[0] - darkest_point[1]) ,c="magenta", s=DRAW_CENTER_SIZE)
            ax6.set_xlim([0, original_image.shape[1]])
            ax6.set_ylim([0, original_image.shape[0]])

            ax6.axis("off")      
            ax6.set_title(f"Differentiate in radical direction")     

            ax7 = fig.add_subplot(3, 3, 6)
            ax7.imshow(all_candidate_image)
            ax7.axis("off")
            ax7.set_title(f"Candidates")     
            plt.tight_layout()
            
            save_path = os.path.join(save_image_folder_path, f"debug_frame{frame_count:04d}.jpg")
            plt.savefig(save_path)
            # plt.show()
            plt.close()
            
            print(total_frame_num, frame_count)

     
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.imshow("img", frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
     
    #     key =cv2.waitKey(10)
    #     if key == 27:
    #         break
            frame_count += 1
        if frame_count >= total_frame_num:
            break 
    cap.release()    
    
    

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
    darkest_point = [240, 230]
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
    ax7.scatter(darkest_point[0], darkest_point[1], c="red")
    for theta in thetas:
        ray_points = get_points(gray=sobel_combined, start=darkest_point, theta=theta)
        ax7.scatter(ray_points[:, 0], ray_points[:, 1], color="cyan")
        value_points = get_values(gray=dilate, ray_points=ray_points)
    
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(sobel_combined, cmap="gray")
    ax8.scatter(darkest_point[0], darkest_point[1], c="red")
    for theta in thetas:
        ray_points = get_points(gray=dilate, start=darkest_point, theta=theta)
        value_points = get_values(gray=dilate, ray_points=ray_points)
        TH = 30
        picked_points = ray_points[value_points >=TH]
        ax8.scatter(picked_points[:, 0], picked_points[:, 1], color="magenta")
    

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.imshow(sobel_combined, cmap="gray")
    ax9.scatter(darkest_point[0], darkest_point[1], c="red")
    for theta in thetas:
        ray_points = get_points(gray=dilate, start=darkest_point, theta=theta)
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

    best = np.argmin(np.linalg.norm((centers - np.array(darkest_point)[None, ...]), axis=1))
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