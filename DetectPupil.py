import os
import cv2
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import torch
import torch.nn.functional as F
import copy
import random
from kalman import KalmanTracker
from tqdm import tqdm
from delete_noise_line import delete_line
from config import get_cfg_defaults
import sys
from tqdm import tqdm

random.seed(314)

cfg = get_cfg_defaults()
config_path = sys.argv[1]
cfg.merge_from_file(config_path)


def blur_image(original_image, type, kernel_size=3): 
    '''
    ノイズ除去 
    '''
    if type=="median":
        blured_image = cv2.medianBlur(original_image, ksize=3)
    if type == "gaussian":
        blured_image = cv2.GaussianBlur(original_image, ksize=(kernel_size, kernel_size), sigmaX=0)
    
    return blured_image

def make_sobel_image(sobel_kernel, blured_image):
    '''
    画像を微分
    '''
    sobel_x = cv2.Sobel(blured_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)               # 水平方向の勾配
    sobel_y = cv2.Sobel(blured_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)               # 垂直方向の勾配
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    # equalize  = cv2.equalizeHist(sobel_combined)
    # sobel_combined_normalize = sobel_combined.astype(np.float32)
    # sobel_combined_normalize = (sobel_combined_normalize - np.min(sobel_combined_normalize)) / (np.max(sobel_combined_normalize) - np.min(sobel_combined_normalize))

    return sobel_combined




def get_points(gray, start, theta):
    '''
    中心(start)から放射上の点群を作成
    '''
    h, w = gray.shape
    max_l = math.sqrt((h - start[1])**2 + (w - start[0])**2)
    lengths = np.linspace(0, max_l, cfg.Params.RADIUS_NUM)
    ray_section = lengths * np.exp(1j * theta)
    points = np.concatenate([ray_section.real[:, None], ray_section.imag[:, None]], axis=1) 
    ray_points = points + np.array(start)[None, ...]
    
    return ray_points

def get_values(gray, ray_points):
    '''
    ray_points の画素値を取得
    '''
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
    '''
    original_pointsの差分をとる（微分する）
    '''
    rotated_points = np.roll(original_points, shift=1)
    diff = original_points - rotated_points
    diff[0] = 0
    return diff


def ransac(picked_points_all_direction, gray, sobel, ellipse_point_num=cfg.Params.ELLIPSE_POINT_NUM):
    '''
    ransacにより瞳をfittingする楕円の候補を求める
    '''
    H, W = gray.shape
    assert len(picked_points_all_direction) >= 5
    good_elipses = [] # ransacで基準を満たす楕円
    fit_mse_scores = [] # ransacで基準を満たすものの候補のfitting mse
    surrounding_values = [] # ransacで基準を満たす楕円周上にそってsobel画像を積分した値
    fitted_points = [] # ransacに使った点
    
    assert len(picked_points_all_direction) >= 5

    indexes = np.arange(len(picked_points_all_direction)).tolist()

    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
   
    
    for _ in range(cfg.Params.MAX_RANSAC_TIME):
        pick_index = random.sample(indexes, ellipse_point_num)
        points = picked_points_all_direction[pick_index]
        points = points.astype(np.float32)
       
        ellipse = cv2.fitEllipse(points)
        (cx, cy), (w, h), deg = ellipse
        a = max(h, w)
        b = min(h, w)
        f = (a - b) / a
        if f > 0.3: # 楕円が細長すぎるものを除去
            continue
        if a > 1 *(W // 2): # 長径が大きすぎるもの（画像サイズの半分以上）を除去
            continue
        omegas = np.linspace(0, 2*np.pi,cfg.Params.OMEGA_NUM)
        w_ = w //2
        h_ = h//2
        deg_ = deg * np.pi/180
        ellipse_x = (w_ * np.cos(omegas) * np.cos(deg_)) - (h_ * np.sin(omegas) * np.sin(deg_)) + cx
        ellipse_y = (w_ * np.cos(omegas) * np.sin(deg_)) + (h_ * np.sin(omegas) * np.cos(deg_)) + cy

        ellpise_surrounding = np.concatenate([ellipse_x[:,None], ellipse_y[:,None]], axis=1)
        
        # 全データサンプルを用いて誤差の小さいもので再度　fitting
        normal_value_points = []# fittingした楕円との距離がRANSAC_MSE_TH以下の点のみを残す
        for point in picked_points_all_direction:
            distance =  np.linalg.norm((ellpise_surrounding-point), axis=1)
            min_dist = np.min(distance)
            if min_dist <= cfg.Params.RANSAC_MSE_TH:
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
        
        omegas = np.linspace(0, 2*np.pi, cfg.Params.OMEGA_NUM)
        w_ = w //2
        h_ = h//2
        deg_ = deg * np.pi/180
        ellipse_x = (w_ * np.cos(omegas) * np.cos(deg_)) - (h_ * np.sin(omegas) * np.sin(deg_)) + cx
        ellipse_y = (w_ * np.cos(omegas) * np.sin(deg_)) + (h_ * np.sin(omegas) * np.cos(deg_)) + cy

        ellpise_surrounding = np.concatenate([ellipse_x[:,None], ellipse_y[:,None]], axis=1) # fittingした楕円上にある点群
        
        
        # fit した楕円とのmseを計算
        min_distance_list = []
        for point in normal_value_points:
            distance = np.linalg.norm((ellpise_surrounding-point), axis=1)
            min_dist = np.min(distance)
            min_distance_list.append(min_dist)
        mse = np.mean(min_distance_list)
        fit_mse_scores.append(mse)

        # sobel画像の楕円周上のピクセル値の平均を計算(画素値が大きい方が良い)
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
        

    min_mse_candidate_indexes = np.argsort(fit_mse_scores)[:cfg.Params.BEST_CHOOSE]
    high_value_candidate_indexes = np.argsort(surrounding_values)[::-1][:cfg.Params.BEST_CHOOSE]
    if cfg.Params.USE_PIKEL and cfg.Params.USE_MSE:
        good_candidates = list(set((list(min_mse_candidate_indexes) + list(high_value_candidate_indexes))))
    elif cfg.Params.USE_MSE:
        good_candidates = list(min_mse_candidate_indexes)
    elif cfg.Params.USE_PIKEL:
        good_candidates = list(high_value_candidate_indexes)


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
    
    final_candidate_center = np.array(final_candidate_center)
    final_candidate_major = np.array(final_candidate_major)
    final_candidate_minor = np.array(final_candidate_minor)
    return final_candidates, final_candidate_center, final_candidate_major, final_candidate_minor, image, final_fitted_points
    




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

    result_image= copy.deepcopy(original_image)
   
    result_image =  cv2.ellipse(result_image,observe_ellipse,(255,255,0),2)
    result_image = cv2.circle(result_image, (int(observe_center[0]), int(observe_center[1])), 5,(0,255,0),-1) # green

    

    for point in observe_points:
        result_image = cv2.circle(result_image, (int(point[0]), int(point[1])), 3,(0,0,255),-1)
   


    return observe_center, observe_major, observe_minor, observe_ellipse, result_image, observe_points



def refine_ransac(candidate_points, gray, sobel, pupil_center, pupil_major, pupil_minor):
    final_ellipse_candidates, final_candidate_center, final_candidate_major, final_candidate_minor, image, final_candidate_fitted_points = ransac(picked_points_all_direction=candidate_points, gray=gray, sobel=sobel, ellipse_point_num=int(cfg.Params.ELLIPSE_POINT_NUM*0.7))

    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    for point in candidate_points:
        result_image = cv2.circle(result_image, (int(point[0]), int(point[1])), 2,(0,255,255),-1)


    observe_center, observe_major, observe_minor, observe_ellipse, result_image, observe_points = select_best(final_candidate_center=final_candidate_center, final_candidate_major=final_candidate_major, final_candidate_minor=final_candidate_minor, final_ellipse_candidates=final_ellipse_candidates, final_candidate_fitted_points=final_candidate_fitted_points, pupil_center=pupil_center, pupil_major=pupil_major, pupil_minor=pupil_minor, original_image=result_image)

    return observe_center, observe_major, observe_minor, observe_ellipse, result_image, observe_points
    

if __name__ == "__main__":
    plt.rcParams["font.size"] = 20
    video_path=cfg.Data.InputMoviePath
    save_image_folder_path = cfg.Data.DebugDir
    os.makedirs(save_image_folder_path, exist_ok=True)
    
        
    pupil_center_points_list = [cfg.Init.PupilCenter]
    pupil_major_minor_list= [cfg.Init.PupilMajorMinor]
    DETECT_FLG = True #瞳をfittingできたか

    pupil_center_tracker = KalmanTracker(result=pupil_center_points_list[0]) # 中心座標をトラッキングするKalman filter
    pupil_major_minor_tracker = KalmanTracker(result=pupil_major_minor_list[0]) # 長径と短径の長さをトラッキングするKalman Filter
    
    cap = cv2.VideoCapture(video_path)
    total_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"total frame {total_frame_num} fps {original_fps}")

    frame_count = 0
    bar = tqdm(total=total_frame_num)
    bar.set_description('Progress Frames')

    df = pd.DataFrame(columns=["frame_num", "xc","yc","major","minor","degree", "detected"]) # フレーム番号、瞳の中心(xc,yc)、瞳の長径、短径、傾き(major,minor,degree)、瞳をfittingできたか(True:できたFalse:できなかった)
    df_ind = 0

    while True:
        ret, frame = cap.read()
        if ret == True:
            

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop_x0, crop_y0, crop_x1, crop_y1 = cfg.Init.ImageCrop
            original_image = gray[crop_y0:crop_y1,crop_x0:crop_x1]
            
            #  cx,257 cy,300 w40, h25, deg -10
            # if DELETE_HIGHLIGHT:
            #     high_light_mask = np.zeros(cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR).shape).astype(np.uint8)
            #     high_light_mask =   cv2.ellipse(high_light_mask,((257,300),(50,30),-10),(255,255,255),20)
            #     high_light_mask  = cv2.cvtColor(high_light_mask, cv2.COLOR_BGR2GRAY)
            #     # image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            #     # copy_image = copy.deepcopy(image)
            #     # image = cv2.ellipse(image,((257,300),(40,25),-10),(0,255,2255),5)
            #     # fig =  plt.figure(figsize=(24, 18))
            #     # ax1 = fig.add_subplot(1, 3, 1)
            #     # ax1.imshow(copy_image)
            #     # ax2 = fig.add_subplot(1, 3, 2)
            #     # ax2.imshow(image)
            #     # ax3 = fig.add_subplot(1, 3, 3)
            #     # ax3.imshow(high_light_mask)
            #     # plt.show()
            #     # plt.close()
            
            
            if cfg.Option.DELETE_LINE: # 横線ノイズを消すか
                input_image = copy.deepcopy(original_image)
                draw_detect_line_image, mask, inpaint_image = delete_line(img=original_image, frame_count=frame_count)
                
            
            else:
                mask = None

           
            kernel_size = 3
            blured_image = blur_image(original_image=original_image, type="median", kernel_size=kernel_size) # median filterでノイズ除去
            sobel_combined = make_sobel_image(sobel_kernel=-1, blured_image=blured_image) #画像を微分
            erode_kernel = np.ones((3, 3), np.uint8)
            erode_original = cv2.erode(sobel_combined, erode_kernel, iterations=1)
            dilate_original = cv2.dilate(erode_original, erode_kernel, iterations=1)
            mask_sobel_combined = copy.deepcopy(sobel_combined)
            mask_sobel_combined[mask==255] = 0 #線ノイズの部分の微分は0にする
            # if DELETE_HIGHLIGHT:
            #     mask_sobel_combined[high_light_mask==255] = 0
            erode = cv2.erode(mask_sobel_combined, erode_kernel, iterations=1)
            dilate = cv2.dilate(erode, erode_kernel, iterations=1)
            
           
            if DETECT_FLG: #1フレーム前でfittingできたときはKalman Filterで予測する
                pupil_center = pupil_center_tracker.predict()
                pupil_major,pupil_minor = pupil_major_minor_tracker.predict()

            else: # fittingできなかったときはfittingできた最後の観測を利用する
                pupil_center = pupil_center_points_list[-1]
                pupil_major,pupil_minor = pupil_major_minor_list[-1]
                
            # print(pupil_center, pupil_major, pupil_minor)
            thetas = np.linspace(0, 2*np.pi, cfg.Params.THETA_NUM)
            picked_points_all_direction = [] #中心から外側に微分して初めて微分値が閾値を超えた点群（楕円fittingに使う点群）
            picked_points_for_show  = None
            
          
                
                
            if mask is not None:
                dilated_mask = cv2.dilate(mask, erode_kernel, iterations=3)
            # if DELETE_HIGHLIGHT:
            #     dilated_high_light_mask = cv2.dilate(high_light_mask, erode_kernel, iterations=10)
              
                    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                    # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # cv2.imshow("img", dilated_mask)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
            for theta in thetas:
               
                # 元の画像の微分値でマスクの中に入っているもの以外
                ray_points = get_points(gray=dilate, start=pupil_center, theta=theta)
                value_points = get_values(gray=dilate, ray_points=ray_points)
                diff_points = take_differencial(original_points=value_points)
                picked_points = ray_points[np.abs(diff_points) >= cfg.Params.DIFF_TH]

                # (微分画像の微分値を0にしたこと由来で二階微分がでている点を除く)
                if mask is not None:
                    
                    mask_value = get_values(gray=dilated_mask, ray_points=picked_points)
                    picked_points= picked_points[mask_value < 200]
                if len(picked_points) > 0:
                    picked_points_all_direction.append(picked_points[0])
                    if picked_points_for_show is None:
                        picked_points_for_show = picked_points
                    else:
                        picked_points_for_show = np.concatenate([picked_points_for_show, picked_points], axis=0)
                

           
                
            
            picked_points_all_direction = np.array(picked_points_all_direction)

            final_ellipse_candidates, final_candidate_center, final_candidate_major, final_candidate_minor, candidate_draw_image, final_candidate_fitted_points = ransac(picked_points_all_direction=picked_points_all_direction, gray=original_image, sobel=sobel_combined)

            
            #候補がないとき
            if len(final_candidate_center) == 0:
                print(f"{frame_count} fitting failed")
                DETECT_FLG = False
                df.loc[df_ind] = [frame_count, None, None, None, None, None, False]
                df_ind += 1

                fig =  plt.figure(figsize=(24, 24))
                ax1 = fig.add_subplot(3, 3, 1)
                if inpaint_image is not None:
                    # ax1.imshow(input_image, cmap="gray")
                    ax1.imshow(cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR))
                    
                else:
                    # ax1.imshow(original_image, cmap="gray")
                    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR))
                    
                ax1.axis("off")
                ax1.set_title(f"Frame {frame_count} original image")
                ax2 = fig.add_subplot(3, 3, 2)
                # ax2.imshow(blured_image, cmap="gray")
                ax2.imshow(cv2.cvtColor(blured_image, cv2.COLOR_GRAY2BGR), cmap="gray")
                
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
                    picked_points = ray_points[np.abs(diff_points) >=  cfg.Params.DIFF_TH] 
                    draw_picked_points = picked_points[(picked_points[:,0] > 0) & (picked_points[:,0]<original_image.shape[1]) & (picked_points[:,1] < original_image.shape[0]) & (picked_points[:, 1] > 0)]
                    if len(picked_points) > 0:
                        ax6.scatter(draw_picked_points[:, 0],(original_image.shape[0]- draw_picked_points[:, 1]), color="yellow")
                        ax6.scatter(picked_points[0, 0], (original_image.shape[0] - picked_points[0, 1]), color="red")
                ax6.scatter(pupil_center[0],(original_image.shape[0] - pupil_center[1]) ,c="magenta", s= 100)

                ax6.set_xlim([0, original_image.shape[1]])
                ax6.set_ylim([0, original_image.shape[0]])

                ax6.axis("off")      
                ax6.set_title(f"Differentiate in radical direction")     

                plt.tight_layout()
                save_path = os.path.join(save_image_folder_path, f"debug_frame{frame_count:04d}_median_kernel{kernel_size}_fale.jpg")
                plt.savefig(save_path)
                plt.close()
                frame_count += 1
                bar.update(1)
                if frame_count >= total_frame_num:
                    break 
                
                continue


            # 候補の楕円の中から最も良いものを選ぶ
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
           
            
            image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result_image =  cv2.ellipse(result_image,observe_ellipse,(255,255,0),2)
            for point in observe_points:
                result_image = cv2.circle(result_image, (int(point[0]), int(point[1])), 3,(0,0,255),-1)
            result_image = cv2.circle(result_image, (int(observe_center[0]), int(observe_center[1])), 5, (0,255,0),-1)

            #fittingした楕円とKalmanFilterの予測が離れすぎているときはKalmanFilterをアップデートしない
            prediction_dist = np.linalg.norm(observe_center - pupil_center)
            if prediction_dist < cfg.Params.PREDICTION_DIST_TH:
                # update
                pupil_center_tracker.update(new_posx=observe_center[0], new_posy=observe_center[1])
                pupil_major_minor_tracker.update(new_posx=observe_major, new_posy=observe_minor)

                pupil_center_points_list.append(observe_center)
                pupil_major_minor_list.append([observe_major, observe_minor])
                DETECT_FLG = True
            else:
                print(f"{frame_count} failed ")
                DETECT_FLG = False

            #保存 (cx, cy), (w, h), deg = draw_ellipse
            (cx_, cy_), (w_, h_), deg_ = observe_ellipse
            df.loc[df_ind] = [frame_count, cx_+crop_x0, cy_+crop_y0, w_, h_, deg_, DETECT_FLG]
            del cx_, cy_,w_, h_, deg_
            df_ind += 1

            fig =  plt.figure(figsize=(24, 24))
            ax1 = fig.add_subplot(3, 3, 1)
            if inpaint_image is not None:
                # ax1.imshow(input_image, cmap="gray")
                ax1.imshow(cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR))
                
            else:
                # ax1.imshow(original_image, cmap="gray")
                ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR))
            ax1.axis("off")
            ax1.set_title(f"Frame {frame_count} original image")
            ax2 = fig.add_subplot(3, 3, 2)
            # ax2.imshow(blured_image, cmap="gray")
            ax2.imshow(cv2.cvtColor(blured_image, cv2.COLOR_GRAY2BGR), cmap="gray")
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
           

            ax5 = fig.add_subplot(3, 3, 5)
            ax5.imshow(cv2.flip(cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR), 0))
            picked_points_for_show =  picked_points_for_show[(picked_points_for_show[:,0] > 0) & (picked_points_for_show[:,0]<original_image.shape[1]) & (picked_points_for_show[:,1] < original_image.shape[0]) & (picked_points_for_show[:, 1] > 0)]
            ax5.scatter(picked_points_for_show[:, 0],(original_image.shape[0]- picked_points_for_show[:, 1]), color="yellow")
            ax5.scatter(picked_points_all_direction[:, 0], (original_image.shape[0] - picked_points_all_direction[:, 1]), color="red")
            ax5.scatter(pupil_center[0],(original_image.shape[0] - pupil_center[1]) ,c="magenta", s=100)
            ax5.set_xlim([0, original_image.shape[1]])
            ax5.set_ylim([0, original_image.shape[0]])

            ax5.axis("off")      
            ax5.set_title(f"Differentiate in radical direction")     

            ax6 = fig.add_subplot(3, 3, 6)
            candidate_draw_image = cv2.cvtColor(candidate_draw_image, cv2.COLOR_BGR2RGB)
            ax6.imshow(candidate_draw_image)
            ax6.axis("off")
            ax6.set_title(f"Candidates")  
          
            
            
            if mask is not None:
            
                ax7 = fig.add_subplot(3, 3, 7)
                ax7.imshow(draw_detect_line_image)
                ax7.axis("off")
                ax7.set_title(f" Detected lines")
                ax8 = fig.add_subplot(3, 3, 8)
                ax8.imshow(cv2.cvtColor(dilate_original, cv2.COLOR_GRAY2BGR))
                ax8.axis("off")
                ax8.set_title(f"Sobel")
                ax9 = fig.add_subplot(3, 3, 9)
                ax9.imshow(cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR))
                ax9.axis("off")
                ax9.set_title(f"Sobel delete line")            
            
            plt.tight_layout()
            
            save_path = os.path.join(save_image_folder_path, f"debug_frame{frame_count:04d}_median_kernel{kernel_size}.jpg")
            plt.savefig(save_path)
            # plt.show()
            plt.close()
            bar.update(1)
            frame_count +=1
            if frame_count >= total_frame_num:
                break 
         
    cap.release()

    #結果をcsvファイルに保存
    df.to_csv(cfg.Data.OutputFilePath, index=False)
    # 結果をvideoに
    frame_rate = 5
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    size = (600, 600) 
    
    video_file_path = os.path.join(save_image_folder_path, f"result.mp4")
    writer = cv2.VideoWriter(video_file_path, fmt, frame_rate, size)
    picture_paths = sorted(glob.glob(os.path.join(save_image_folder_path, "*.jpg")))
    for path in picture_paths:
        frame = cv2.imread(path)
        frame = cv2.resize(frame, dsize=size)
        writer.write(frame)
    writer.release()  
    

