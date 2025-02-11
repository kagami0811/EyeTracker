import os
import matplotlib.pyplot as plt
import cv2
import copy
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

ALL_X_INPAINT = True # x軸全てをinpaintするか
NOISE_LINE_WIDTH = 15

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def soft_max(x, axis=1):
    x -= x.max(axis, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / x_exp.sum(axis, keepdims=True)


def inpaint_hline(img, lines):
    '''
    線ノイズ(lines)の部分を表すマスクを作成 画像自体はinpaintしない。(微分画像上で線ノイズ由来の微分を無視する)
    '''
    
    inpaint_img = copy.deepcopy(img)
    draw_image = copy.deepcopy(img)
    h, w, _ = img.shape
    mask = np.zeros(img.shape).astype(np.uint8)
    for line in lines:

        x1, y1, x2, y2 = line[0]
        if ALL_X_INPAINT:
            x1 = 0
            x2 = w-1
        mask = cv2.line(mask, (x1,y1), (x2,y2), (255,255,255), NOISE_LINE_WIDTH)
        draw_image = cv2.line(draw_image, (x1,y1), (x2,y2), (0,0,255), NOISE_LINE_WIDTH)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return draw_image, mask, inpaint_img
    # original をinpaintするとき
    for line in lines:
        linewise_mask = np.zeros(img.shape).astype(np.uint8)
        x1, y1, x2, y2 = line[0]
        if ALL_X_INPAINT:
            x1 = 0
            x2 = w-1
        linewise_mask = cv2.line(linewise_mask, (x1,y1), (x2,y2), (255,255,255), NOISE_LINE_WIDTH)
        linewise_mask = cv2.cvtColor(linewise_mask, cv2.COLOR_BGR2GRAY)
        # 垂線
        if y2 - y1 == 0:
            theta = (np.pi /2)
        else:
            theta = np.arctan2((x1-x2), (y2-y1))
        
        inpaint_points = np.where(linewise_mask==255)
        inpaint_points = np.concatenate([inpaint_points[1][...,None], inpaint_points[0][..., None]], axis=1)
        inpaint_pixel_num = len(inpaint_points)
        # inpaint_length = np.array(list(np.arange(-20,-6)) + list(np.arange(6, 20)))
        inpaint_length = np.arange(-15,15)
        ray_section = inpaint_length * np.exp(1j * theta) 
        points = np.concatenate([ray_section.real[:, None], ray_section.imag[:, None]], axis=1) 
        inpaint_points = np.repeat(inpaint_points, len(inpaint_length), axis=0)
        points = np.tile(points, (inpaint_pixel_num, 1))
        
        for_inpaint_points = inpaint_points - points
        
        
        ray_points_ = copy.deepcopy(for_inpaint_points)
        ray_points_[:, 0] = (for_inpaint_points[:, 0] - ( w//2)) / (w //2)
        ray_points_[:, 1] = (for_inpaint_points[:, 1] - ( h // 2)) / ( h //2)
        
        # 実画像からとってくる
        input = torch.from_numpy(gray.astype(np.float32)).reshape(1,1,h,w)
        grid = torch.from_numpy(ray_points_.astype(np.float32)).reshape(1,1,-1,2)
        #https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        output = F.grid_sample(input=input, grid=grid, mode="bilinear", padding_mode="zeros", align_corners=True).reshape(-1)
        output = output.detach().numpy().copy()
        output = output.reshape(inpaint_pixel_num, -1)
        
        # ただ平均をとる
        # output = np.mean(output, axis=1)
        # output = np.clip(output, 0, 255).astype(np.uint8)    
        # inpaint_img[np.where(linewise_mask==255)] = output[..., None]

        # inpaintすべきところからサンプルしていないか（mask画像からとってくる）inpaintすべきところからとらないか
        input_mask = torch.from_numpy(mask.astype(np.float32)).reshape(1,1,h,w)
        grid_mask = torch.from_numpy(ray_points_.astype(np.float32)).reshape(1,1,-1,2)
        
        output_mask = F.grid_sample(input=input_mask, grid=grid_mask, mode="bilinear", padding_mode="zeros", align_corners=True).reshape(-1)
        output_mask = output_mask.detach().numpy().copy()
        output_mask = output_mask.reshape(inpaint_pixel_num, -1)
        calc_point =  output_mask<200
        calc_point = calc_point.astype(np.int64)
        ok_calc_point_num = np.sum(calc_point, axis=1)
        assert (ok_calc_point_num >= 1).all()
        # 距離に応じた重み
        input_length_weight = np.repeat(inpaint_length[None,...], inpaint_pixel_num, axis=0)
        input_length_weight  = input_length_weight / np.max(input_length_weight)
        input_length_weight = 1 -  sigmoid(np.abs(input_length_weight))
        input_length_weight[output_mask>200] = -float('inf')
        input_length_weight = soft_max(input_length_weight)
        # ただの平均
        # output = np.sum((output * calc_point), axis=1) / ok_calc_point_num
        # 重みつき平均
        output = np.sum((output * input_length_weight), axis=1) 
        output =np.clip(output, 0, 255).astype(np.uint8) 
        inpaint_img[np.where(linewise_mask==255)] = output[:, None]
        
        # mask_ = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # fig =  plt.figure(figsize=(24, 18))
        # ax1 = fig.add_subplot(1, 1, 1)
        # ax1.imshow(mask_)
        # radius = 10
        # ax1.scatter(for_inpaint_points[:,0], for_inpaint_points[:,1],s=radius, c="blue")
        # plt.show()
        # plt.close()
        

        
    # merge = np.hstack([img, draw_image,inpaint_img])
    
    # cv2.imwrite(f"DEBUG_inpaint_{frame_count:03d}.jpg", merge)
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow("img", merge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return draw_image, mask, inpaint_img
        

def delete_line(img, frame_count=None):
    '''
    img上の線ノイズの部分のmaskを作成 ここで画像自体のinpaintはしない
    '''
    gray = copy.deepcopy(img)
    draw_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color_image = copy.deepcopy(draw_image)

    #https://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #https://qiita.com/tifa2chan/items/d2b6c476d9f527785414    
    #https://qiita.com/kotai2003/items/662c33c15915f2a8517e


    med_val = np.median(gray)
    sigma = 0.33  # 0.33
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))

    edges = cv2.Canny(gray,min_val,max_val,apertureSize = 3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=50, minLineLength=50, maxLineGap=30)
    if lines is not None:
        
        # x軸にほぼ並行(なす角20度以下)でない線はノイズではないので除去
        refined_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            theta = np.arctan2((y2 - y1)  ,(x2 - x1 + 1e-12))
            
            if abs(theta) < (20 * np.pi / 180):
                refined_lines.append([[x1, y1, x2, y2]])
        if len(refined_lines) > 0:
            
        
        
           
            draw_detect_line_image, mask, inpaint_image = inpaint_hline(img=color_image, lines=refined_lines)
            return draw_detect_line_image, mask, inpaint_image

    return None, None, None
