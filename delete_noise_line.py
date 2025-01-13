import os
import matplotlib.pyplot as plt
import cv2
import copy
import numpy as np


def inpaint_line(img, lines, frame_count=None):
    mask = np.zeros(img.shape).astype(np.uint8)
    draw_image = copy.deepcopy(img)
    for line in lines:
        # draw white  
        x1, y1, x2, y2 = line[0]
        mask = cv2.line(mask, (x1,y1), (x2,y2), (255,255,255), 3)
        draw_image = cv2.line(draw_image, (x1,y1), (x2,y2), (0,0,255), 3)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #https://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_photo/py_inpainting/py_inpainting.html

    inpaint_radius = 20
    
    draw_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)

    dst11 = cv2.inpaint(img,mask,0,cv2.INPAINT_TELEA)
    dst12 = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    dst13 = cv2.inpaint(img,mask,inpaint_radius,cv2.INPAINT_TELEA)
    dst21 = cv2.inpaint(img,mask,0,cv2.INPAINT_NS)
    dst22 = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)
    dst23 = cv2.inpaint(img,mask,inpaint_radius,cv2.INPAINT_NS)


    if frame_count is not None:
        fig =  plt.figure(figsize=(24, 24))
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title(f"original image ")
        ax2 = fig.add_subplot(3, 3, 2)
        ax2.imshow(mask, cmap="gray")
        ax2.axis("off")
        ax2.set_title(f"mask image ")
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.imshow(draw_image)
        ax3.axis("off")
        ax3.set_title(f"inpaint image ")
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.imshow(dst11)
        ax4.axis("off")
        ax4.set_title(f" telea 0")
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.imshow(dst12)
        ax5.axis("off")
        ax5.set_title(f" telea 3")
        ax6 = fig.add_subplot(3, 3, 6)
        ax6.imshow(dst13)
        ax6.axis("off")
        ax6.set_title(f" telea{inpaint_radius}")
        ax7 = fig.add_subplot(3, 3, 7)
        ax7.imshow(dst21)
        ax7.axis("off")
        ax7.set_title(f" ns 0")
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.imshow(dst22)
        ax8.axis("off")
        ax8.set_title(f" ns 3")
        ax9 = fig.add_subplot(3, 3, 9)
        ax9.imshow(dst23)
        ax9.axis("off")
        ax9.set_title(f" ns {inpaint_radius}")
        plt.tight_layout()
        plt.savefig(f"debug_inpaint_{frame_count:04d}.jpg")
        plt.close()
        

    return draw_image, mask, dst13

    
    


def delete_line(img, frame_count=None):
    #
    gray = copy.deepcopy(img)
    draw_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color_image = copy.deepcopy(draw_image)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    
        draw_detect_line_image, mask, inpaint_image = inpaint_line(img=color_image, lines=lines, frame_count=frame_count)
        return draw_detect_line_image, mask, inpaint_image

    return None, None, None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 赤線を引く
            draw_image = cv2.line(draw_image, (x1,y1), (x2,y2), (0,0,255), 3)
    draw_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
    fig =  plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(gray, cmap="gray")
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(edges, cmap="gray")
    ax2.set_title("canny")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(draw_image)
    ax3.axis("off")

    ax1.set_title(f"Frame {frame_count} original image")
    plt.tight_layout()
    plt.savefig("DEBUG.jpg")
    plt.close()
    import pdb; pdb.set_trace()




if __name__ == "__main__":
    video_path = "241208cut2_crop_trim22m_b01_c18.mov"

    pupil_center_points_list = [[240, 230]]
    pupil_major_minor_list= [[125, 135]]

    cap = cv2.VideoCapture(video_path)
    total_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if ret == True:
            

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (600, 600))
    
            original_image = gray[120:520,60:440]
            delete_line(img=original_image)
            frame_count += 1
            if frame_count > 100:
                break
    cap.release()


    

    # fig =  plt.figure(figsize=(24, 8))
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax1.imshow(original_image, cmap="gray")
    # ax1.axis("off")
    # ax1.set_title(f"Frame {frame_count} original image")
    # plt.tight_layout()
    # plt.savefig("DEBUG.jpg")
    # plt.close()