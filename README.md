# EyeTracker 

## 環境
    

```
singularity build --fakeroot EyeTrack.sif EyeTrack.def

singularity shell --nv EyeTrack.sif
```

## 前処理

<!-- 
- 目の周辺のみを切りとる


```
ffmpeg -i input.mov -vf crop=w=600:h=600 output.mov
``` -->


<!-- - trimming

```
ffmpeg -ss 00:22:00 -t 20  -i 241208cut2_crop.mov -c copy 241208cut2_crop_trim22m.mov
ffmpeg -ss 00:16:33 -t 20  -i 241208cut2_crop.mov -c copy 241208cut2_crop_trim21633m.mov
ffmpeg -ss 00:16:42 -t 20  -i 241208cut2_crop.mov -c copy 241208cut2_crop_trim21637m.mov


``` -->

- コントラストをできるだけ上げる 

    contrast, brightnessの値は適宜調節する

```
ffmpeg -i input.mov -vf "eq=brightness=0.1:contrast=1.8" output.mov

```

- 目の周辺を600px x 600 px の大きさで切り取る


## 実行

- config.yamlファイルで入出力,初期中心座標,瞳の長径,短径(Data, Init)とその他の設定を変更する

```
python DetectPupil.py config.yaml
```


<!-- ## Tracking

```
python OrloskyPupilDetector.py 241208cut2_crop_trim22m_b01_c18.mov
```



# EyeTracker (Original)
A lightweight, robust Python eye tracker

This repository is an open-source eye tracking algorithm written in Python. Currently, it is an updated version of the pupil tracker from https://github.com/YutaItoh/3D-Eye-Tracker/blob/master/main/pupilFitter.h that has been optimized and simplified. 

To use the script, run "python .\OrloskyPupilDetector.py" from your shell. If the hardcoded file path in the select_video() function does not find a video at the specified path, it will open a browse window that allows you to select a video. The process_video() function handles the majority of the processing and can be easily modified to work with a camera capture or image. It returns a rotated_rect that represents the pupil ellipse. A lite version is also included that is more efficient, but less robust. Be sure to have an adequate light source for the lite version. 

A test video (eye_test.mp4) is included in the root directory for testing. Algorithm details are explained here: https://www.youtube.com/watch?v=bL92JUBG8xw

When running the script on this test video, your results should look like this: https://youtu.be/B06cUMplDHw.  

If you need an eye camera, I have instructions for building your own IR camera for under $100 here: https://www.youtube.com/watch?v=8lZqCMRMtC8

To help support this software and other open-source projects, please consider subscribing to my YouTube channel: https://www.youtube.com/@jeoresearch

Requirements:
- A Python environment

Packages
- numpy ****There is a known issue with numpy 2.0.0. Downgrading to 1.26.0 or another version can solve this issue.
- opencv

Assumptions
- Works best with 640x480 videos. Images will be cropped to size equally horizontally/vertically if aspect ratio is not 4:3.
- The image must be that of the entire eye. Dark regions in the corners of the image (e.g. VR display lens borders) should be cropped.  -->
