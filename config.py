from yacs.config import CfgNode as CN

config = CN()

config.Data = CN()
config.Data.InputMoviePath = "input_crop.mp4" 
config.Data.DebugDir = "./debug_image" 
config.Data.OutputFilePath = "./debug_image/result.csv" 


config.Init = CN()
config.Init.ImageCrop = [60,120, 440, 520]
config.Init.PupilCenter = [240, 230]
config.Init.PupilMajorMinor = [135, 125]
config.Init.HighLightCenter = [257, 230]
config.Init.HighLightMajorMinor = [50,30]
config.Init.HighLightDeg = -10


config.Option = CN()
config.Option.DELETE_HIGHLIGHT = True
config.Option.DELETE_LINE = True

config.Params = CN()
config.Params.MAX_RANSAC_TIME = 5000
config.Params.ELLIPSE_POINT_NUM = 10
config.Params.RANSAC_MSE_TH = 10
config.Params.USE_MSE = True
config.Params.USE_PIKEL = True
config.Params.THETA_NUM = 100
config.Params.DIFF_TH = 20
config.Params.TH = 30
config.Params.OMEGA_NUM = 200
config.Params.RADIUS_NUM = 30
config.Params.BEST_CHOOSE = 10
config.Params.PREDICTION_DIST_TH = 30




def get_cfg_defaults():
    return config.clone()

