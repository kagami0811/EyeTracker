from filterpy.kalman import KalmanFilter
import numpy as np



#カルマンフィルター
class KalmanTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as position. (x, y)
  x:(x, y, dx, dy) 
    """
    count = 0
    def __init__(self, result, dt=0.25, dim_z=4):
        self.posx = result[0]
        self.posy = result[1]
        self.kf = KalmanFilter(dim_x=4, dim_z=dim_z)
        # define constant velocity model 
        self.kf.x = np.array([self.posx, self.posy, 0, 0], np.float32) # state(x, y,  dx,dy)
        self.kf.F = np.array([[1, 0, dt, 0], 
                                [0, 1, 0, dt],
                                [0, 0, 1, 0],  
                                [0, 0, 0, 1], 
                               ], np.float32) # state transistion matrix
        self.kf.P = np.array([[1, 0., 0., 0.],
                                [0., 1, 0., 0.],
                                [0., 0., 0.25, 0.],
                                [0., 0., 0., 0.25],
                            ], np.float32) # covariance matrix
        self.kf.Q = np.eye(4) * 0.05  #Process uncertainty/noise
        
        if dim_z == 2:
            self.kf.H = np.array([[1, 0, 0, 0], 
                                [0, 1, 0, 0],
                                ], np.float32) #measurement function
            self.kf.R = np.eye(2) * 0.001  #measurement uncertainty/noise

        if dim_z == 4:
            self.kf.H = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]
                                ], np.float32) #measurement function
            self.kf.R = np.eye(4) * 0.001  #measurement uncertainty/noise
        
        self.dim_z = dim_z
        self.dt = dt
        # self.color = classname_to_color(cls)
        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.history = []
        self.time_since_update = 0
  
        



    def predict(self):
        """
        Advances the state vector and returns the predicted position.
        """
        self.kf.predict()

        if(self.time_since_update>-1): ##### 0
            self.hit_streak = 0
            self.time_since_update += 1
            new_pos = self.kf.x[:2]
            self.history.append(new_pos)
        return self.history[-1]




    def update(self, new_posx, new_posy, new_velocity_x=None, new_velocity_y=None):
        """
        Updates the state vector with observed position.
        """
        if self.dim_z == 4:
            assert new_velocity_x is not None and new_velocity_y is not None
        if (new_posx != None) and (new_posy != None):
            self.time_since_update = 0
            self.history = []
            if self.dim_z == 2:
                z = np.array([new_posx, new_posy], np.float32)
            elif self.dim_z == 4:
                z = np.array([new_posx, new_posy, new_velocity_x, new_velocity_y], np.float32)
            self.kf.update(z)
        else:
            self.kf.update(None)
            self.time_since_update += 1 


