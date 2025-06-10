import pandas
import time
import lib.callbacks as call
import roboticstoolbox as rtb
import numpy as np

PATH = call.handle_path("restricted_area.csv")
df = pandas.read_csv(PATH)
PATH = call.handle_path("points.csv")
df_p = pandas.read_csv(PATH)

if np.isnan(df_p.values[0,:]).any():
    df_p.values[0,:] = np.nan_to_num(df_p.values[0,:], nan=0.0)

if np.isnan(df_p.values[-1,:]).any():
    df_p.values[-1,:] = np.nan_to_num(df_p.values[-1,:], nan=0.0)

limits = [[-0.4, 0.4], [-0.6, 0.6], [0.0, 0.7]]
resources = {"start_loc":   rtb.models.Panda().fkine(df_p.values[0,:]).t,
             "dest_loc":    rtb.models.Panda().fkine(df_p.values[-1,:]).t,
             "radius":      0.04,
             "start_color": (0, 255, 0),
             "dest_color":  (0, 0, 255),
             "box_color":   (255, 10, 10),
             "box_info":    df.values,
             "limits":      limits}

objects, env = call.setup_env(panda=True,
                              boxes = True, 
                              start = True, 
                              dest = True, 
                              resources=resources)  
time.sleep(2)
call.robot_move(objects, env, df_p.values, joint_q=True)    
env.close()
del env