import pandas
from spatialgeometry import Sphere
import numpy as np
import lib.callbacks as call
from spatialmath import SO3, SE3
import roboticstoolbox as rtb
import time
from lib.rrt import find_tree
from lib.callbacks import Node

PATH = call.handle_path("restricted_area.csv")
# PATH = call.handle_path("wall.csv")
df = pandas.read_csv(PATH)
panda_qr = rtb.models.Panda().fkine(rtb.models.Panda().qr).t
panda_qr = np.array([panda_qr[0], panda_qr[1], panda_qr[2]-0.2])
limits = [[-0.6, 0.6], [-0.6, 0.6], [0.0, 0.8]]
resources = {"radius":                  0.04,
             "point_to_check_color":    (0, 0, 0),
             "map_limits":              [-2, 2, -2, 2,-1, 3],
            #  "start_loc":               [np.random.uniform(limits[index][0], limits[index][1]) for index, _ in enumerate(limits)],
            #  "start_loc":               [-0.16, -0.1, 0.67],
            #  "start_loc":               [0.1, -0.3, 0.5],
             "start_loc":               panda_qr,
             "start_color":             (0, 255, 0),
             "dest_color":              (0, 0, 255),
            #  "dest_loc":                [np.random.uniform(limits[index][0], limits[index][1]) for index,_ in enumerate(limits)],
            #  "dest_loc":                [0.26, 0.40, 0.69],
            #  "dest_loc":                [-0.35, -0.35, 0.40],
             "dest_loc":                [-0.49041187, -0.00293996, 0.11924733],
             "iterations":              10,
             "box_info":                df.values,
             "limits":                  limits}

objects, env = call.setup_env(panda = True,
                              start=True, 
                              dest = True, 
                              boxes = True,
                              resources = resources)

print(f'Start point: {objects["start"].T[:3,3]}, dest point: {objects["dest"].T[:3,3]}.')

current = Sphere(radius=resources["radius"], color=(255,255,0))

path, _, _ = find_tree(env, 
                       resources["limits"], 
                       objects, 
                       step_size=0.1, 
                       rotation_limits=[[-np.pi*3/2, np.pi*3/2], [-np.pi*3/2, np.pi*3/2], [-np.pi*3/2, np.pi*3/2]])

headers = [f'j_{joint}' for joint in range(0, len(objects["panda"].q))]
PATH = call.handle_path("points.csv")
call.generate_csv(PATH, headers=headers, array=[path[i].q for i in range(len(path))])
print(f'Path found with {len(path)} nodes.')
print("Press Enter to close the window.")
input()

env.close()
del env