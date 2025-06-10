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
limits = [[-0.7, 0.7], [-0.7, 0.7], [0.0, 0.8]]
resources = {"radius":                  0.04,
             "point_to_check_color":    (0, 0, 0),
             "map_limits":              [-2, 2, -2, 2,-1, 3],
            #  "start_loc":               [np.random.uniform(limits[index][0], limits[index][1]) for index, _ in enumerate(limits)],
            #  "start_loc":               [-0.16, -0.1, 0.67],
             "start_loc":               [0.1, -0.3, 0.5],
             "start_color":             (0, 255, 0),
             "dest_color":              (0, 0, 255),
            #  "dest_loc":                [np.random.uniform(limits[index][0], limits[index][1]) for index,_ in enumerate(limits)],
            #  "dest_loc":                [0.26, 0.40, 0.69],
             "dest_loc":                [-0.55, -0.3, 0.5],
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

path, _, _ = find_tree(env, resources["limits"], objects)
headers = [f'j_{joint}' for joint in range(0, len(objects["panda"].q))]
PATH = call.handle_path("points.csv")
call.generate_csv(PATH, headers=headers, array=[path[i].q for i in range(len(path))])
env.hold()

env.close()
del env