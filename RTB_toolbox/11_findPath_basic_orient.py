import pandas
from spatialgeometry import Sphere
import numpy as np
import lib.callbacks as call
from spatialmath import SO3, SE3
import roboticstoolbox as rtb
import time

#PATH = call.handle_path("restricted_area.csv")
PATH = call.handle_path("restricted_area.csv")
df = pandas.read_csv(PATH)
limits = [[-0.4, 0.4], [-0.6, 0.6], [0.0, 0.7]]
resources = {"radius":                  0.04,
             "point_to_check_color":    (0, 0, 0),
             "map_limits":              [-2, 2, -2, 2,-1, 3],
            #  "start_loc":               [np.random.uniform(limits[index][0], limits[index][1]) for index, _ in enumerate(limits)],
             "start_loc":               [-0.16, -0.1, 0.67],
             "start_color":             (0, 255, 0),
             "dest_color":              (0, 0, 255),
            #  "dest_loc":                [np.random.uniform(limits[index][0], limits[index][1]) for index,_ in enumerate(limits)],
             "dest_loc":                [0.26, 0.40, 0.69],
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

Togo = [objects['start']]
# Togo_list = [np.concatenate((
#     objects['start'].T[0:3, 3],
#     SO3(SE3.Rt(SO3.Rx(3.14) @ SO3.Ry(0.0)).R).rpy(order='xyz', unit='deg')
# ))]
Togo_list = []
Temp = objects['start']
cnt = 0
in_collision = False
q_last = None
q = objects["panda"].qr
pandas = []

while True:
    # Position check
    print('trying position')


    Temp = call.try_positions(Togo, cnt, resources, objects, env, Temp, search_radius=0.02)
    # Orientation check
    Togo.append(Temp)
    xyz = Temp.T[:3, 3]
    print('trying orientation')
    rot_last = SO3(SE3.Rt(SO3.Rx(3.14) @ SO3.Ry(0.0)).R)
    q = objects["panda"].ik_GN(SE3.Rt(rot_last, xyz), q0=objects["panda"].qr, joint_limits=True, pinv=True)
    q = q[0]
    bias = 0.1
    if q_last is None:
        q_last = q
        thresh = 3.14
    else:
        thresh = 0.002
    # Final = SE3.Rt(rot_last, xyz)
    last = objects["panda"].fkine(q_last)
    orient_tries = 1
    while any(objects["panda"].iscollided(q, instance_box) for instance_box in objects["box"]) or call.joints_changed_significantly(q, q_last, bias + thresh*orient_tries):
        if orient_tries > 1000:
            print("Orientation tries exceeded 1000, trying new position.")
            Temp = call.try_positions(Togo, cnt, resources, objects, env, Temp, search_radius=0.02)
            Togo.append(Temp)
            xyz = Temp.T[:3, 3]
            rot_last = SO3(SE3.Rt(SO3.Rx(3.14) @ SO3.Ry(0.0)).R)
            q = objects["panda"].ik_GN(SE3.Rt(rot_last, xyz), q0=objects["panda"].qr, joint_limits=True, pinv=True)
            q = q[0]
            orient_tries = 1
            last = objects["panda"].fkine(q_last)
            continue
        rot = call.generate_orientation(rot_last, max_change=bias + orient_tries*thresh)
        q = objects["panda"].ik_GN(SE3.Rt(rot, xyz), q0=last, joint_limits=True, pinv=True)
        q = q[0]
        orient_tries += 1
        # Final = SE3.Rt(rot, xyz)
    # print(f"joints are moved significantly: {call.joints_changed_significantly(q, q_last, thresh)}, joint config:\n q: {q}, \nq_last: {q_last}")
    # If any value in q is nan, set it to 0
    if np.isnan(q).any():
        q = np.nan_to_num(q, nan=0.0)
    q_last = q
    # Create and store Panda robots for visualization
    panda = rtb.models.Panda()
    panda.q = q
    env.add(panda, collision_alpha=0.3, robot_alpha=0)
    cnt += 1
#     Togo_list.append(np.concatenate((
#     xyz.flatten(),  # [x, y, z]
#     SO3(Temp.T[:3, :3]).rpy(order='xyz', unit='deg')  # [r, p, y]
# )))
#     Togo_list.append(np.concatenate((
#     Final.t,  # [x, y, z]
#     SO3(Final.R).rpy(order='xyz', unit='deg')  # [r, p, y]
# )))
    Togo_list.append(q)
    env.add(Togo[-1])
    if objects['dest'].iscollided(Temp):
        headers = [f'j_{joint}' for joint in range(0, len(panda.q))]
        PATH = call.handle_path("points.csv")
        call.generate_csv(PATH, headers=headers, array=Togo_list)
        break
print(f"The point has arrived to its destination with {cnt} itterations")
env.close()
del env