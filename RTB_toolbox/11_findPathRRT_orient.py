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
             "start_loc":               [np.random.uniform(limits[index][0], limits[index][1]) for index, _ in enumerate(limits)],
             "start_color":             (0, 255, 0),
             "dest_color":              (0, 0, 255),
             "dest_loc":                [np.random.uniform(limits[index][0], limits[index][1]) for index,_ in enumerate(limits)],
             "iterations":              10,
             "box_info":                df.values,
             "limits":                  limits}

objects, env = call.setup_env(panda = True,
                              start=True, 
                              dest = True, 
                              boxes = True,
                              resources = resources)

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

while True:
    # Position check
    q = [0, 0, 0]
    print('trying position')
    for i in range(resources["iterations"]):
        best_pose = Togo[cnt].T[0:3,3]
        center = call.generate_point(best_pose, radius=0.03)
        current = Sphere(radius=resources["radius"], color=(10,10,10))
        current_coll_robot = Sphere(radius=3*resources["radius"], color=(10,10,10))
        current_coll_box = Sphere(radius=3*resources["radius"], color=(10,10,10))
        call.update_obj(current, center)
        call.update_obj(current_coll_robot, center)
        call.update_obj(current_coll_box, center)
        for instance_box in objects["box"]:
            if current_coll_box.iscollided(instance_box) or objects["panda"].iscollided(q[:2], current_coll_robot):
                in_collision = True
                break
            else:
                in_collision = False
        # env.add(current_coll_robot)
        env.add(current)
        if (call.euclidean_distance(current.T[0:3,3], objects['dest'].T[0:3,3]) < call.euclidean_distance(Temp.T[0:3,3], objects['dest'].T[0:3,3])) and not in_collision:
            if Temp!=objects['start']:
                env.remove(Temp)
            Temp = current
            if objects['dest'].iscollided(Temp):
                break
        else:
            env.remove(current)
    # Orientation check
    Togo.append(Temp)
    xyz = Temp.T[:3, 3]
    print('trying orientation')
    rot_last = SO3(SE3.Rt(SO3.Rx(3.14) @ SO3.Ry(0.0)).R)
    q = objects["panda"].ik_GN(SE3.Rt(rot_last, xyz), q0=objects["panda"].qr, joint_limits=True, pinv=True)
    q = q[0]
    if q_last is None:
        q_last = q
    Final = SE3.Rt(rot_last, xyz)
    while any(objects["panda"].iscollided(q, instance_box) for instance_box in objects["box"]) or call.joints_changed_significantly(q, q_last, 0.5):
        rot = call.generate_orientation(rot_last)
        q_last = q
        q = objects["panda"].ik_GN(SE3.Rt(rot, xyz), q0=Final, joint_limits=True, pinv=True)
        q = q[0]
        Final = SE3.Rt(rot, xyz)
    panda = rtb.models.Panda()
    panda.q = q
    env.add(panda, collision_alpha=0.4, robot_alpha=0.1)
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