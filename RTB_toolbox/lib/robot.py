import numpy as np
import roboticstoolbox as rtb
from spatialmath import SO3, SE3
from lib.callbacks import Node, joints_changed_significantly, generate_orientation

def validate_robot(env, objects, node : Node, thresh=0.002, bias=0.1):
    print('Beggining the process of validating the robot pose...')
    robot = objects["panda"]
    xyz = node.pose()
    rot_last = node.rot_matrix
    q_base = robot.ik_GN(SE3.Rt(rot_last, xyz), q0=robot.qr, joint_limits=True, pinv=True)[0]
    q = q_base
    pose_base = robot.fkine(q_base)
    orient_tries = 1
    while (
        any(robot.iscollided(q, instance_box) for instance_box in objects["box"]) or \
        joints_changed_significantly(q, q_base, bias + thresh*orient_tries)
    ):
        rot = generate_orientation(rot_last, max_change=bias + orient_tries*thresh)
        q = robot.ik_GN(SE3.Rt(rot, xyz), q0=pose_base, joint_limits=True, pinv=True)[0]
        orient_tries += 1
    if np.isnan(q).any():
        q = np.nan_to_num(q, nan=0.0)
    panda = rtb.models.Panda()
    panda.q = q
    env.add(panda, collision_alpha=0.6, robot_alpha=0)
    return Node(node.x, node.y, node.z, SO3(robot.fkine(q).R), node.parent, node.cost, q)