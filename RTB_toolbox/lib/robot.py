import numpy as np
import roboticstoolbox as rtb
from spatialmath import SO3, SE3
from lib.callbacks import Node, joints_changed_significantly, generate_orientation

def validate_robot(env, objects, node : Node, prev_node : Node, thresh=0.002, bias=0.1, rotation_limits=None):
    print('Beggining the process of validating the robot pose...')
    robot = objects["panda"]
    xyz = node.pose()
    rot_last = prev_node.rot_matrix
    q_base = prev_node.q if all(prev_node.q) else robot.ik_GN(SE3.Rt(rot_last, xyz), q0=robot.qr, joint_limits=True, pinv=True)[0]
    pose_base = robot.fkine(q_base)
    joints_bug, collision_bug, orient_tries = 0, 0, 0
    in_collision, joints_exceeded = True, True
    while in_collision or joints_exceeded:
        rot = generate_orientation(rot_last, max_change=5*bias, limits=rotation_limits)
        q = robot.ik_GN(SE3.Rt(rot, xyz), q0=pose_base, joint_limits=True, pinv=True)[0]
        in_collision = any(robot.iscollided(q, instance_box) for instance_box in objects["box"])
        joints_exceeded = joints_changed_significantly(q_base, q, bias + orient_tries*thresh)
        collision_bug += 1 if in_collision else 0
        joints_bug += 1 if joints_exceeded else 0
        orient_tries += 1
        # if all(node.pose() == objects["dest"].T[:3,3]):
        #     orient_tries = 5000

    if np.isnan(q).any():
        q = np.nan_to_num(q, nan=0.0)
    print(f'Robot pose validated after \n{orient_tries} tries with \n\t{joints_bug} joints bugs and\n\t{collision_bug} collision bugs when joints ok. \nThis results in thresh of {bias + orient_tries*thresh}.')
    panda = rtb.models.Panda()
    panda.q = q
    env.add(panda, collision_alpha=0.6, robot_alpha=0)
    return Node(node.x, node.y, node.z, SO3(robot.fkine(q).R), node.parent, node.cost, q)