import os
import random
import csv
import numpy as np
from spatialgeometry import Sphere, Cuboid, CollisionShape
from spatialmath import SO3, SE3
import swift
import roboticstoolbox as rtb
from roboticstoolbox.tools import jtraj, quintic
import math

COLL_BOX_RADIUS_MULTIPLIER = 2
COLL_ROBOT_RADIUS_MULTIPLIER = 3

class Node:
    def __init__(self, x, y, z, rot_matrix = None, parent=None, cost=0.0, q=None):
        self.x = x
        self.y = y
        self.z = z
        self.rot_matrix = rot_matrix
        self.parent = parent
        self.cost = cost
        self.q = q 

    def pose(self):
        return np.array([self.x, self.y, self.z])

    def __lt__(self, other):
        return self.cost < other.cost  # needed for heapq

def handle_path(file):
    """
    Generate the full path for a file located in the 'Resources' directory.

    This function takes a file name as an argument and returns the full path
    for the file, considering its potential location within the 'Resources'
    directory. If the 'Resources' directory is present in the current working
    directory, the path is constructed accordingly. If not, it looks for the
    'Resources' directory in the parent directory.

    Parameters:
    - file (str): The name of the file.

    Returns:
    - str: The full path for the specified file.
    """
    current_path = os.getcwd()

    # Check if 'Resources' directory exists in the current working directory
    if os.path.exists(os.path.join(current_path, "Resources")):
        _PATH = os.path.join(current_path, "Resources", file)
    else:
        # If 'Resources' directory is not in the current working directory,
        # look for it in the parent directory
        parent_path = os.path.dirname(current_path)
        if os.path.exists(os.path.join(parent_path, "Resources")):
            _PATH = os.path.join(parent_path, "Resources", file)
            
    return _PATH

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in 3D space.

    Parameters:
    - point1: NumPy array containing (x, y, z) coordinates of the first point.
    - point2: NumPy array containing (x, y, z) coordinates of the second point.

    Returns:
    - distance: Euclidean distance between the two points.
    """
    # Calculate the Euclidean distance
    distance = np.linalg.norm(point2 - point1)
    
    return distance

def generate_point(p, radius = 0.03):
    """
    Generate a random point in 3D space around a given point.

    This function generates a random point in 3D space around the specified point
    'p'. The point is randomly positioned within a spherical region of radius 0.1.

    Parameters:
    - p (list or ndarray): The coordinates of the center point in 3D space.

    Returns:
    - list: A list containing the x, y, and z coordinates of the generated point.
    """
    # Generate random spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)

    # Convert spherical coordinates to Cartesian coordinates
    x = p[0] + radius * np.sin(phi) * np.cos(theta)
    y = p[1] + radius * np.sin(phi) * np.sin(theta)
    z = p[2] + radius * np.cos(phi)

    return [x, y, z]

def joints_changed_significantly(q1, q2, threshold=0.01):
    """
    Check if the values on every joint did not change much.
    Parameters:
    q1, q2: Arrays representing joint values.
    threshold: Maximum allowed change for each joint.
    Returns:
    True if any joint changed significantly, False otherwise.
    """
    # Use different thresholds for the first 3 joints and the last 3 joints
    thresholds = [threshold/2] * 3 + [min(threshold, 3.14)] * (len(q1) - 3)
    return any(abs(a - b) > t for a, b, t in zip(q1, q2, thresholds))

def generate_orientation(last_orientation=None, max_change=2*np.pi, limits=None):
    """
    Generate a random orientation in 3D space with a constraint on the maximum change and limits.

    Parameters:
    - last_orientation (SO3, optional): The last orientation as an SO3 object. Default is None.
    - max_change (float, optional): The maximum allowable change in radians for each angle.
    - limits (list or tuple, optional): List/tuple of (min, max) for roll, pitch, yaw in radians.

    Returns:
    - SO3: A new orientation as an SO3 object.
    """
    # Default limits if not provided: roll [-pi, pi], pitch [-pi/2, pi/2], yaw [-pi, pi]
    if limits is None:
        limits = [(-np.pi, np.pi), (-np.pi/2, np.pi/2), (-np.pi, np.pi)]

    if last_orientation is None:
        roll = np.random.uniform(limits[0][0], limits[0][1])
        pitch = np.random.uniform(limits[1][0], limits[1][1])
        yaw = np.random.uniform(limits[2][0], limits[2][1])
    else:
        last_rpy = last_orientation.rpy(order='xyz', unit='rad')
        roll = np.clip(
            np.random.uniform(last_rpy[0] - max_change, last_rpy[0] + max_change),
            limits[0][0], limits[0][1]
        )
        pitch = np.clip(
            np.random.uniform(last_rpy[1] - max_change, last_rpy[1] + max_change),
            limits[1][0], limits[1][1]
        )
        yaw = np.clip(
            np.random.uniform(last_rpy[2] - max_change, last_rpy[2] + max_change),
            limits[2][0], limits[2][1]
        )

    return SO3.RPY([roll, pitch, yaw], order='xyz', unit='rad')

# def generate_orientation():
#     """
#     Generate a random orientation in 3D space.
#     This function generates a random orientation represented as roll, pitch, and yaw
#     angles. The angles are uniformly distributed within their respective ranges.
#     Returns:
#     - list: A list containing the roll, pitch, and yaw angles in radians.
#     """
#     roll = np.random.uniform(-np.pi, np.pi)
#     pitch = np.random.uniform(-np.pi / 2, np.pi / 2)
#     yaw = np.random.uniform(-np.pi, np.pi)
#     return SO3.RPY([roll, pitch, yaw])

def update_obj(box: CollisionShape, pos: list):
    """
    Update the position of a CollisionShape object.

    This function updates the position of a given CollisionShape object, 'box',
    by applying a rotation and translation based on the specified position.

    Parameters:
    - box (CollisionShape): The CollisionShape object to be updated.
    - pos (list): A list containing the x, y, and z coordinates of the new position.

    Returns:
    - None: The function updates the position of the CollisionShape object in-place.
    """
    axis = SO3.Rx(0) @ SO3.Ry(0)
    box.T = SE3.Rt(axis, pos)

def setup_env(**kwargs):
    """
    Set up a Swift environment with specified objects.

    This function sets up a Swift environment with various objects, including a start
    sphere, a destination sphere, cuboid boxes, and a Panda robot. The objects are
    positioned and colored based on the provided keyword arguments.

    Parameters:
    - **kwargs: Keyword arguments to customize the environment setup. Possible keys include:
        - "start": Boolean indicating whether to include a start sphere.
        - "dest": Boolean indicating whether to include a destination sphere.
        - "boxes": Boolean indicating whether to include cuboid boxes.
        - "panda": Boolean indicating whether to include a Panda robot.
        - "resources": Dictionary containing additional information for object customization.

    Returns:
    - tuple: A tuple containing a dictionary of created objects and the Swift environment.
    """
    env = swift.Swift()
    env.launch(realtime=True, reload=True)

    objects = {}
    
    if "panda" in kwargs and kwargs["panda"]:
        panda = rtb.models.Panda()
        panda.q = panda.qr
        env.add(panda)
        objects["panda"] = panda
    if "boxes" in kwargs and "resources" in kwargs and kwargs["boxes"]:
        box = [Cuboid(scale=_scale, collision=True, color=(255, 10, 10)) for _scale in kwargs["resources"]["box_info"][:, 0:3] / 10]
        pos = [[_xyz[0], _xyz[1], _xyz[2]] for _xyz in kwargs["resources"]["box_info"][:, 3:6]]
        for i, _ in enumerate(box):
            update_obj(box[i], pos[i])
            # Check for collisions with already generated boxes
            if all(not box[i].iscollided(b) for b in box[:i]) and not panda.iscollided(panda.qr, box[i]):
                env.add(box[i])
        objects["box"] = box
    if "start" in kwargs and "resources" in kwargs and kwargs["start"]:
        start_coll_robot = Sphere(radius=COLL_ROBOT_RADIUS_MULTIPLIER*kwargs["resources"]["radius"], color=kwargs["resources"]["start_color"])
        start_coll_box = Sphere(radius=COLL_BOX_RADIUS_MULTIPLIER*kwargs["resources"]["radius"], color=kwargs["resources"]["start_color"])
        start = Sphere(radius=kwargs["resources"]["radius"], color=kwargs["resources"]["start_color"])
        loc = [np.random.uniform(kwargs["resources"]["limits"][index][0], kwargs["resources"]["limits"][index][1]) for index, _ in enumerate(kwargs["resources"]["limits"])]
        if "start_loc" in kwargs["resources"]:
            loc = kwargs["resources"]["start_loc"][:3]
        update_obj(start, loc)
        update_obj(start_coll_robot, loc)
        update_obj(start_coll_box, loc)
        while not all(not start_coll_box.iscollided(b) for b in box) or (panda.iscollided(panda.qr, start_coll_robot)):
            loc = [np.random.uniform(kwargs["resources"]["limits"][index][0], kwargs["resources"]["limits"][index][1]) for index, _ in enumerate(kwargs["resources"]["limits"])]
            update_obj(start, loc)
            update_obj(start_coll_robot, loc)       
            update_obj(start_coll_box, loc)
            # env.add(start_coll_robot)
        # Check for collisions with boxes
        env.add(start)
        objects["start"] = start
    if "dest" in kwargs and "resources" in kwargs and kwargs["dest"]:
        dest_coll_robot = Sphere(radius=COLL_ROBOT_RADIUS_MULTIPLIER*kwargs["resources"]["radius"], color=kwargs["resources"]["dest_color"])
        dest_coll_box = Sphere(radius=COLL_BOX_RADIUS_MULTIPLIER*kwargs["resources"]["radius"], color=kwargs["resources"]["dest_color"])
        dest = Sphere(radius=kwargs["resources"]["radius"], color=kwargs["resources"]["dest_color"])
        loc = [np.random.uniform(kwargs["resources"]["limits"][index][0], kwargs["resources"]["limits"][index][1]) for index, _ in enumerate(kwargs["resources"]["limits"])]
        if "dest_loc" in kwargs["resources"]:
            loc = kwargs["resources"]["dest_loc"][:3]
        update_obj(dest, loc)
        update_obj(dest_coll_robot, loc)
        update_obj(dest_coll_box, loc)
        while not all(not dest_coll_box.iscollided(b) for b in box) or (panda.iscollided(panda.qr, dest_coll_robot)):
            loc = [np.random.uniform(kwargs["resources"]["limits"][index][0], kwargs["resources"]["limits"][index][1]) for index, _ in enumerate(kwargs["resources"]["limits"])]
            update_obj(dest, loc)
            update_obj(dest_coll_robot, loc)       
            update_obj(dest_coll_box, loc)
            # env.add(start_coll_robot)
        # Check for collisions with boxes
        env.add(dest)
        objects["dest"] = dest
    return objects, env



def generate_csv(filename, **kwargs):
    """
    Generate a CSV file with customizable content.

    This function creates a CSV file with the specified filename and content based
    on the provided keyword arguments. The content can include headers, random data,
    or an array of values.

    Parameters:
    - filename (str): The name of the CSV file to be generated.

    Keyword Arguments:
    - headers (list, optional): A list of headers for the CSV file.
    - random (int, optional): The number of rows of random data to generate.
    - limits (list, optional): A list of tuples specifying the limits for random data generation.
    - array (list, optional): A list of objects to be written to the CSV file.

    Returns:
    - None: The function writes the generated data directly to the specified CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if "headers" in kwargs:
            csv_writer.writerow(kwargs['headers'])

        if "random" in kwargs and "limits" in kwargs and kwargs["random"] > 0:
            for _ in range(kwargs['random']):
                row = [random.uniform(kwargs['limits'][index][0], kwargs['limits'][index][1]) for index, _ in enumerate(kwargs['limits'])]
                csv_writer.writerow(row)
        if "array" in kwargs:
            for index, object in enumerate(kwargs['array']):
                x = object[0]
                y = object[1]
                z = object[2]

                r = object[3]
                p = object[4]
                y = object[5]


                # row = [x, y, z, r, p, y]
                row = [object[0], object[1], object[2], object[3], object[4], object[5]]
                csv_writer.writerow(row)

def generate_random_locs(amount: int):
    """
    Generate random locations in 3D space.

    This function generates a specified number of random locations in 3D space.
    The locations are represented as a NumPy array where each row corresponds to
    a location with x, y, and z coordinates.

    Parameters:
    - amount (int): The number of random locations to generate.

    Returns:
    - ndarray: A NumPy array containing random locations in 3D space.
    """
    rand = np.empty([amount,3])
    for i in range(amount):
        rand[i] = [random.uniform(-0.5, 0)*_ for _ in np.ones(3)]
    return rand

def try_positions(Togo, cnt, resources, objects, env, Temp, search_radius=0.1):
    """
    Attempts to generate a new position for the robot, checking for collisions and proximity to the destination.
    Updates Temp if a better, collision-free position is found.
    Returns the updated Temp.
    """
    in_collision = False
    for i in range(resources["iterations"]):
        best_pose = Togo[cnt].T[0:3,3]
        center = generate_point(best_pose, radius=search_radius)
        current = Sphere(radius=resources["radius"], color=(10,10,10))
        current_coll_robot = Sphere(radius=3*resources["radius"], color=(10,10,10))
        current_coll_box = Sphere(radius=3*resources["radius"], color=(10,10,10))
        update_obj(current, center)
        update_obj(current_coll_robot, center)
        update_obj(current_coll_box, center)
        for instance_box in objects["box"]:
            if current_coll_box.iscollided(instance_box) or \
            objects["panda"][0].iscollided(current_coll_robot) or \
            objects["panda"][1].iscollided(current_coll_robot) or \
            objects["panda"][2].iscollided(current_coll_robot) or \
            objects["panda"][3].iscollided(current_coll_robot):
                in_collision = True
                break
            else:
                in_collision = False
        # env.add(current_coll_robot)
        env.add(current)
        if (euclidean_distance(current.T[0:3,3], objects['dest'].T[0:3,3]) < euclidean_distance(Temp.T[0:3,3], objects['dest'].T[0:3,3])) and not in_collision:
            if Temp!=objects['start']:
                env.remove(Temp)
            Temp = current
            if objects['dest'].iscollided(Temp):
                break
        else:
            env.remove(current)
    return Temp

def robot_move(objects, env: swift.Swift, points: list, joint_q=False):
    """
    Move a robot to a series of specified points in Cartesian space.

    This function controls the movement of a robot to a sequence of specified points
    in Cartesian space. The robot uses a proportional control scheme to adjust its
    joint velocities and reach the desired positions.

    Parameters:
    - robot (rtb.models): The roboticstoolbox robot model.
    - env (swift.Swift): The Swift environment for robot simulation.
    - points (list): A list of 3D points representing the desired end-effector positions.

    Returns:
    - None: The function controls the robot's movement in the specified environment.
    """
    robot : rtb.models.Panda = objects["panda"]
    dt = 0.01

    sum2 = 0

    if joint_q:
        time = np.array([i*dt for i in range(0, 100)])
        for _,i in enumerate(points):
            # traj = quintic(robot.q,  i, time)
            traj = jtraj(robot.q,  i, time)
            for vel in traj.qd:
                vel = np.nan_to_num(vel, nan=0.0)
                robot.qd = vel
                env.step(dt)
                # for instance_box in objects["box"]:
                #     if robot.iscollided(robot.q, instance_box) == True:
                #         sum2 += 1
                #         print(f'Robot in collision: {sum2}')
    else:
        arrived = False
        for _,i in enumerate(points):
            while not arrived:
                v, arrived = rtb.p_servo(robot.fkine(robot.q), SE3.Rt(SO3.RPY(i[3:], order='xyz', unit='deg'),i[:3]), gain=0.1, threshold=0.01)
                robot.qd = np.linalg.pinv(robot.jacobe(robot.q)) @ v
                env.step(dt)
            for instance_box in objects["box"]:
                if robot.iscollided(robot.q, instance_box) == True:
                    sum2 += 1
                    print(f'Robot in collision: {sum2}')
            
            arrived = False