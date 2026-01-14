"""
@file geometry.py
@package legarm_wbc
@author Xinyuan Liu (liuxinyuan872@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2026, Harbin Institute of Technology.
@date 2026-01
"""

import numpy as np

def positionPD(des_pos, cur_pos, 
               des_vel=np.zeros(3), cur_vel=np.zeros(3), 
               des_acc=np.zeros(3), 
               kp=np.array([100, 100, 100]), kd=np.array([0.0, 0.0, 0.0])):
    return kp*(des_pos-cur_pos) + kd*(des_vel-cur_vel) + des_acc

def near_zero(z):
    """Determines whether a scalar is small enough to be treated as zero

    Args: 
        z (float): A scalar input to check
    Returns:
        True if z is close to zero, false otherwise

    Example Input:
        z = -1e-7
    Output:
        True
    """
    return abs(z) < 1e-6

def normalize(vec):
    """Normalize a vector

    Args: 
        vec (ndarray): A vector
    Returns: 
        A unit vector pointing in the same direction as z

    Example Input:
        vec = np.array([1, 2, 3])
    Output:
        np.array([0.26726124, 0.53452248, 0.80178373])
    """
    return vec / np.linalg.norm(vec)

def vec_to_so3(omg):
    """Converts a 3-vector to an so(3) representation

    Args:
        omg (ndarray): A 3-vector
    Returns:
        A 3x3 skew-symmetric matrix

    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])

def so3_to_vec(so3mat):
    """Converts an so(3) representation to a 3-vector
    
    Args:
        so3mat: A 3x3 skew-symmetric matrix
    Returns: 
        The 3-vector corresponding to so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def axis_ang3(expc3):
    """Converts a 3-vector of exponential corordinates for rotation into
       axis-angle form.

    Args:
        expc3: A 3-vector of exponential coordinates for rotation
    Returns: 
        omghat: A unit rotation axis
        theta: The corresponding rotation angle

    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """
    return (normalize(expc3), np.linalg.norm(expc3))

def matrix_exp3(so3mat):
    """Computes the matrix exponential of a matrix in so(3)

    Args:
        so3mat: A 3x3 skew-symmetrix matrix
    Returns:
        The matrix exponential of so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
    """
    omgtheta = so3_to_vec(so3mat)
    if near_zero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = axis_ang3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def matrix_log3(R):
    """Computes the matrix logarithm of a rotation matrix
    
    Args:
        mat: A 3x3 rotation matrix
    Returns:
        The matrix logarithm of R

    Example Input:
        R = np.array([[0, 0, 1], 
                      [1, 0, 0], 
                      [0, 1, 0]])
    Output: 
        np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
    """
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not near_zero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not near_zero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return vec_to_so3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def rotationPD(des_rot, cur_rot, 
               des_omega=np.zeros(3), cur_omega=np.zeros(3), 
               des_omega_dot=np.zeros(3), 
               kp=np.array([100, 100, 100]), kd=np.array([0.0, 0.0, 0.0])):
    return (kp * cur_rot.dot(so3_to_vec(matrix_log3(cur_rot.T.dot(des_rot)))) + 
        kd * (des_omega - cur_omega) + des_omega_dot)