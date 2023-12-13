###
### This homework is modified from CS231.
###


import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, sigma, VT = np.linalg.svd(E)
    Q_1 = U.dot(W).dot(VT)
    Q_2 = U.dot(W.T).dot(VT)
    R_1 = np.dot(np.linalg.det(Q_1),Q_1)
    R_2 = np.dot(np.linalg.det(Q_2),Q_2)
    T_1 = U[:, 2]
    T_2 = U[:, 2]*(-1)
    RT = np.zeros((4, 3, 4))
    RT[0, :, :] = np.hstack((R_1, np.expand_dims(T_1.T, axis=1)))
    RT[1, :, :] = np.hstack((R_1, np.expand_dims(T_2.T, axis=1)))
    RT[2, :, :] = np.hstack((R_2, np.expand_dims(T_1.T, axis=1)))
    RT[3, :, :] = np.hstack((R_2, np.expand_dims(T_2.T, axis=1)))

    return RT

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    linear_matrix = np.zeros((image_points.shape[0]*2, camera_matrices.shape[-1]))
    for i in range(image_points.shape[0]) :
        pi = image_points[i]
        Mi = camera_matrices[i]
        u = pi[0]
        v = pi[1]
        Mi_1 = Mi[0, :]
        Mi_2 = Mi[1, :]
        Mi_3 = Mi[2, :]
        linear_matrix[i*2, :] = v*Mi_3 - Mi_2
        linear_matrix[i*2+1, :] = Mi_1 - u*Mi_3

    U, sigma, VT = np.linalg.svd(linear_matrix)
    P = VT[-1, :]
    P = (P / P[-1])[:-1]

    return P

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2M reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    P = np.zeros(point_3d.shape[0]+1)
    P[:-1] = point_3d
    error = np.zeros(image_points.shape[0]*2)
    for i in range(image_points.shape[0]) :
        Mi_1 = camera_matrices[i][0, :]
        Mi_2 = camera_matrices[i][1, :]
        Mi_3 = camera_matrices[i][2, :]
        X = P[0]
        Y = P[1]
        Z = P[2]
        y1 = X * Mi_1[0] + Y * Mi_1[1] + Z * Mi_1[2] + Mi_1[3]
        y2 = X * Mi_2[0] + Y * Mi_2[1] + Z * Mi_2[2] + Mi_2[3]
        y3 = X * Mi_3[0] + Y * Mi_3[1] + Z * Mi_3[2] + Mi_3[3]
        pi = np.array([y1/y3, y2/y3])
        error[i*2] = pi[0] - image_points[i][0]
        error[i*2+1] = pi[1] - image_points[i][1]
    
    return error

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # jacobian matrix
    J = np.zeros((camera_matrices.shape[0]*2, point_3d.shape[0]))
    P = np.zeros(point_3d.shape[0]+1)
    P[:-1] = point_3d
    for i in range(camera_matrices.shape[0]) :
        Mi_1 = camera_matrices[i][0, :]
        Mi_2 = camera_matrices[i][1, :]
        Mi_3 = camera_matrices[i][2, :]
        X = P[0]
        Y = P[1]
        Z = P[2]
        y1 = X * Mi_1[0] + Y * Mi_1[1] + Z * Mi_1[2] + Mi_1[3]
        y2 = X * Mi_2[0] + Y * Mi_2[1] + Z * Mi_2[2] + Mi_2[3]
        y3 = X * Mi_3[0] + Y * Mi_3[1] + Z * Mi_3[2] + Mi_3[3]

        # multiply by y3 first, then divide by y3**2, to avoid wrong answer
        J[i*2, 0] = (Mi_1[0]*y3 - Mi_3[0]*y1)/y3**2      
        J[i*2, 1] = (Mi_1[1]*y3 - Mi_3[1]*y1)/y3**2
        J[i*2, 2] = (Mi_1[2]*y3 - Mi_3[2]*y1)/y3**2
        J[i*2+1, 0] = (Mi_2[0]*y3 - Mi_3[0]*y2)/y3**2
        J[i*2+1, 1] = (Mi_2[1]*y3 - Mi_3[1]*y2)/y3**2
        J[i*2+1, 2] = (Mi_2[2]*y3 - Mi_3[2]*y2)/y3**2

    return J


'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    point_3d = linear_estimate_3d_point(image_points, camera_matrices)
    for i in range(10) :
        error = reprojection_error(point_3d, image_points, camera_matrices)
        J = jacobian(point_3d, camera_matrices)
        point_3d = point_3d - np.linalg.inv(J.T @ J) @ J.T @ error

    return point_3d


'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''


def estimate_RT_from_E(E, image_points, K):
    estimated_RT = estimate_initial_RT(E)
    max_count = -1
    correct_RT = None
    for i in range(estimated_RT.shape[0]) :
        count = 0
        RT = estimated_RT[i]
        camera_matrices = np.zeros((2, 3, 4))
        camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))

        R = RT[:,:3]
        T = RT[:,3:]
        new_RT = np.concatenate((R.T, -R.T.dot(T)), axis=1)
        camera_matrices[1, :, :] = K.dot(new_RT)

        for j in range(image_points.shape[0]) :
            point_3d_camera1_coor = nonlinear_estimate_3d_point(image_points[j], camera_matrices)

            # convert camera1 coordinate to camera2 coordinate
            point_3d_tmp = np.ones((4,1))
            point_3d_tmp[0:3, :] = point_3d_camera1_coor.reshape((3,1))
            point_3d_camera2_coor = new_RT.dot(point_3d_tmp)

            if point_3d_camera1_coor[2] > 0 and point_3d_camera2_coor[2] > 0 :
                count += 1

        if count > max_count :
            max_count = count
            correct_RT = RT

    return correct_RT

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    np.save('results.npy', dense_structure)
    print ('Save results to results.npy!')
