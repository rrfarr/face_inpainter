#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:09:48 2017

@author: reubenfarrugia
"""

import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import csv
from scipy.spatial import Delaunay
from collections import namedtuple
from skimage import feature
import time
from matplotlib import path
from scipy import ndimage
import profile

# This function simply clears the console
def cls():
    # This function is simply used to clear the screen of the console
    os.system('cls' if os.name == 'nt' else 'clear')

###############################################################################
# Face Warping package
###############################################################################
#def sign(p1, p2, p3):
#  return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


#def PointInAABB(pt, c1, c2):
#  return c2[0] <= pt[0] <= c1[0] and \
#         c2[1] <= pt[1] <= c1[1]


#def PointInTriangle(pt, v1, v2, v3):
#  b1 = sign(pt, v1, v2) < 0.0
#  b2 = sign(pt, v2, v3) < 0.0
#  b3 = sign(pt, v3, v1) < 0.0

#  return ((b1 == b2) & (b2 == b3))
  #return ((b1 == b2) and (b2 == b3)) and \
  #       PointInAABB(pt, map(max, v1, v2, v3), map(min, v1, v2, v3))    
def delaunay_segmentation(I,tri):
    print('Segment image using Delaunay segmentation ...')
    
    # Determine the dimensions of the image
    sz = np.asarray(I).shape                   
    
    #print(tri.points[tri.simplices[0]]) 
    #print(np.reshape(np.indices([sz[0],sz[1]]),[sz[0]*sz[1],2]))
              
    #mask = matplotlib.nxutils.points_inside_poly(np.indices([sz[0],sz[1]]), tri.points[tri.simplices[0]])  
    #mask = matplotlib.path.Path.contains_points(np.indices([sz[0],sz[1]]), tri.points[tri.simplices[0]])  
    
    # Compute the grid of indeces to be considered for the mask
    xv,yv = np.meshgrid(np.linspace(0,sz[0]-1,sz[0]),np.linspace(0,sz[1]-1,sz[1]))
    
    xv = xv.flatten().astype('int')
    yv = yv.flatten().astype('int')
    #seg = np.zeros((sz[0],sz[1]),dtype='bool')
    seg = -np.ones((sz[0],sz[1]))
    
    for n in range(len(tri.simplices)):
        
        # Define the polygon
        p = path.Path(tri.points[tri.simplices[n]])
   
        # Check which pixels are contained in the triangle
        flags = p.contains_points(np.hstack((xv[:,np.newaxis],yv[:,np.newaxis])))
        
        # Determine the grid
        grid = np.zeros((sz[0],sz[1]),dtype='bool')
        grid[xv,yv] = flags
               
        # Set those which are true to this triangle
        coord = np.nonzero(grid == 1)
        seg[coord] = n   

    return seg

def affine_transform_estimation(src_x,src_y, dst_x,dst_y):
    
    # Get the coefficinets from the source points    
    v1 = src_x[0]
    v2 = src_x[1]
    v3 = src_x[2]
    w1 = src_y[0]
    w2 = src_y[1]
    w3 = src_y[2]
    
    # Get the coefficients from the destination points
    x1 = dst_x[0]
    x2 = dst_x[1]
    x3 = dst_x[2]
    y1 = dst_y[0]
    y2 = dst_y[1]
    y3 = dst_y[2]
    
    # Determine the inverse of the matrix to be used to compute affine trans
    Beta = np.linalg.pinv(np.array([[v1, w1, 1],[v2, w2, 1], [v3,w3,1]]))
    
    #t1 = Beta * np.array([x1,x2,x3])
    #t2 = Beta * np.array([y1,y2,y3])
        
    t1 = Beta.dot(np.array([[x1],[x2],[x3]]))
    t2 = Beta.dot(np.array([[y1],[y2],[y3]]))
    
    # Get the individual coefficients of the affine transformation
    t11 = t1[0,0]
    t21 = t1[1,0]
    t31 = t1[2,0]
    t12 = t2[0,0]
    t22 = t2[1,0]
    t32 = t2[2,0]
        
    # Derive the forward affine transformation
    Tforward = np.array([[t11,t12,0],[t21,t22,0],[t31,t32,1]])

    # Compute the inverse transform of the affine transformation
    Tinverse = np.linalg.pinv(Tforward)
    
    return Tforward,Tinverse

def is_green(I,coord,i):
    if  (I[coord[i,1],coord[i,0],0].astype(int) == 0) & (I[coord[i,1],coord[i,0],1].astype(int) == 255) & (I[coord[i,1],coord[i,0],2].astype(int) == 0):
        return 1
    return 0
        
def warp(I,I_warp,coord,new_coord):
    # Determine the number of pixels to consider
    Ncoord = len(coord)
    
    new_coord = new_coord.astype(int)
    coord     = coord.astype(int)
    
    # Determine the dimension of the target image
    img_size = I_warp.shape
    for i in range(Ncoord):
        if (new_coord[i,1] >= img_size[1]) | (new_coord[i,1] < 0) | (new_coord[i,0] >= img_size[0]) | (new_coord[i,0] < 0):
            continue # target coordinate is out of range => ignore

        # If the source pixel is green set the target pixel to green  
        if is_green(I,coord,i):           
            # This pixel is marked in green
            I_warp[new_coord[i,1],new_coord[i,0],:] = np.array([0, 255, 0])
            continue
        
        # If the target pixel is green keep it green
        if is_green(I_warp,new_coord,i):
            I_warp[new_coord[i,1],new_coord[i,0],:] = np.array([0, 255, 0])
            continue
        
        if I_warp[new_coord[i,1],new_coord[i,0],0] == -1:
            #print(new_coord[i,:])
            #print(coord[i,:])
            # Put the source pixel in the warped image
            I_warp[new_coord[i,1],new_coord[i,0],:] = I[coord[i,1],coord[i,0],:]
        else:
            # Average the source pixel and target pixel
            I_warp[new_coord[i,1],new_coord[i,0],:] = (I_warp[new_coord[i,1],new_coord[i,0],:] + I[coord[i,1],coord[i,0],:])/2
    return I_warp    
    
def forward_face_registration(Iface, landmarks, model_landmarks,sz):
    
    # Pring messages on console
    print('Register face to the reference frontal face model ...')
    print('-----------------------------------------------------')
    print('Compute Delaunay triangulation ...')
    
    # Compute the Delaunay triangularization
    TRI_d = Delaunay(landmarks)
    
    # Extract the coordinates of the source image
    Xd = landmarks[:,0]
    Yd = landmarks[:,1]
    
    Xr = []
    Yr = []
        
    for i in range(len(model_landmarks)):
        Xr.append(model_landmarks[i][1])
        Yr.append(model_landmarks[i][2])
    # Convert list to an array    
    Xr = np.asarray(Xr) - 1 # Model is derived using MATLAB
    Yr = np.asarray(Yr) - 1 # Model is derived using MATLAB  
    
    
    #--------------------------------------------------------------------------
    # Show the input image
    plt.imshow(np.uint8(Iface))
 
    plt.triplot(landmarks[:,0],landmarks[:,1],TRI_d.simplices.copy())
    plt.plot(landmarks[:,0],landmarks[:,1],'o')
    plt.show()
    #--------------------------------------------------------------------------
               
    # Determine the number of triangles
    Ntri = len(TRI_d.simplices)
        
    # Create the parameter structure
    ParamStruct = namedtuple('ParamStruct','TRI_d Xd Yd img_seg Tforward_list')
    t = time.time()


    # Segment the face image using delaunay
    img_seg = delaunay_segmentation(Iface,TRI_d)
    
    elapsed = time.time() - t
    print('Img Seg took %fs' %(elapsed))       

    # Initialize Iwarp
    I_warp = -np.ones((sz[0],sz[1],3)) # Assume color images
    
    Tforward_list = []
    Tinverse_list = []
    print('Compute the face frontalization ...')
    t = time.time()
    for n in range(Ntri):
        # Get the vertex coordinates of the source triangle
        t1_x = np.transpose(Xd[TRI_d.simplices[n]])
        t1_y = np.transpose(Yd[TRI_d.simplices[n]])
        
        # Get tge vertex ciirdubates if tge destubatuib truabgke
        t2_x = np.transpose(Xr[TRI_d.simplices[n]])
        t2_y = np.transpose(Yr[TRI_d.simplices[n]])
                
        # Derive the forward and inverse affine transformation
        Tforward, Tinverse = affine_transform_estimation(t1_x, t1_y,t2_x,t2_y)
        
        # Get the coordinates from source image belonging to segment n
        coord1 = np.nonzero(img_seg == n)
                
        # Convert the tuple into an array
        coord1 = np.asarray(coord1).transpose()
                       
        # Add a column of 1s to be able to compute the matrix multiplication
        coord1 = np.concatenate((coord1,np.ones((len(coord1),1))),axis=1)
            
        # Determine the target coordinates using affine transformation
        coord2 = (coord1.dot(Tforward)).round()  
                
        # Warp the pixels from the source triangle onto the target triangle
        I_warp = warp(Iface,I_warp,coord1,coord2)
        
        Tforward_list.append(Tforward)
        Tinverse_list.append(Tinverse)
    
    I_warp[I_warp[:,:,0] == -1] = np.array([0, 255, 0])  
    elapsed = time.time() - t
    print('F. Warping took %fs' %(elapsed))       

    # Put the delaunay tri into the parameter structrue
    param = ParamStruct(TRI_d,Xd,Yd,img_seg,Tforward_list)

    return I_warp, param
###############################################################################

###############################################################################
# Face Inpainting package
###############################################################################

def gradient(f):
    # Determine the shape of the image
    sz = f.shape
    
    # Initialize the       
    gx = np.zeros(sz)
    gy = np.zeros(sz)
        
    # Compute gradient in x direction
    for i in range(sz[0]):
        if i == 0:
            # Forward-difference
            gx[i,:] = f[i,:] - f[0,:]
        elif i == sz[0]-1:
            # Forward-difference
            gx[i,:] = f[sz[0]-1,:] - f[sz[0]-2,:]
        else:
            # Centered difference
            gx[i,:] = (f[i+1,:] - f[i-1,:])/2
                       
    # Compute gradient in y direction
    for i in range(sz[1]):
        if i == 0:
            # Forward-difference
            gy[:,i] = f[:,1] - f[:,0]
        elif i == sz[1]-1:
            # Forward difference
            gy[:,i] = f[:,sz[1]-1] - f[:,sz[1]-2]
        else:
            # Centered difference
            gy[:,i] = (f[:,i+1] - f[:,i-1])/2
    return gx,gy

def get_patch(sz, coord, patch_size):
    # Determine the number of pixels from the center
    w = int((patch_size - 1)/2)
    
    rows = np.zeros([1,2],'int')
    cols = np.zeros([1,2],'int'
                    )
    # Get the coordinates of the patch
    x = int(coord[0])
    y = int(coord[1])
    
    x_min = x - w
    if x_min < 0:
        x_min = 0
    x_max = x+w+1
    if x_max >= sz[0]:
        x_max = sz[0]
    y_min = y-w
    if y_min < 0:
        y_min = 0
    y_max = y+w+1
    if y_max >= sz[1]:
        y_max = sz[1]
    # Derive the affected rows and columns    
    #rows = slice(max(x-w,0),min(x+w+1,sz[0]))
    #cols = slice(max(y-w,0),min(y+w+1,sz[1]))    
    rows = slice(x_min,x_max)
    cols = slice(y_min,y_max)
    
    return rows, cols

def  lle_prediction(XT, XS):
    
    tol = 1E-4
    
    # Determine the size of the dictionary
    sz = XS.shape
        
    # Compute the difference between each atom and the test sample
    z = np.zeros([sz[0]*sz[1],sz[2]])     
    for i in range(sz[2]):
        # Compute the difference between the atom and the test sample
        z_i = np.reshape(XS[:,:,i],[sz[0]*sz[1],1]) - np.reshape(XT,[sz[0]*sz[1],1])
        
        # Put the result in the array
        z[:,i] = z_i.reshape(sz[0]*sz[1],)
    # Compute the covariance matrix
    C = z.transpose().dot(z)
    
    # Determine the number of atoms in dictionary
    K = sz[2]
    # Regularize the covariance matrix to make it invertible
    if np.trace(C) == 0:
        C = C + np.identity(K)*tol
    else:
        C = C + np.identity(K)*tol*np.trace(C)
    # Compute the weights inv(C)*1^T
    W = np.linalg.inv(C).dot(np.ones([K,1]))
    # Enforce sum to unity
    W = W/sum(W)
    
    return W
    
def face_hallucination(img_patch, msk_patch, patch_dict,k):        
    # Find the indices of known pixels
    omega = msk_patch == 1
        
    # Extract the known pixels
    X_omega = img_patch[omega]
    
    # Extract the known pixel dictionary
    D_omega = patch_dict[omega]
    
    # Determine the number of entries in dictionary
    sz = D_omega.shape
    
    Nimgs = sz[2]

    dist = np.zeros([1,Nimgs])
    X_omega_v = X_omega.flatten()
    for n in range(Nimgs):
        # Compute the sum of absolute distance between the known part of the patch 
        # and corresponding known part in the dictionary
        #dist[0,n] = sum(sum(abs(X_omega - D_omega[:,:,n])))
        dist[0,n] = np.sum(np.abs(X_omega_v-D_omega[:,:,n].flatten()))
 
    # Find the patch with pargest priority
    idx = np.argpartition(dist,k)
    
    # Select only the k nearest neighbours
    idx = idx[:,0:k]
    
    # Extract the sub-dictionary using only the k-nearest neighbours
    D_omega_k = D_omega[:,:,idx[0,:]]
    
    # Derive the optimal weighted combination of known pixels
    alpha = lle_prediction(X_omega, D_omega_k)
    
    # Derive the dictionary of unknown pixels
    D_omega_hat = patch_dict[~omega]
    
    # Consider the sub-dictionary of corresponding k atoms
    D_omega_hat_k = D_omega_hat[:,:,idx[0,:]]
    
    # Determine the unknown pixels
    X_omega_hat = D_omega_hat_k.dot(alpha)
    
    # Clip the restored pixels
    X_omega_hat[X_omega_hat < 0] = 0
    X_omega_hat[X_omega_hat > 255] = 255
    

    # Initialize the reconstructed patch
    prd = np.zeros(img_patch.shape)
        
    # Derive the shape of X_omega
    sz = X_omega.shape
    
    # Put the known part
    prd[omega] = X_omega.reshape([sz[0],sz[1],])
    
    # Derive the size of X_omega_hat
    sz = X_omega_hat.shape
    
    # Put the unknown part
    prd[~omega] = X_omega_hat.reshape([sz[0],sz[1],])
    
    # Round the values and convert to unit8
    prd = np.uint8(prd.round())
    
    # Convert the current patch of the mask to 1
    msk_patch[~omega] = 1
                
    return prd,msk_patch
        
def face_inpainting(X):   
    K          = 500    # Number of neighbours to restore a patch
    patch_size = 15      # Size of the patch
    
    
    # Get the mask to be used for inpainting (1 indicates green pixels)
    mask = (X[:,:,0] == 0) & (X[:,:,1] == 255) & (X[:,:,2] == 0)
    mask = ~mask
                
    # Derive the fill region (0 indicates green pixels)
    fillRegion = (mask == 0)
    
    # Initialize the fill image to be equal to X
    fillImg = X
    img     = fillImg
    
    # Derive the source region mask
    sourceRegion = (mask == 1)
        
    
    # Determine the dimmensions of the image to be filled
    sz = fillImg.shape
                   
    # Initialize the confidence terms
    C = np.double(sourceRegion)
    
    # Load the face dictionary from mat file
    dict = sio.loadmat('model/dict.mat')
    
    # Extract the list of images contained in dictionary
    dict_lst = dict['dict']
    
    # Determine the number of images to be included
    Nimgs = len(dict_lst)
    
    print('Load the dictionary...')
    
    # Initialize the dictionary
    face_dict = np.zeros([sz[0],sz[1],sz[2],Nimgs])
    for n in range(Nimgs):        
        # Put the loaded image into the dictionary
        face_dict[:,:,:,n] = dict_lst[n,0]
    
    print('')
    print('Inpaint the frontal face ...')
    print('-----------------------------------------------------')

    while 1:
        #######################################################################
        # Find the patch to be processed
        #######################################################################
        
        # Find the contour and normalized gradients of fill region
        fillRegionD = np.double(fillRegion)
        
        
        # Convolve the image for image detection (rate of change)
        imgLap = ndimage.convolve(fillRegionD,np.array([[1,1,1],[1,-8,1],[1,1,1]]))
        
        # Get the coordinates from source image belonging to segment n
        dR = np.asarray(np.nonzero(imgLap > 0)).transpose()
        
        # Initialize the priorities          
        priorities = np.zeros([len(dR),1])
        
        # Compute confidence along the fill front
        for k, dr in enumerate(dR):
            # Derive the index of the patch to be considered
            rows,cols = get_patch(sz,dr,patch_size)
                        
            # Extract the confidence at the current patch             
            C_patch = C[rows,cols]
            
            # Derive the size of the sub-img
            patch_sz = C_patch.shape
            
            # Extract the fill region at the current patch
            fill_patch = fillRegion[rows,cols]
            
            # Compute the confidence
            conf = sum(C_patch[fill_patch == 0])/(patch_sz[0]*patch_sz[1])
                        
            # Compute the priority for each patch
            priorities[k,0] = conf
            
        
        # Find the patch with largest priority
        idx = np.argmax(priorities,axis=0)
        
        # Extract the patch position to be considered
        p = dR[idx[0],:]
        #######################################################################
       
        # Get the rows and columns of the patch to be processed
        rows, cols = get_patch(sz,p,patch_size)
        
        # Extract the patch from the image to be restored
        img_patch = img[rows, cols]
        msk_patch = mask[rows,cols]
        
        # Determine the patch size
        patch_sz = img_patch.shape
        
        # Initialize the patch dictionary
        patch_dict = np.zeros([patch_sz[0],patch_sz[1],patch_sz[2],Nimgs])
        for n in range(Nimgs):        
            # Put the loaded image into the dictionary
            I = face_dict[:,:,:,n]
                      
            # Extract the corresponding patch from dict
            patch_dict[:,:,:,n] = I[rows,cols]
        
        # Derive the restored patch
        [img_patch_hat,msk_patch] = face_hallucination(img_patch, msk_patch,patch_dict,K)
        
        # Update the mask
        mask[rows, cols] = msk_patch
            
        # Update the fillRegion
        fillRegion[rows,cols] = (mask[rows,cols] == 0)
        
        # Update the current image with the current patch values
        img[rows,cols] = img_patch_hat
           
        # Update the confidence term   
        C[rows,cols] = priorities[idx[0],:]
           
        # Determine the number of pixels to be inpainted
        all_zeros = not np.any(mask==0)

        if all_zeros :
            break
    return img

def inverse_warp_restored_region(X,Y_reg,param):
    
    # Get information contained in param
    TRI_d = param.TRI_d     # Triangles of Iface
    img_seg = param.img_seg
    Tforward_list = param.Tforward_list
    
    # Determine the number of triangles
    Ntri = len(TRI_d.simplices)
    
    # Get the mask to be used for inpainting (1 indicates green pixels)
    mask = (X[:,:,0] == 0) & (X[:,:,1] == 255) & (X[:,:,2] == 0)
    mask = ~mask
    
    # Initialize the output image
    Y = X
    
    for n in range(Ntri):
        # Get the coordinates from source image belonging to segment n
        coord1 = np.nonzero(img_seg == n)
        
        # Get the nth affine transformation matrix
        Tforward = Tforward_list[n]
        
        # Convert the tuple into an array
        coord1 = np.asarray(coord1).transpose()
                      
        # Add a column of 1s to be able to compute the matrix multiplication
        coord1 = np.concatenate((coord1,np.ones((len(coord1),1))),axis=1).astype(int)
          
        # Determine the target coordinates using affine transformation
        coord2 = (coord1.dot(Tforward)).round() .astype(int) 
        
        # Derive the size of the frontal face
        img_size = Y_reg.shape
        
        for i in range(len(coord1)):
            if (coord2[i,1] < img_size[1]) & (coord2[i,1] >= 0) & (coord2[i,0] < img_size[0]) & (coord2[i,0] >= 0):
                # If pixel has to be restored
                if mask[coord1[i,1],coord1[i,0]] == 0:
                       # Put the pixel from the frontal to the source image
                       Y[coord1[i,1],coord1[i,0],:] = Y_reg[coord2[i,1],coord2[i,0],:]
                       # Mark with the mask that the pixel was restored
                       mask[coord1[i,1].astype(int),coord1[i,0].astype(int)] = 1
    return Y   
    
###############################################################################
# INITIALIZATION
###############################################################################
cls() # clear the console
print('---------------------------------------------------------------')
# Define the face id to be considered
face_id = 13;

# Define the filename of the image to be restored
img_filename = 'imgs/%d_green.bmp' %(face_id)

# Print a message to show which image is being loaded
print('The image %s is being loaded...' %(img_filename))

# Load the face to be inpainted
Iface = scipy.misc.imread(img_filename)

# Derive the filename of the corresponding landmark file
lm_filename = 'lms/%d.MAT' %(face_id)

# Print a message to show which image is being loaded
print('Loading the landmarks %s...' %(lm_filename))

# Load the landmark points
lm = sio.loadmat(lm_filename)

# Extract the landmark points for this fac
landmarks = lm['landmarks'] - 1 # Models were derived using MATLAB

# Deruve the reference model
ref_model_filename = 'model/refModel21.csv'

# Print a message to show which image is being loaded
print('Loading the reference face model...')

# Load the reference model landmark points
with open(ref_model_filename,'r') as p:
    ref_landmarks = [list(map(int,rec)) for rec in csv.reader(p, delimiter=',')] 


# Get the number of landmakrs in the reference model
Nlm = len(landmarks)

# Print a message showing the number of landmarks
print('The face model uses %d landmark points' %(Nlm))

#------------------------------------------------------------------------------
# Visualize the input image and landmark points

x_lms = []
y_lms = []
# Show the input image
implot = plt.imshow(np.uint8(Iface))


# Extract the landmarks as separate lists
for i in range(Nlm):
    x_lms.append(landmarks[i][0])
    y_lms.append(landmarks[i][1])                         

# Show the landmarks on the input face
plt.scatter(x=x_lms, y=y_lms,c='r',s=40)
plt.show()

print('---------------------------------------------------------------')

#------------------------------------------------------------------------------

###############################################################################
# FORWARD WARPING
###############################################################################
#t = time.time()
# Forward transform the face region - Face frontalization
I_front, param = forward_face_registration(Iface,landmarks,ref_landmarks,[67,67])
#elapsed = time.time() - t
#print('Frontalization took %fs' %(elapsed))

# Visualization
#plt.imshow(np.uint8(I_front))
#plt.title('Frontalized face')
#plt.show()

###############################################################################

###############################################################################
# FACE INPAINTING
###############################################################################
t = time.time()
# Inpaint the frontal face image
#profile.run('face_inpainting(I_front)', sort='tottime')
I_front_inp = face_inpainting(I_front)
elapsed = time.time() - t
print('Inpainting took %fs' %(elapsed))

# Visualization
#plt.imshow(np.uint8(I_front_inp))
#plt.title('Restored frontal face')
#plt.show()
###############################################################################

###############################################################################
# INVERSE WARPING
###############################################################################
t = time.time()


# Restore the unknown part on the original face
I_inp = inverse_warp_restored_region(Iface,I_front_inp,param)

elapsed = time.time() - t
print('Final Restore took %fs' %(elapsed))

# Visualization
plt.imshow(np.uint8(I_inp))
plt.title('Restored image')
plt.show()