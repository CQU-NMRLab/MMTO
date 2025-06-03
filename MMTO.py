########################################################################################################
### MMTO-Python         															                 ### 
### This file is part of MMTO-Python.                                                                ###
### This is the python version of the code written Zhengxu and Yue shen.                             ###
### version 01-06-2025                                                                               ###
########################################################################################################

import gmsh                        # A library for grid generation
import os
import sys
import time
import numpy as np                 # Basic numpy array operations
import matplotlib.pyplot as plt    # Import the library used for drawing
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve     # Solution of sparse matrix equations
from MMA import mmasub,subsolv

# =============================================================================
M = conn2.shape[0]   # Number of units
N = x.shape[0]       # Number of nodes
# =============================================================================
# Identify different physical domains
domain = np.where((conn2[:,6].astype(int) == 1))[0].astype(int)
Rowcoil = np.where((conn2[:,6].astype(int) == 2) | (conn2[:, 6].astype(int)==3))[0].astype(int)   # Antenna unit serial number
Rowroi  = np.where(conn2[:,6].astype(int) == 4)[0].astype(int)   # Target area unit serial number
Rowair  = np.where(conn2[:,6].astype(int) == 5)[0].astype(int)   # Serial number of the peripheral air domain unit
area = np.zeros(M)   # Design unit area
u0 = 4*np.pi*1e-7    # Vacuum permeability
# =============================================================================
# Calculate the objective function under each structure and update it in real time in topology optimization   
mm = 8             # The penalty coefficient based on the logistic function density penalty model
rmin = 2 * 1e-3                       # 2mm filtering radius
rou1 = 0.5*np.ones(M)              # The initial densities of all units 
rou2 = 0.5*np.ones(M)              # The initial densities of all units  
centroids_x = np.mean(x[conn2[:, 0:3].astype(int) - 1], axis=1)
centroids_y = np.mean(y[conn2[:, 0:3].astype(int) - 1], axis=1)
centroids_distance = np.sqrt(centroids_x**2 + centroids_y**2)  # 单元中心点到原点的距离
for e in domain:
    if (centroids_distance[e] > 0.016) and (centroids_y[e] > 0):
       rou2[e] = 0
loop = 0       # Number of iterations
change1 = 1     # Element density variation
change2 = 1     # Element density variation
# Basic parameters of MMA
n = 2*domain.shape[0]    # The number of variables 
m = int(n//2) 
xmin = np.zeros((n, 1))  # The minimum boundary of the design variable
xmax = np.ones((n, 1))   # The maximum boundary of the design variable
x_part1 = rou1[domain][np.newaxis].T  
x_part2 = rou2[domain][np.newaxis].T  
combined = np.vstack((x_part1, x_part2))
xold1 = combined.copy()
xold2 = combined.copy()
low = np.ones((n, 1))   # Column vector with the lower asymptotes from the previous iteration (provided that iter>1)
upp = np.ones((n, 1))   # The upper and lower asymptotic lines are unique to the MMA algorithm and are used to control the movement range of design variables
a0 = 1.0                  # The constants a_0 in the term a_0*z
am = np.zeros((m, 1))     
cm = 1e4*np.ones((m, 1))
dm = np.zeros((m, 1))
move = 0.08             
obj = []       # Objective function
"""========================== Logistic + MMA +MMTO ==========================="""
while (change1>0.001) and (change2>0.001) and (loop<80):
    loop = loop +1 
    for e in domain:
         if (centroids_distance[e] > 0.016) and (centroids_y[e] > 0):
             rou2[e] = 0
    """============================= FEM post-processing============================="""
    # Evaluation of magnetic field distribution in the target area    
    obj_x, obj_y, obj_B0, obj_B1 = np.zeros(12), np.zeros(12), np.zeros((12,3)), np.zeros((12,3))
    mea = 45  #roi measures the position deflection Angle
    mea = mea/180*np.pi         
    obj_x = 1e-3*np.array([0,0,0,0,0,0,\
                      dis1*np.sin(mea),(dis1+2)*np.sin(mea),(dis1+4)*np.sin(mea),(dis1+6)*np.sin(mea),(dis1+8)*np.sin(mea),(dis1+10)*np.sin(mea)])
    obj_y = 1e-3*np.array([ dis1,dis1+2,dis1+4,dis1+6,dis1+8,dis1+10,\
               dis1*np.cos(mea),(dis1+2)*np.cos(mea),(dis1+4)*np.cos(mea),(dis1+6)*np.cos(mea),(dis1+8)*np.cos(mea),(dis1+10)*np.cos(mea)])     
    dB_dA = np.zeros((2,N,12))       
    # Calculate B0 and B1 of each point in the target area
    for i in range(0,12):              
        xp, yp = obj_x[i], obj_y[i]    
        for e in Rowroi:              
            xa, xb, xc = x[conn2[e,0].astype(int)-1], x[conn2[e,1].astype(int)-1], x[conn2[e,2].astype(int)-1]    # Obtain the coordinates of the three vertices of the partitioned triangle
            ya, yb, yc = y[conn2[e,0].astype(int)-1], y[conn2[e,1].astype(int)-1], y[conn2[e,2].astype(int)-1]
            # Calculate the local coordinates (u, v) of the point within the triangle
            u = np.linalg.det(np.array([[xp-xa,xb-xa],[yp-ya,yb-ya]])) / np.linalg.det(np.array([[xc-xa,xb-xa],[yc-ya,yb-ya]]))
            v = np.linalg.det(np.array([[xc-xa,xp-xa],[yc-ya,yp-ya]])) / np.linalg.det(np.array([[xc-xa,xb-xa],[yc-ya,yb-ya]]))
            # Check whether the calculated local coordinate points u and v are within the triangle (including the boundary).
            if u>=0 and v>=0 and (u+v)<=1:
                # Initialize the variables, calculate the linear basis functions and related parameters
                a, b, c, L = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)    # L is a linear basis function matrix
                a[0], a[1], a[2] = xb*yc-yb*xc, xc*ya-yc*xa, xa*yb-ya*xb
                b[0], b[1], b[2] = yb - yc, yc - ya, ya - yb
                c[0], c[1], c[2] = xc - xb, xa - xc, xb - xa
                areae = 0.5*(b[0]*c[1] - b[1]*c[0])
                for jj in range(0,3):                                              
                    L[jj] = (a[jj]+b[jj]*xp+c[jj]*yp)/(2*areae)                  
                A0 = A0z[conn2[e,0:6].astype(int)-1]
                A1 = A1z[conn2[e,0:6].astype(int)-1]
                #Bx and By     #Tesla   
                obj_B0[i,0] = np.dot(A0, np.array([c[0]*(4*L[0]-1), c[1]*(4*L[1]-1), c[2]*(4*L[2]-1), \
                                               4*(L[0]*c[1]+L[1]*c[0]), 4*(L[1]*c[2]+L[2]*c[1]), 4*(L[2]*c[0]+L[0]*c[2])])/(2*areae))
                obj_B0[i,1] = -np.dot(A0, np.array([b[0]*(4*L[0]-1), b[1]*(4*L[1]-1), b[2]*(4*L[2]-1), \
                                               4*(L[0]*b[1]+L[1]*b[0]), 4*(L[1]*b[2]+L[2]*b[1]), 4*(L[2]*b[0]+L[0]*b[2])])/(2*areae))                                            
                obj_B0[i,2] = np.sqrt(obj_B0[i,0]**2 + obj_B0[i,1]**2)
                
                obj_B1[i,0] = np.dot(A1, np.array([c[0]*(4*L[0]-1), c[1]*(4*L[1]-1), c[2]*(4*L[2]-1), \
                                               4*(L[0]*c[1]+L[1]*c[0]), 4*(L[1]*c[2]+L[2]*c[1]), 4*(L[2]*c[0]+L[0]*c[2])])/(2*areae))
                obj_B1[i,1] = -np.dot(A1, np.array([b[0]*(4*L[0]-1), b[1]*(4*L[1]-1), b[2]*(4*L[2]-1), \
                                               4*(L[0]*b[1]+L[1]*b[0]), 4*(L[1]*b[2]+L[2]*b[1]), 4*(L[2]*b[0]+L[0]*b[2])])/(2*areae))                                            
                obj_B1[i,2] = np.sqrt(obj_B1[i,0]**2 + obj_B1[i,1]**2)               
                # dBx/dA     
                dB_dA[0,conn2[e,0:6].astype(int)-1,i] = np.array([c[0]*(4*L[0]-1), c[1]*(4*L[1]-1), c[2]*(4*L[2]-1), \
                                                           4*(L[0]*c[1]+L[1]*c[0]), 4*(L[1]*c[2]+L[2]*c[1]), 4*(L[2]*c[0]+L[0]*c[2])])/(2*areae)
                # dBy/dA
                dB_dA[1,conn2[e,0:6].astype(int)-1,i] = -np.array([b[0]*(4*L[0]-1), b[1]*(4*L[1]-1), b[2]*(4*L[2]-1), \
                                                           4*(L[0]*b[1]+L[1]*b[0]), 4*(L[1]*b[2]+L[2]*b[1]), 4*(L[2]*b[0]+L[0]*b[2])])/(2*areae)        
                break
    kk1 = (obj_B0[:,0]*obj_B1[:,0] + obj_B0[:,1]*obj_B1[:,1]) / obj_B0[:,2]    # The component of B1 along B0
    B1m = 0.5*np.sqrt(obj_B1[:,2]**2 - kk1**2)     
    f0val = 1 / np.mean(obj_B0[:,2]**2 * 0.5*obj_B1[:,2])      # Objective function value
    obj.append(np.mean(obj_B0[:,2]**2 * B1m) * 1e12)           
    txt = 'Current topology optimization algebra: {};  Objective function: {}'
    print(txt.format(loop,obj[-1]))          
    # =============================================================================
    #Sensitivity matrix
    deno = np.sum(obj_B0[:,2]**2 * obj_B1[:,2])  
    dfai_dA0 =  -24/(deno**2) *  (obj_B1[0,2]*(2*obj_B0[0,0]*dB_dA[0,:,0]+2*obj_B0[0,1]*dB_dA[1,:,0])+\
                                  obj_B1[1,2]*(2*obj_B0[1,0]*dB_dA[0,:,1]+2*obj_B0[1,1]*dB_dA[1,:,1])+\
                                  obj_B1[2,2]*(2*obj_B0[2,0]*dB_dA[0,:,2]+2*obj_B0[2,1]*dB_dA[1,:,2])+\
                                  obj_B1[3,2]*(2*obj_B0[3,0]*dB_dA[0,:,3]+2*obj_B0[3,1]*dB_dA[1,:,3])+\
                                  obj_B1[4,2]*(2*obj_B0[4,0]*dB_dA[0,:,4]+2*obj_B0[4,1]*dB_dA[1,:,4])+\
                                  obj_B1[5,2]*(2*obj_B0[5,0]*dB_dA[0,:,5]+2*obj_B0[5,1]*dB_dA[1,:,5])+\
                                  obj_B1[6,2]*(2*obj_B0[6,0]*dB_dA[0,:,6]+2*obj_B0[6,1]*dB_dA[1,:,6])+\
                                  obj_B1[7,2]*(2*obj_B0[7,0]*dB_dA[0,:,7]+2*obj_B0[7,1]*dB_dA[1,:,7])+\
                                  obj_B1[8,2]*(2*obj_B0[8,0]*dB_dA[0,:,8]+2*obj_B0[8,1]*dB_dA[1,:,8])+\
                                  obj_B1[9,2]*(2*obj_B0[9,0]*dB_dA[0,:,9]+2*obj_B0[9,1]*dB_dA[1,:,9])+\
                                  obj_B1[10,2]*(2*obj_B0[10,0]*dB_dA[0,:,10]+2*obj_B0[10,1]*dB_dA[1,:,10])+\
                                  obj_B1[11,2]*(2*obj_B0[11,0]*dB_dA[0,:,11]+2*obj_B0[11,1]*dB_dA[1,:,11]))   
                           
    dfai_dA1 =  -24/(deno**2) *  (obj_B0[0,2]**2 * (obj_B1[0,0]*dB_dA[0,:,0]+obj_B1[0,1]*dB_dA[1,:,0])/obj_B1[0,2]+\
                                  obj_B0[1,2]**2 * (obj_B1[1,0]*dB_dA[0,:,1]+obj_B1[1,1]*dB_dA[1,:,1])/obj_B1[1,2]+\
                                  obj_B0[2,2]**2 * (obj_B1[2,0]*dB_dA[0,:,2]+obj_B1[2,1]*dB_dA[1,:,2])/obj_B1[2,2]+\
                                  obj_B0[3,2]**2 * (obj_B1[3,0]*dB_dA[0,:,3]+obj_B1[3,1]*dB_dA[1,:,3])/obj_B1[3,2]+\
                                  obj_B0[4,2]**2 * (obj_B1[4,0]*dB_dA[0,:,4]+obj_B1[4,1]*dB_dA[1,:,4])/obj_B1[4,2]+\
                                  obj_B0[5,2]**2 * (obj_B1[5,0]*dB_dA[0,:,5]+obj_B1[5,1]*dB_dA[1,:,5])/obj_B1[5,2]+\
                                  obj_B0[6,2]**2 * (obj_B1[6,0]*dB_dA[0,:,6]+obj_B1[6,1]*dB_dA[1,:,6])/obj_B1[6,2]+\
                                  obj_B0[7,2]**2 * (obj_B1[7,0]*dB_dA[0,:,7]+obj_B1[7,1]*dB_dA[1,:,7])/obj_B1[7,2]+\
                                  obj_B0[8,2]**2 * (obj_B1[8,0]*dB_dA[0,:,8]+obj_B1[8,1]*dB_dA[1,:,8])/obj_B1[8,2]+\
                                  obj_B0[9,2]**2 * (obj_B1[9,0]*dB_dA[0,:,9]+obj_B1[9,1]*dB_dA[1,:,9])/obj_B1[9,2]+\
                                  obj_B0[10,2]**2 * (obj_B1[10,0]*dB_dA[0,:,10]+obj_B1[10,1]*dB_dA[1,:,10])/obj_B1[10,2]+\
                                  obj_B0[11,2]**2 * (obj_B1[11,0]*dB_dA[0,:,11]+obj_B1[11,1]*dB_dA[1,:,11])/obj_B1[11,2])   

    lmd2 = spsolve(K2, dfai_dA0)  # Adjoint variable K2*A0z = b2   #B0
    lmd3 = spsolve(K3, dfai_dA1)  # Adjoint variable K3*A1z = b3   #B1
    # The sensitivity of the objective function to each cell density rou
    dfai_drou1 = np.zeros(M)     # The sensitivity of the objective function to the design variables 
    dfai_drou2 = np.zeros(M)     # The sensitivity of the objective function to the design variables 
    fdfai_drou1 = np.zeros(M)    # Sensitivity after filtering
    fdfai_drou2 = np.zeros(M)    # Sensitivity after filtering
    
    for e in domain:
        Ne = 6
        dK_drou1 = lil_matrix((N,N))     
        dK_drou2 = lil_matrix((N,N))     
        dbe_drou1 = np.zeros(N)
        dbe_drou2 = np.zeros(N)
        dv_drou1 = mm*np.exp(mm*(rou1[e]-0.5))/(1+np.exp(mm*(rou1[e]-0.5)))**2 * (1/(16*u0)-1/u0)  
        dBr_drou1 =np.array([0, 0])
        dv_drou2 = 0
        dBr_drou2 = np.array([1.14*mm*np.exp(mm*(rou2[e]-0.5))/(1+np.exp(mm*(rou2[e]-0.5)))**2, 0])
        tri2 = np.zeros([2,6])  
        tri2[0,:] = x[conn2[e,0:6].astype(int)-1]    
        tri2[1,:] = y[conn2[e,0:6].astype(int)-1]  
        [Ke1,_] = element_matrix_tri2(tri2,dv_drou1)
        [Ke2,_] = element_matrix_tri2(tri2,dv_drou2)
        
        for ie in range(0,Ne):
            ig = conn2[e,ie]             
            for je in range(0,Ne):
                jg = conn2[e,je]         
                dK_drou1[int(ig-1),int(jg-1)] = dK_drou1[int(ig-1),int(jg-1)] + Ke1[ie,je]   
                dK_drou2[int(ig-1),int(jg-1)] = dK_drou2[int(ig-1),int(jg-1)] + Ke2[ie,je]   
         
        be1 = element_matrix_tri2_be(tri2,dv_drou1,Br) + element_matrix_tri2_be(tri2,v,dBr_drou1)  
        be2 = element_matrix_tri2_be(tri2,dv_drou2,Br) + element_matrix_tri2_be(tri2,v,dBr_drou2)
        for ie in range(0,Ne):
           ig = conn2[e,ie] 
           dbe_drou1[int(ig-1)] = dbe_drou1[int(ig-1)] + be1[ie]
           dbe_drou2[int(ig-1)] = dbe_drou2[int(ig-1)] + be2[ie]
        
        dK_drou1 = csr_matrix(dK_drou1)
        dK_drou2 = csr_matrix(dK_drou2)
        dfai_drou1[e] = sparse.csr_matrix.dot(lmd2.T, (dbe_drou1 - sparse.csr_matrix.dot(dK_drou1, A0z)))
        dfai_drou1[e] = dfai_drou1[e] - sparse.csr_matrix.dot(lmd3.T, sparse.csr_matrix.dot(dK_drou1, A1z))
        dfai_drou2[e] = sparse.csr_matrix.dot(lmd2.T, dbe_drou2)
        dfai_drou2[e] = dfai_drou2[e] - sparse.csr_matrix.dot(lmd3, sparse.csr_matrix.dot(dK_drou2, A1z)) 
    # =============================================================================
    #Sensitivity filtering
    for e in domain:
        i = np.hstack((np.where((distance[e,:]>0) & (distance[e,:]<rmin))[0].astype(int), int(e))) 
        H = np.maximum(0, rmin - distance[e,i])   #np.maximum(a,b):逐位取较大值  #向量H的维度为i
        fdfai_drou1[e] = np.sum(H*rou1[i]*dfai_drou1[i]) / (np.max([0.001,rou1[e]]) * np.sum(H))  
# =============================================================================
    #MMA  
    mu0 = 1e-1 # Scale factor for objective function          
    mu1 = 1.0 # Scale factor for volume constraint function     
    xval1 = rou1[domain][np.newaxis].T   
    xval2 = rou2[domain][np.newaxis].T   
    xval = np.vstack((xval1, xval2))
    f0val = mu0 * f0val     #Objective function value scaling
    fdfai_drou1 = mu0 * fdfai_drou1
    fdfai_drou2 = mu0 * fdfai_drou2
    df0dx1 = fdfai_drou1[domain][np.newaxis].T    #The sensitivity of the objective function to the design variables 
    df0dx2 = fdfai_drou2[domain][np.newaxis].T    
    df0dx = np.vstack((df0dx1, df0dx2))
    fval = xval1 + xval2 - 1  
    dfdx = np.zeros((m, n))  
    
    def dfdx_values(n):
        m = int(n//2)
        matrix = np.zeros((m, n))  
        for i in range(n // 2):
           matrix[i, i] = 1
           matrix[i, i + n // 2] = 1
        return matrix
       
    dfdx = dfdx_values(n)
    xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
        mmasub(m,n,loop,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,am,cm,dm,move)   
    xold2 = xold1.copy()
    xold1 = xval.copy()
    xmma_flat= xmma.copy().flatten()   
    n1 = len(xval1)
    n2 = len(xval2)
    rou1[domain] = xmma_flat[:n1]  
    rou2[domain] = xmma_flat[n1:n1+n2]  
    change1 = np.linalg.norm(rou1[domain] - xval1)  
    change2 = np.linalg.norm(rou2[domain] - xval2)  
    
# =============================================================================
#Visualization of the design domain   
colors1[rou1[domain] <= 0.5] = np.nan  
plt.tripcolor(1e3 * x, 1e3 * y, conn2[domain, 0:3] - 1, facecolors=colors1, edgecolors='none', cmap=plt.cm.gray_r, alpha=1, vmin=0, vmax=1)
plt.tripcolor(-1e3 * x, 1e3 * y, conn2[domain, 0:3] - 1, facecolors=colors1, edgecolors='none', cmap=plt.cm.gray_r, alpha=1, vmin=0, vmax=1)
plt.gca().set_aspect('equal')
plt.xlim(-25, 25)
plt.ylim(-25, 26.5)
plt.xticks(np.linspace(-25, 25, 11), fontproperties='Times New Roman', size=12)
plt.yticks(np.linspace(-25, 25, 11), fontproperties='Times New Roman', size=12)
plt.xlabel('X [mm]', fontproperties='Times New Roman', fontsize=15)
plt.ylabel('Y [mm]', fontproperties='Times New Roman', fontsize=15)
plt.title('Topology Optimization', fontproperties='Times New Roman', fontsize=17, color='black', pad=7)
colors2 = np.clip(rou2[domain], 0, 1)  
colors2[rou2[domain] <= 0.5] = np.nan 
plt.tripcolor(1e3 * x, 1e3 * y, conn2[domain, 0:3] - 1, facecolors=colors2, edgecolors='none', cmap=plt.cm.Blues, alpha=1, vmin=0, vmax=1)
plt.tripcolor(-1e3 * x, 1e3 * y, conn2[domain, 0:3] - 1, facecolors=colors2, edgecolors='none', cmap=plt.cm.Blues, alpha=1, vmin=0, vmax=1)
colors3 = np.ones(Rowcoil.shape[0]) * 0.6
plt.tripcolor(1e3*x, 1e3*y, conn2[Rowcoil, 0:3] - 1, facecolors=colors3, edgecolors='none', cmap=plt.cm.YlOrBr, alpha=1, vmin=0, vmax=1)
plt.tripcolor(-1e3*x, 1e3*y, conn2[Rowcoil, 0:3] - 1, facecolors=colors3, edgecolors='none', cmap=plt.cm.YlOrBr, alpha=1, vmin=0, vmax=1)
plt.show()
