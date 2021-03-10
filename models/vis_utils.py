from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as LA

import scipy.linalg as SLA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_axis_3dax(ax, color='gray', linestyle='dashed', alpha=0.5):
    ax.plot3D([-1,1], [0,0], [0,0], color=color, linestyle=linestyle, alpha=alpha)
    ax.plot3D([0,0], [-1,1], [0,0], color=color, linestyle=linestyle, alpha=alpha)
    ax.plot3D([0,0], [0,0], [-1,1], color=color, linestyle=linestyle, alpha=alpha)

def draw_sphere_3dax(ax, radius=1.0, resolution=50, color='b', alpha=0.1):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius*np.outer(np.cos(u), np.sin(v))
    y = radius*np.outer(np.sin(u), np.sin(v))
    z = radius*np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def draw_circle_3dax(ax, line, radius=1.0, resolution=100, color='r', alpha=1.0, linestyle='-'):
    theta = np.linspace(0, 2*np.pi, resolution)
    
    v = SLA.null_space([line])
    
    x = radius*(v[0,0]*np.cos(theta) + v[0,1]*np.sin(theta))
    # for convention
    z = -radius*(v[1,0]*np.cos(theta) + v[1,1]*np.sin(theta)) 
    y = radius*(v[2,0]*np.cos(theta) + v[2,1]*np.sin(theta))
    
    ax.plot3D(x, y, z, color=color, alpha=alpha, linestyle=linestyle)

def draw_vps_3dax(ax, vps, length=1.0, colors=None, alpha=1.0, normalize=True):    
    if normalize:
        vps = vps.copy()
        vps = vps/LA.norm(vps, axis=-1, keepdims=True)
    
    ax.quiver(0, 0, 0, vps[0,0], vps[0,2], -vps[0,1], length=length, arrow_length_ratio=0.3, color='r', alpha=alpha)
    ax.quiver(0, 0, 0, vps[1,0], vps[1,2], -vps[1,1], length=length, arrow_length_ratio=0.3, color='g', alpha=alpha)
    ax.quiver(0, 0, 0, vps[2,0], vps[2,2], -vps[2,1], length=length, arrow_length_ratio=0.3, color='b', alpha=alpha)
    ax.scatter(vps[0,0], vps[0,2], -vps[0,1], color='r', marker='o', alpha=alpha)
    ax.scatter(vps[1,0], vps[1,2], -vps[1,1], color='g', marker='o', alpha=alpha)
    ax.scatter(vps[2,0], vps[2,2], -vps[2,1], color='b', marker='o', alpha=alpha)    
    
def draw_pts_3dax(ax, pts, length=1.0, c='y', alpha=1.0, normalize=True, zorder=1.0, cmap=None):        
    if normalize:
        pts = pts.copy()
        pts = pts/LA.norm(pts, axis=-1, keepdims=True)
    
    ax.scatter(pts[:,0], pts[:,2], -pts[:,1], c=c, alpha=alpha, zorder=zorder, cmap=cmap)

def draw_frame_plt(frame, size=1.0, alpha=1.0):
    xaxis = size*frame[0:3]
    yaxis = size*frame[3:6]
    zaxis = size*frame[6:9]

    arrow_params = {
        'length_includes_head':True, 'shape':'full', 'alpha':alpha, 'zorder':2.0, 'linewidth':3.0,
    }
    
    plt.arrow(x=0, y=0, dx=xaxis[0], dy=xaxis[1], c='r', **arrow_params)
    plt.arrow(x=0, y=0, dx=yaxis[0], dy=yaxis[1], c='g', **arrow_params)
    plt.arrow(x=0, y=0, dx=zaxis[0], dy=zaxis[1], c='b', **arrow_params)

def update_background_3ch(img, thresh=0.5):
    black_areas = (img[:,:,0] < thresh) & (img[:,:,1] < thresh) & (img[:,:,2] < thresh)
    img[...,:][black_areas] = (1.0, 1.0, 1.0)
    return img

def update_background_1ch(img, thresh=0.5):
    black_areas = (img[:,:,0] < thresh)
    img[...,:][black_areas] = 1.0
    return img

if __name__ == '__main__':
    pass

