#!/usr/bin/env python3

import numpy as np
import numba as nb

# Aggregate onto regular grid; compute sums and counts, and then return a mean if counts is large enough
# This function is compact and flexible, but cannot be accelerated with numba, so wrap below
#@nb.jit(nopython=True)
def gridded_sum(d, coords, bounds): #x, y, z, xb, yb, zb):
        
    # Compute grid centers, since we are passing grid bounds
    centers = [(b[:-1] + b[1:]) / 2.0 for b in bounds]
    
    # Initialize sums and counts
    new_shape = [len(b) - 1 for b in bounds]
    dsum = np.zeros(new_shape)
    dcnt = np.zeros(new_shape)
    
    # Loop over points, aggregate appropriate sums
    for idx in range(len(d)):
        
        if np.isfinite(d[idx]):
            # Find nearest point on grid
            index = tuple([np.argmin((x[idx] - xc)**2.0) for x, xc in zip(coords, centers)])# ix,iy,iz])
            dsum[index] = dsum[index] + d[idx]
            dcnt[index] = dcnt[index] + 1
    
    return dsum, dcnt

# Aggregate onto a regular grid by finding nearest target grid center for each profile location
# (i.e., "nearest neighbor" or "nn"). This function wraps the 2d and 3d versions so that we
# can explicitly specify the coordinates as individual variables, rather than tuples, so that
# we can accelerate with numba.
def gridded_sum_nn(d, coords, centers):
    if len(coords) == 2:
        return gridded_sum2d(d, *coords, *centers)
    elif len(coords) == 3:
        return gridded_sum3d(d, *coords, *centers)

# 2d implementation of gridding routine
@nb.jit(nopython=True)
def gridded_sum2d(d, x, y, xc, yc):
      
    # Initialize sums and counts
    new_shape = (len(xc), len(yc))
    dsum = np.zeros(new_shape)
    dcnt = np.zeros(new_shape)
    
    # Loop over points, aggregate appropriate sums
    for idx in range(len(d)):
        
        if np.isfinite(d[idx]):
            # Find nearest point on grid
            ix, iy = [np.argmin((coord[idx] - center)**2.0) for coord, center in zip((x, y), (xc, yc))]
            dsum[ix,iy] = dsum[ix,iy] + d[idx]
            dcnt[ix,iy] = dcnt[ix,iy] + 1
    
    return dsum, dcnt

# 3d implementation of gridding routine
@nb.jit(nopython=True)
def gridded_sum3d(d, x, y, z, xc, yc, zc):
    
    # Initialize sums and counts
    new_shape = (len(xc), len(yc), len(zc))
    dsum = np.zeros(new_shape)
    dcnt = np.zeros(new_shape)
    
    # Loop over points, aggregate appropriate sums
    for idx in range(len(d)):
        
        if np.isfinite(d[idx]) and d[idx] >= 0:
            # Find nearest point on grid
            ix, iy, iz = [np.argmin((coord[idx] - center)**2.0) for coord, center in zip((x, y, z), (xc, yc, zc))]
            dsum[ix,iy,iz] = dsum[ix,iy,iz] + d[idx]
            dcnt[ix,iy,iz] = dcnt[ix,iy,iz] + 1
    
    return dsum, dcnt
