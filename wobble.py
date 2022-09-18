##!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import re
from symfit import parameters, variables, sin, cos, Fit
from symfit.core.minimizers import DifferentialEvolution

################################################################################
# ABL Z-Mesh 'wobble' correction script for Ender 3D printers
# Bernhard Stampfer, 2022, GPLv3
#
# Tries to isolate and correct periodic components in the mesh which should be
# related to x/y axis wheel position and thus affected by the probe offset.
#
# Input: text file containing 'M420 V' response:
# Output: gcode files containing M421 (Set Mesh Value) commands to be 'printed'

INPUTFILE  = "M420.txt"
OUTPUTFILE = "mesh_corrected.gcode"
BACKUPFILE = "mesh_original.gcode"
FIGUREFILE = "mesh_components.png"

WHEEL_RAD = 12          # POM V-groove wheel diameter in mm
N = (9, 9)              # 9x9 mesh points
OFFSET = (-31.8, -40.5) # cr-touch offset from nozzle (Ender3 S1 values)
CENTER = (110, 110)     # bed center in mm (Ender3 S1 values)

CORRECT_X_WOBBLE = True
CORRECT_Y_WOBBLE = True
CORRECT_YX_WOBBLE = True

PLOT_LIMITS = 0.2        # min/max z limit for plots in mm
USE_GLOBAL_OPTIMIZER = True # slower but sometimes more reliable

################################################################################

# load mesh from M420 V command response
def loadmesh(filename):
    mesh = np.zeros(N)
    limits = np.zeros((2, 2))
    with open(filename, 'r') as f:
        ll = f.readlines()
    for i, l in enumerate(ll):
        if "Bed Topography Report" in l:
            start = i
            break
    (_,_),(x1,y1) = re.findall("(\d+),\s*(\d+)\)" , ll[start+2])

    for i in range(N[1]):
        m = re.findall('(-?\d+\.?\d*)', ll[start+4+2*i])
        print(m)
        if m:
            mesh[int(m[0]),:] = np.array(m[1:])
    #print(mesh)
    (x0,y0),(_,_) = re.findall("(\d+),\s*(\d+)\)" , ll[start+2+2+2*N[1]])
    print(x0,x1,y0,y1)
    limits = np.array([(float(x0),float(x1)),(float(y0),float(y1))])
    return mesh, limits

# save mesh to gcode file with M421 commands
def savemesh(filename, mesh):
    with open(filename, 'w') as fo:
        for i in range(N[0]):
            for j in range(N[1]):
                fo.write(f"M421 I{i} J{j} Z{mesh[j,i].round(3)}\n")


if __name__ == "__main__":
    
    z_mesh, limits = loadmesh(INPUTFILE)
    # calculate coordinates from mesh limits, mesh is given in bed coordinates
    x_vec_bed = np.linspace(limits[0][0], limits[0][1], N[0]) - CENTER[0]
    y_vec_bed = np.linspace(limits[1][0], limits[1][1], N[1]) - CENTER[1]
    # calculate the corresponding axis coordinates
    x_vec_axis = x_vec_bed - OFFSET[0]
    y_vec_axis = y_vec_bed - OFFSET[1]
    # make coordinate meshes for fitting
    x_mesh_bed, y_mesh_bed = np.meshgrid(x_vec_bed, y_vec_bed)
    x_mesh_axis, y_mesh_axis = np.meshgrid(x_vec_axis, y_vec_axis)
    print("mesh loaded")

    # variables: axis and bed x,y values, z mesh values
    xa, ya, xb, yb, z = variables("xa, ya, xb, yb, z")
    
    # fit parameters
    # polynomial factors
    c, kx, ky, kxx, kyy, kyx = parameters('c, kx, ky, kxx, kyy, kyx')
    c.value = kx.value = ky.value = kxx.value = kyy.value = kyx.value = 0.0
    c.min   = kx.min   = ky.min   = kxx.min   = kyy.min   = kyx.min   = -0.01
    c.max   = kx.max   = ky.max   = kxx.max   = kyy.max   = kyx.max   = 0.01
    # wheel radius (effective radius is lower than outer radius)
    rx, ry = parameters('rx, ry')
    rx.value = ry.value = WHEEL_RAD-2
    rx.min   = ry.min   = WHEEL_RAD-4
    rx.max   = ry.max   = WHEEL_RAD
    # x axis wobble amplitudes
    ax, bx   = parameters('ax, bx')
    ax.value = bx.value = 0
    ax.min   = bx.min   = -0.2
    ax.max   = bx.max   = +0.2
    # y axis wobble (up/down) amplitudes
    ay, by   = parameters('ay, by')
    ay.value = by.value = 0
    ay.min   = by.min   = -0.2
    ay.max   = by.max   = +0.2
    # y axis wobble (bed left-right tilt) amplitudes
    ayx, byx  = parameters('ayx, byx')
    ayx.value = byx.value = 0
    ayx.min   = byx.min   = -0.2
    ayx.max   = byx.max   = +0.2
    
    # model
    linear    = c + kx*xb + ky*yb
    quadratic = kxx*xb*xb+ kyy*yb*yb
    roll      = kyx*yb*xb
    xwobble   = ax*cos(xa/rx) + bx*sin(xa/rx)
    ywobble   = ay*cos(ya/ry) + by*sin(ya/ry)
    yxwobble  = ayx*cos(ya/ry)*xa + byx*sin(ya/ry)*xa
    model = { z: linear+quadratic+roll+xwobble+ywobble+yxwobble }    
    
    # fit using symfit
    if USE_GLOBAL_OPTIMIZER:
        print('fitting, this may take some time')
        fit = Fit(model, xa=x_mesh_axis, ya=y_mesh_axis, xb=x_mesh_bed, yb=y_mesh_bed, z=z_mesh, minimizer=DifferentialEvolution)
    else:
        print('fitting')
        fit = Fit(model, xa=x_mesh_axis, ya=y_mesh_axis, xb=x_mesh_bed, yb=y_mesh_bed, z=z_mesh)
    fit_result = fit.execute()
    z_mesh_fit = fit.model(xa=x_mesh_axis, ya=y_mesh_axis, xb=x_mesh_bed, yb=y_mesh_bed, **fit_result.params).z
    print(fit_result)

    # calculate the individual components using the fit parameters
    # polynomial components, we assume they come from the bed
    z_mesh_l  = linear(xb=x_mesh_bed, yb=y_mesh_bed, **fit_result.params)
    z_mesh_q  = quadratic(xb=x_mesh_bed, yb=y_mesh_bed, **fit_result.params)
    z_mesh_r  = roll(xb=x_mesh_bed, yb=y_mesh_bed, **fit_result.params)
    
    # wobble components as measured, we assume they come from the axis and are thus incorrect
    z_mesh_x  = xwobble(xa=x_mesh_axis, ya=y_mesh_axis, **fit_result.params)
    z_mesh_y  = ywobble(xa=x_mesh_axis, ya=y_mesh_axis, **fit_result.params)
    z_mesh_yx = yxwobble(xa=x_mesh_axis, ya=y_mesh_axis, **fit_result.params)
    
    # calculate the corrected wobble components, calculated for axis position=bed position as is the case during printing
    z_mesh_x_c = xwobble(xa=x_mesh_bed, ya=y_mesh_bed, **fit_result.params)
    z_mesh_y_c = ywobble(xa=x_mesh_bed, ya=y_mesh_bed, **fit_result.params)
    z_mesh_yx_c = yxwobble(xa=x_mesh_bed, ya=y_mesh_bed, **fit_result.params)
    
    # component that could not be fitted
    z_mesh_res = z_mesh - z_mesh_fit
    
    # copy and modify mesh with corrected components
    z_mesh_out = z_mesh.copy()
    if CORRECT_X_WOBBLE:
        z_mesh_out += -z_mesh_x + z_mesh_x_c
    if CORRECT_Y_WOBBLE:
        z_mesh_out += -z_mesh_y + z_mesh_y_c
    if CORRECT_YX_WOBBLE:
        z_mesh_out += -z_mesh_yx + z_mesh_yx_c

    # write out
    savemesh(BACKUPFILE, z_mesh)
    savemesh(OUTPUTFILE, z_mesh_out)
 
    # plot
    fig, axs = plt.subplots(3,4)
    cmap='RdBu_r'
    
    x = x_mesh_bed + CENTER[0]
    y = y_mesh_bed + CENTER[1]
    
    axs[0,0].pcolormesh(x, y, z_mesh, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[0,0].set_title("wobble.txt")

    axs[0,1].pcolormesh(x, y, z_mesh_fit, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[0,1].set_title("fit")
    
    axs[0,2].pcolormesh(x, y, z_mesh_res, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[0,2].set_title("remaining")

    axs[0,3].pcolormesh(x, y, z_mesh_out, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[0,3].set_title("wobble.gcode")

    axs[1,0].pcolormesh(x, y, z_mesh_l + z_mesh_r, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[1,0].set_title("linear (tram?)")

    axs[1,1].pcolormesh(x, y, z_mesh_q, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[1,1].set_title("quadratic")
    
    axs[1,2].pcolormesh(x, y, z_mesh_l + z_mesh_r + z_mesh_q, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[1,2].set_title("sum polynomial")
    
    axs[1,3].pcolormesh(x, y, z_mesh_l + z_mesh_r + z_mesh_q + z_mesh_res, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[1,3].set_title("sum static (bed?)")

    axs[2,0].pcolormesh(x, y, z_mesh_x, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[2,0].set_title("x periodic")

    axs[2,1].pcolormesh(x, y, z_mesh_y, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[2,1].set_title("y periodic")

    axs[2,2].pcolormesh(x, y, z_mesh_yx, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[2,2].set_title("xy periodic")
    
    axs[2,3].pcolormesh(x, y, z_mesh_x + z_mesh_y + z_mesh_yx, vmin=-PLOT_LIMITS, vmax=PLOT_LIMITS, cmap=cmap)
    axs[2,3].set_title("sum periodic (axes)")
   
    plt.tight_layout()
    plt.savefig(FIGUREFILE)
    plt.show()
