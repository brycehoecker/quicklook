import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import pandas as pd

nmodules=25
npixels=64   

# Module positions are for use with pcolormesh - cannot be used with imshow

# Module Positions (Skyview)
#modPos = {  4:(0,4),   5:(0,3),   1:(0,2),   3:(0,1), 2:(0,0),
#          103:(1,4), 125:(1,3), 126:(1,2), 106:(1,1), 9:(1,0),
#          119:(2,4), 108:(2,3), 110:(2,2), 121:(2,1), 8:(2,0),
#          115:(3,4), 123:(3,3), 124:(3,2), 112:(3,1), 7:(3,0),
#          100:(4,4), 111:(4,3), 114:(4,2), 107:(4,1), 6:(4,0)}

# Module Positions (Camera view)
modPos = {  4:(0,0),   5:(0,1),   1:(0,2),   3:(0,3), 2:(0,4),
          103:(1,0), 125:(1,1), 126:(1,2), 106:(1,3), 9:(1,4),
          119:(2,0), 108:(2,1), 110:(2,2), 121:(2,3), 8:(2,4),
          115:(3,0), 123:(3,1), 124:(3,2), 112:(3,3), 7:(3,4),
          100:(4,0), 111:(4,1), 114:(4,2), 107:(4,3), 6:(4,4)}

# Create a list of module numbers present in the currents file
def list_modules(data):
    module_numbers = set()
    for sensor in data.keys():
        if sensor == 'timestamp_start':
            continue
        if sensor == 'timestamp_end':
            continue
        module_numbers.add(sensor.split('_')[0])
    return module_numbers


# calculates row and column of pixel in a module from the pixel number
def pixel_row_col_coords(index):
    # Convert bits 1, 3 and 5 to row
    row = 4*((index & 0b100000) > 0) + 2*((index & 0b1000) > 0) + 1*((index & 0b10) > 0)
    # Convert bits 0, 2 and 4 to col
    col = 4*((index & 0b10000) > 0) + 2*((index & 0b100) > 0) + 1*((index & 0b1) > 0)
    return (row, col)


# Assigns current value to appropriate location in module
def arrange_pixels(values_by_pixel):
    values_by_position = np.zeros([8,8])
    for i, val in enumerate(values_by_pixel):
        row, col = pixel_row_col_coords(i)
        values_by_position[row,col] = val
    return values_by_position

#Creates a dictionary with key:module number and value:list of currents for a given current reading
#assumes checking for data < 0
def get_currents(data,modules,reading):
     
    # Get currents for each module
    modCurrents = {}
    for mod in modules:
        mod = int(mod)
        currents_by_pixel = np.full(64, np.nan)
        for pixel in range(npixels):
            value = data[str(mod)+'_pixel'+str(pixel)][reading]
            currents_by_pixel[pixel] = value
        currents_by_position = arrange_pixels(currents_by_pixel)
        
        # rotate modules in odd columns
        loc = modPos[mod]
        if loc[1]%2 == 0: #FIXME should actually be odd columns which rotate
            currents_by_position = np.rot90(currents_by_position, k=2)
        
        # Add currents to dictionary
        modCurrents[mod] = currents_by_position
    
    return modCurrents

#--Ruo Y. Shang's python implementation---#
def ConvertElevAzimToXYZ(a_el_az):
    """
    el, az in radian
    """
    el = a_el_az[0]
    az = a_el_az[1]
    
    x = np.cos(el)*np.sin(az)
    y = np.cos(el)*np.cos(az)
    z = np.sin(el)
    
    norm = np.sqrt(x**2+y**2+z**2)
    
    a_xyz = np.array([x/norm,y/norm,z/norm])
    return a_xyz

def ConvertXYZToElevAzim(a_xyz):
    """
    returns el, az in radian
    """
    x = a_xyz[0]
    y = a_xyz[1]
    z = a_xyz[2]
    
    norm = np.sqrt(x**2+y**2+z**2)
    
    x/=norm
    y/=norm
    z/=norm
    
    phi = np.arccos(z)
    el = np.pi/2 - p
    theta = np.arctan2(y,x)
    az = np.pi/2 - theta
    
    a_el_az = np.array([el,az])
    return a_el_az

def ConvertCameraCoordToElevAzim(a_camera_coord, a_tel_el_az):
    
    cam_x = a_camera_coord[0]
    cam_y = a_camera_coord[1]
    
    tel_el = a_tel_el_az[0]
    tel_az = a_tel_el_az[1]
    
    a_delta_xyz_rotated = np.array([-cam_x,0,-cam_y])
    a_tel_xyz_rotated = np.array([0.,1.,0.])
    
    a_xyz_rotated = a_tel_xyz_rotated + a_delta_xyz_rotated
    
    r_z = R.from_matrix([[np.cos(-tel_az),-np.sin(-tel_az),0.],
                        [np.sin(-tel_az),np.cos(-tel_az),0.],
                        [0.,0.,1.]])
    r_x = R.from_matrix([[1.,0.,0.],
                        [0.,np.cos(tel_el),-np.sin(tel_el)],
                        [0.,np.sin(tel_el),np.cos(tel_el)]])
    r = r_z * r_x
    
    a_xyz = r.apply(a_xyz_rotated)
    
    a_el_az = ConvertXYZToElevAzim(a_xyz)
    
    return a_el_az

def ConvertElevAzimToCameraCoord(a_el_az, a_tel_el_az):
    
    tel_el = a_tel_el_az[0]
    tel_az = a_tel_el_az[1]
    
    a_xyz = ConvertElevAzimToXYZ(a_el_az)
    
    r_z = R.from_matrix([[np.cos(-tel_az),-np.sin(-tel_az),0.],
                        [np.sin(-tel_az),np.cos(-tel_az),0.],
                        [0.,0.,1.]])
    r_x = R.from_matrix([[1.,0.,0.],
                        [0.,np.cos(tel_el),-np.sin(tel_el)],
                        [0.,np.sin(tel_el),np.cos(tel_el)]])
    
    r_inv = r_x.inv() * r_z.inv()
    
    a_xyz_rotated = r_inv.apply(a_xyz)
    
    a_tel_xyz_rotated = np.array([0.,1.,0.])
    
    a_delta_xyz_rotated = a_xyz_rotated - a_tel_xyz_rotated
    
    cam_x = -a_delta_xyz_rotated[0]
    cam_y = -a_delta_xyz_rotated[2]
    
    a_camera_coord = np.array([cam_x,cam_y])
    
    return a_camera_coord
