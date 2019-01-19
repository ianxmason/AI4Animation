import sys
import os
import numpy as np
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters

sys.path.append('./motion')

import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from Learning import RBF

"""
This file is for converting the txt output files from the Unity preprocessing into a npz file for training the neural networks.
This file creates the style database with 8 styles and a large amount of data for each style.
The txt files are output from an input bvh converted to 60fps

The file format is:
Each line is a frame (the first and last frames of the input bvh have been removed as a previous/next frame is required to calculate velocities)
The meaning of each value in a frame is given in InputLabels.txt and OutputLabels.txt
The root direction is lightly smoothed
The values corresponding to unused end points of the bvh (those with the word 'site' in them in InputLabels.txt) are ignored.

We extract the following information
Inputs:
Trajectory Positions x and z - past and future
Trajectory Dirextions x and z - past and future
Joint Positions for input frame
Joint Velocities for input frame

Outputs:
Root x and z velocity
Root y rotational velocity
Change in Phase
Trajectory Positions x and z - only future
Trajectory Directions x and z - only future
Joint Positions for next frame
Joint Velocities for next frame
Joint rotations for next frame
      
Labels:
Label representing the style
Label representing the gait (walk/run)
Label representing if the clip is mirrored or not
"""

""" Options """

rng = np.random.RandomState(1234)
window = 60
njoints = 31

""" Data """

data_list = [
    './Export/angry_motion_01_000.bvh',
    './Export/angry_motion_02_000.bvh',
    './Export/angry_motion_03_000.bvh',
    './Export/angry_motion_04_000.bvh',
    './Export/angry_motion_05_000.bvh',
    './Export/angry_motion_06_000.bvh',

    './Export/angry_motion_01_000_mirror.bvh',
    './Export/angry_motion_02_000_mirror.bvh',
    './Export/angry_motion_03_000_mirror.bvh',
    './Export/angry_motion_04_000_mirror.bvh',
    './Export/angry_motion_05_000_mirror.bvh',
    './Export/angry_motion_06_000_mirror.bvh',

    './Export/childlike_motion_01_000.bvh',
    './Export/childlike_motion_02_000.bvh',
    './Export/childlike_motion_03_000.bvh',
    './Export/childlike_motion_04_000.bvh',

    './Export/childlike_motion_01_000_mirror.bvh',
    './Export/childlike_motion_02_000_mirror.bvh',
    './Export/childlike_motion_03_000_mirror.bvh',
    './Export/childlike_motion_04_000_mirror.bvh',

    './Export/depressed_motion_01_000.bvh',
    './Export/depressed_motion_02_000.bvh',
    './Export/depressed_motion_03_000.bvh',

    './Export/depressed_motion_01_000_mirror.bvh',
    './Export/depressed_motion_02_000_mirror.bvh',
    './Export/depressed_motion_03_000_mirror.bvh',

    './Export/neutral_motion_01_000.bvh',
    './Export/neutral_motion_02_000.bvh',
    './Export/neutral_motion_03_000.bvh',
    './Export/neutral_motion_04_000.bvh',

    './Export/neutral_motion_01_000_mirror.bvh',
    './Export/neutral_motion_02_000_mirror.bvh',
    './Export/neutral_motion_03_000_mirror.bvh',
    './Export/neutral_motion_04_000_mirror.bvh',

    './Export/old_motion_01_000.bvh',
    './Export/old_motion_02_000.bvh',
    './Export/old_motion_03_000.bvh',
    './Export/old_motion_04_000.bvh',

    './Export/old_motion_01_000_mirror.bvh',
    './Export/old_motion_02_000_mirror.bvh',
    './Export/old_motion_03_000_mirror.bvh',
    './Export/old_motion_04_000_mirror.bvh',

    './Export/proud_motion_01_000.bvh',
    './Export/proud_motion_02_000.bvh',
    './Export/proud_motion_03_000.bvh',
    './Export/proud_motion_04_000.bvh',

    './Export/proud_motion_01_000_mirror.bvh',
    './Export/proud_motion_02_000_mirror.bvh',
    './Export/proud_motion_03_000_mirror.bvh',
    './Export/proud_motion_04_000_mirror.bvh',

    './Export/sexy_motion_01_000.bvh',
    './Export/sexy_motion_02_000.bvh',
    './Export/sexy_motion_03_000.bvh',

    './Export/sexy_motion_01_000_mirror.bvh',
    './Export/sexy_motion_02_000_mirror.bvh',
    './Export/sexy_motion_03_000_mirror.bvh',

    './Export/strutting_motion_01_000.bvh',
    './Export/strutting_motion_02_000.bvh',
    './Export/strutting_motion_03_000.bvh',

    './Export/strutting_motion_01_000_mirror.bvh',
    './Export/strutting_motion_02_000_mirror.bvh',
    './Export/strutting_motion_03_000_mirror.bvh',
]

""" Processing Functions """

def process_data(anim_in, anim_out, phase, gait, cls, type='flat'):
    """ Process Phase """
    
    # We throw away first and last frames of mocap as we need next and previous frames to preprocess
    dphase = phase[2:] - phase[1:-1]
    dphase[dphase < 0] = (1.0-phase[1:-1]+phase[2:])[dphase < 0]

    """ Process Style """

    cls_label = np.eye(8)[cls]
    
    """ Extract and Store Required Data """
     
    Pc, Xc, Yc = [], [], []

    count_in = 0  # counter over number of frames
    count_out = 0

    # keep_tuples discards unnecessary joints
    keep_tuples = [0,1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,18,20,21,22,23,24,25,27,29,30,31,32,33,34,36]

    for line in anim_in:
        values_in = line.split(' ')
        
        traj_pos_x_in = np.array(values_in[0:48:4]).astype(np.float32)
        traj_pos_z_in = np.array(values_in[1:48:4]).astype(np.float32)
        traj_dir_x_in = np.array(values_in[2:48:4]).astype(np.float32)
        traj_dir_z_in = np.array(values_in[3:48:4]).astype(np.float32)
        pos_xyz_in = zip(values_in[48::12], values_in[49::12], values_in[50::12])
        rot_fwd_xyz_in = zip(values_in[51::12], values_in[52::12], values_in[53::12])
        rot_up_xyz_in = zip(values_in[54::12], values_in[55::12], values_in[56::12])
        vel_xyz_in = zip(values_in[57::12], values_in[58::12], values_in[59::12])
        local_pos_in = np.array([pos_xyz_in[i] for i in keep_tuples]).flatten().astype(np.float32)
        local_rot_fwd_in = np.array([rot_fwd_xyz_in[i] for i in keep_tuples]).flatten().astype(np.float32)
        local_rot_up_in = np.array([rot_up_xyz_in[i] for i in keep_tuples]).flatten().astype(np.float32)
        local_vel_in = np.array([vel_xyz_in[i] for i in keep_tuples]).flatten().astype(np.float32)

        if count_in < window:
            pass
        elif count_in >= len(anim_in)-window-1:
            pass
        else:
            Pc.append(phase[count_in+1]) # plus 1 because the phase file has frame 0 but anim_in does not

            rootgait = gait[count_in-window:count_in+window:10]
            style_lbl = np.repeat(np.expand_dims(cls_label, axis=0), rootgait.shape[0], axis=0)

            Xc.append(np.hstack([
                traj_pos_x_in.ravel(), traj_pos_z_in.ravel(),
                traj_dir_x_in.ravel(), traj_dir_z_in.ravel(), 
                rootgait[:,0].ravel(), rootgait[:,1].ravel(), # Gait - Walk or Run
                style_lbl[:,0].ravel(), style_lbl[:,1].ravel(), # Style Labels
                style_lbl[:,2].ravel(), style_lbl[:,3].ravel(),
                style_lbl[:,4].ravel(), style_lbl[:,5].ravel(),
                style_lbl[:,6].ravel(), style_lbl[:,7].ravel(),
                local_pos_in.ravel(), 
                local_vel_in.ravel(),
                ]))
   
        count_in += 1

    for line in anim_out:
        values_out = line.split(' ')

        traj_pos_x_out = np.array(values_out[0:24:4]).astype(np.float32)
        traj_pos_z_out = np.array(values_out[1:24:4]).astype(np.float32)
        traj_dir_x_out = np.array(values_out[2:24:4]).astype(np.float32)
        traj_dir_z_out = np.array(values_out[3:24:4]).astype(np.float32)
        pos_xyz_out = zip(values_out[24:480:12], values_out[25:480:12], values_out[26:480:12])
        rot_fwd_xyz_out = zip(values_out[27:480:12], values_out[28:480:12], values_out[29:480:12])
        rot_up_xyz_out = zip(values_out[30:480:12], values_out[31:480:12], values_out[32:480:12])
        vel_xyz_out = zip(values_out[33:480:12], values_out[34:480:12], values_out[35:480:12])
        local_pos_out = np.array([pos_xyz_out[i] for i in keep_tuples]).flatten().astype(np.float32)
        local_rot_fwd_out = np.array([rot_fwd_xyz_out[i] for i in keep_tuples]).flatten().astype(np.float32)
        local_rot_up_out = np.array([rot_up_xyz_out[i] for i in keep_tuples]).flatten().astype(np.float32)
        local_vel_out = np.array([vel_xyz_out[i] for i in keep_tuples]).flatten().astype(np.float32)
        root_vel_x_out = np.array(values_out[480])
        root_rot_y_out = np.array(values_out[481])
        root_vel_z_out = np.array(values_out[482])        

        if count_out < window:
            pass
        elif count_out >= len(anim_out)-window-1:
            pass
        else:         
            Yc.append(np.hstack([
                root_vel_x_out.ravel(),
                root_vel_z_out.ravel(), 
                root_rot_y_out.ravel(),    
                dphase[count_out],
                traj_pos_x_out.ravel(), traj_pos_z_out.ravel(),
                traj_dir_x_out.ravel(), traj_dir_z_out.ravel(),
                local_pos_out.ravel(),
                local_vel_out.ravel(),
                local_rot_fwd_out.ravel(),
                local_rot_up_out.ravel()
                ]))

        count_out += 1
                                                   
    return np.array(Pc), np.array(Xc), np.array(Yc)
    

""" Phases, Inputs, Outputs """
    
P, X, Y = [], [], []
P_mirror, X_mirror, Y_mirror = [], [], []
            
for data in data_list:
    
    print('Processing Clip %s' % data)
    
    """ Data Types """
    type = 'flat'
    
    """ Load Data """
    
    # Preprocessing in Unity converts the animation from 120 to 60 fps
    if len(data.split('/')[2].split('_')) == 5:  # Mirrored clips
        with open(data.replace('_mirror.bvh', '.bvh_Mirror_Input.txt'), 'r') as f:
            anim_in = f.readlines()

        with open(data.replace('_mirror.bvh', '.bvh_Mirror_Output.txt'), 'r') as f:
            anim_out = f.readlines()
    else:
        with open(data + '_Default_Input.txt', 'r') as f:
            anim_in = f.readlines()

        with open(data + '_Default_Output.txt', 'r') as f:
            anim_out = f.readlines()

    """ Load Phase / Gait / Style """
        
    styletransfer_styles = [
    'angry', 'childlike', 'depressed', 'neutral', 
    'old', 'proud', 'sexy', 'strutting']

    phase = np.loadtxt(data.replace('.bvh', '.phase'))[::2] # convert 120 to 60 fps 
    gait = np.loadtxt(data.replace('.bvh', '.gait'))[::2] # convert 120 to 60 fps
    cls = styletransfer_styles.index(data.split('/')[2].split('_')[0]) 

    """ Preprocess Data """    
    
    Pc, Xc, Yc = process_data(anim_in, anim_out, phase, gait, cls, type=type)

    if len(data.split('/')[2].split('_')) == 5:
        with open(data.replace('_mirror.bvh', '_footsteps.txt'), 'r') as f:
            footsteps = f.readlines()
    else:
        with open(data.replace('.bvh', '_footsteps.txt'), 'r') as f:
            footsteps = f.readlines()
    
    """ Slice data into one cycle length pieces """
    
    for li in range(len(footsteps)-1):  
    
        curr, next = footsteps[li+0].split('\t'), footsteps[li+1].split('\t')
        
        """ Ignore frames before first and after last footstep """
        
        if len(next) <  2: continue # ignore if only one footstep on a given line
        if int(curr[0])//2-window-1 < 0: continue  # if first footstep does not have enough history, ignore it
        if int(next[0])//2-window-1 >= len(Xc): continue  

        
        """ Slice and Append Data """
        # divide by 2 because of framerate change. 
        slc = slice(int(curr[0])//2-window-1, int(next[0])//2-window-1) # -1 again because the footsteps file has frame 0 but anim_in does not  

        if len(data.split('/')[2].split('_')) == 5: 
            P_mirror.append(np.hstack([0.5, Pc[slc][1:]]).astype(np.float32))
            X_mirror.append(Xc[slc].astype(np.float32))
            Y_mirror.append(Yc[slc].astype(np.float32))
        else:
            P.append(np.hstack([0.0, Pc[slc][1:]]).astype(np.float32))
            X.append(Xc[slc].astype(np.float32))
            Y.append(Yc[slc].astype(np.float32))
        
  
""" Clip Statistics """
  
print('Total Clips: %i' % (len(X) + len(X_mirror)))
print('Shortest Clip: %i' % min(map(len,X)))
print('Longest Clip: %i' % max(map(len,X)))
print('Average Clip: %i' % np.mean(list(map(len,X))))

""" Merge Clips """

print('Merging Clips...')

Xin = np.concatenate(X, axis=0)
Yin = np.concatenate(Y, axis=0)
Pin = np.concatenate(P, axis=0)
Xin_mirror = np.concatenate(X_mirror, axis=0)
Yin_mirror = np.concatenate(Y_mirror, axis=0)
Pin_mirror = np.concatenate(P_mirror, axis=0)

print(Xin.shape, Yin.shape, Pin.shape)
print(Xin_mirror.shape, Yin_mirror.shape, Pin_mirror.shape)

print('Saving Database...')

np.savez_compressed('style_database3.npz', Xin=Xin, Yin=Yin, Pin=Pin, Xin_mirror=Xin_mirror, Yin_mirror=Yin_mirror, Pin_mirror=Pin_mirror)
