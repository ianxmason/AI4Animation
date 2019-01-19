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

""" Options """

rng = np.random.RandomState(1234)
window = 60
njoints = 31

""" Data """

data_fewshot = [
    './Export/Balance_000.bvh',
    './Export/Balance_001.bvh',
    './Export/Balance_003.bvh',
    './Export/Balance_004.bvh',
    './Export/Balance_006.bvh',
    './Export/Balance_007.bvh',
    './Export/Balance_008.bvh',
    './Export/Balance_000_mirror.bvh',
    './Export/Balance_001_mirror.bvh',
    './Export/Balance_003_mirror.bvh',
    './Export/Balance_004_mirror.bvh',
    './Export/Balance_006_mirror.bvh',
    './Export/Balance_007_mirror.bvh',
    './Export/Balance_008_mirror.bvh',

    './Export/BentForward_000.bvh',
    './Export/BentForward_001.bvh',
    './Export/BentForward_000_mirror.bvh',
    './Export/BentForward_001_mirror.bvh',

    './Export/BentKnees_000.bvh',
    './Export/BentKnees_000_mirror.bvh',

    './Export/Bouncy_000.bvh',
    './Export/Bouncy_001.bvh',
    './Export/Bouncy_002.bvh',
    './Export/Bouncy_003.bvh',
    # './Export/Bouncy_000_mirror.bvh',
    # './Export/Bouncy_001_mirror.bvh',
    # './Export/Bouncy_002_mirror.bvh',
    # './Export/Bouncy_003_mirror.bvh',

    './Export/Cat_000.bvh',
    './Export/Cat_000_mirror.bvh',

    './Export/Chicken_000.bvh',
    './Export/Chicken_000_mirror.bvh',

    './Export/Cool_000.bvh',
    './Export/Cool_000_mirror.bvh',  

    './Export/Crossover_000.bvh',
    './Export/Crossover_000_mirror.bvh',

    './Export/Crouched_000.bvh',
    './Export/Crouched_000_mirror.bvh',

    './Export/Dance3_000.bvh',
    # './Export/Dance3_000_mirror.bvh',

    './Export/Dinosaur_000.bvh',
    './Export/Dinosaur_000_mirror.bvh',

    './Export/DragLeg_000.bvh',
    # './Export/DragLeg_000_mirror.bvh',

    './Export/Drunk_001.bvh',
    './Export/Drunk_001_mirror.bvh',

    './Export/DuckFoot_000.bvh',
    './Export/DuckFoot_000_mirror.bvh',    

    './Export/Elated_000.bvh',
    './Export/Elated_000_mirror.bvh',

    './Export/Frankenstein_000.bvh',
    './Export/Frankenstein_000_mirror.bvh',

    './Export/Gangly_000.bvh',
    './Export/Gangly_000_mirror.bvh',

    './Export/Gedanbarai_000.bvh',
    './Export/Gedanbarai_000_mirror.bvh',

    './Export/Graceful_000.bvh',
    './Export/Graceful_000_mirror.bvh',

    './Export/Heavyset_000.bvh',
    './Export/Heavyset_000_mirror.bvh',

    './Export/Heiansyodan_000.bvh',
    './Export/Heiansyodan_000_mirror.bvh',

    './Export/Hobble_000.bvh',
    # './Export/Hobble_000_mirror.bvh',

    './Export/HurtLeg_000.bvh',
    # './Export/HurtLeg_000_mirror.bvh',

    './Export/Jaunty_000.bvh',
    './Export/Jaunty_000_mirror.bvh',

    './Export/Joy_000.bvh',
    './Export/Joy_000_mirror.bvh',

    './Export/LeanRight_000.bvh',
    './Export/LeanRight_001.bvh',
    # './Export/LeanRight_000_mirror.bvh',
    # './Export/LeanRight_001_mirror.bvh',

    './Export/LeftHop_002.bvh',
    './Export/LeftHop_003.bvh',
    './Export/LeftHop_004.bvh',
    # './Export/LeftHop_002_mirror.bvh',
    # './Export/LeftHop_003_mirror.bvh',
    # './Export/LeftHop_004_mirror.bvh',

    './Export/LegsApart_000.bvh',
    './Export/LegsApart_000_mirror.bvh',

    './Export/Mantis_000.bvh',
    './Export/Mantis_000_mirror.bvh',

    './Export/March_001.bvh',
    './Export/March_001_mirror.bvh',

    './Export/Mawashigeri_000.bvh',
    './Export/Mawashigeri_000_mirror.bvh',

    './Export/OnToesBentForward_000.bvh',
    './Export/OnToesBentForward_001.bvh',
    './Export/OnToesBentForward_000_mirror.bvh',
    './Export/OnToesBentForward_001_mirror.bvh',

    './Export/OnToesCrouched_000.bvh',
    './Export/OnToesCrouched_001.bvh',
    './Export/OnToesCrouched_000_mirror.bvh',
    './Export/OnToesCrouched_001_mirror.bvh',

    './Export/PainfulLeftknee_000.bvh',
    # './Export/PainfulLeftknee_000_mirror.bvh',

    './Export/Penguin_000.bvh',
    './Export/Penguin_000_mirror.bvh',

    './Export/PigeonToed_000.bvh',
    './Export/PigeonToed_000_mirror.bvh',

    './Export/PrarieDog_000.bvh',
    './Export/PrarieDog_000_mirror.bvh',

    './Export/Quail_001.bvh',
    './Export/Quail_001_mirror.bvh',

    './Export/Roadrunner_000.bvh',
    './Export/Roadrunner_000_mirror.bvh',

    './Export/Rushed_000.bvh',
    './Export/Rushed_000_mirror.bvh',

    './Export/Sneaky_004.bvh',
    './Export/Sneaky_004_mirror.bvh',

    './Export/Squirrel_001.bvh',
    './Export/Squirrel_001_mirror.bvh',

    './Export/Stern_000.bvh',
    './Export/Stern_001.bvh',
    './Export/Stern_002.bvh',
    './Export/Stern_003.bvh',
    './Export/Stern_000_mirror.bvh',
    './Export/Stern_001_mirror.bvh',
    './Export/Stern_002_mirror.bvh',
    './Export/Stern_003_mirror.bvh',

    './Export/Stuff_000.bvh',
    './Export/Stuff_000_mirror.bvh',

    './Export/SwingShoulders_000.bvh',
    './Export/SwingShoulders_001.bvh',
    './Export/SwingShoulders_002.bvh',
    './Export/SwingShoulders_000_mirror.bvh',
    './Export/SwingShoulders_001_mirror.bvh',
    './Export/SwingShoulders_002_mirror.bvh',

    './Export/WildArms_000.bvh',
    './Export/WildArms_000_mirror.bvh',

    './Export/WildLegs_000.bvh',
    './Export/WildLegs_000_mirror.bvh',

    './Export/WoundedLeg_000.bvh',
    './Export/WoundedLeg_001.bvh',
    './Export/WoundedLeg_002.bvh',
    # './Export/WoundedLeg_000_mirror.bvh',
    # './Export/WoundedLeg_001_mirror.bvh',
    # './Export/WoundedLeg_002_mirror.bvh',

    './Export/Yokogeri_000.bvh',
    # './Export/Yokogeri_000_mirror.bvh',

    './Export/Zombie_000.bvh',
    './Export/Zombie_001.bvh',
    './Export/Zombie_002.bvh',
    './Export/Zombie_000_mirror.bvh',
    './Export/Zombie_001_mirror.bvh',
    './Export/Zombie_002_mirror.bvh',    
]


""" Processing Functions """

def process_data(anim_in, anim_out, phase, gait, cls, type='flat'):
    """ Process Phase """
    
    # We throw away first and last frames of mocap as we need next and previous frames to preprocess
    dphase = phase[2:] - phase[1:-1]
    dphase[dphase < 0] = (1.0-phase[1:-1]+phase[2:])[dphase < 0]

    """ Process Style """

    cls_label = np.eye(50)[cls]
    
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
            Pc.append(phase[count_in+1])

            rootgait = gait[count_in-window:count_in+window:10]
            style_lbl = np.repeat(np.expand_dims(cls_label, axis=0), rootgait.shape[0], axis=0)

            Xc.append(np.hstack([
                traj_pos_x_in.ravel(), traj_pos_z_in.ravel(),
                traj_dir_x_in.ravel(), traj_dir_z_in.ravel(), 
                rootgait[:,0].ravel(), rootgait[:,1].ravel(), # Gait - Walk or Run
                style_lbl[:,0].ravel(), style_lbl[:,1].ravel(), # Style labels
                style_lbl[:,2].ravel(), style_lbl[:,3].ravel(),
                style_lbl[:,4].ravel(), style_lbl[:,5].ravel(),
                style_lbl[:,6].ravel(), style_lbl[:,7].ravel(),
                style_lbl[:,8].ravel(), style_lbl[:,9].ravel(),
                style_lbl[:,10].ravel(), style_lbl[:,11].ravel(),
                style_lbl[:,12].ravel(), style_lbl[:,13].ravel(),
                style_lbl[:,14].ravel(), style_lbl[:,15].ravel(),
                style_lbl[:,16].ravel(), style_lbl[:,17].ravel(),
                style_lbl[:,18].ravel(), style_lbl[:,19].ravel(),
                style_lbl[:,20].ravel(), style_lbl[:,21].ravel(),
                style_lbl[:,22].ravel(), style_lbl[:,23].ravel(),
                style_lbl[:,24].ravel(), style_lbl[:,25].ravel(),
                style_lbl[:,26].ravel(), style_lbl[:,27].ravel(),
                style_lbl[:,28].ravel(), style_lbl[:,29].ravel(),
                style_lbl[:,30].ravel(), style_lbl[:,31].ravel(),
                style_lbl[:,32].ravel(), style_lbl[:,33].ravel(),
                style_lbl[:,34].ravel(), style_lbl[:,35].ravel(),
                style_lbl[:,36].ravel(), style_lbl[:,37].ravel(),
                style_lbl[:,38].ravel(), style_lbl[:,39].ravel(),
                style_lbl[:,40].ravel(), style_lbl[:,41].ravel(),
                style_lbl[:,42].ravel(), style_lbl[:,43].ravel(),
                style_lbl[:,44].ravel(), style_lbl[:,45].ravel(),
                style_lbl[:,46].ravel(), style_lbl[:,47].ravel(),
                style_lbl[:,48].ravel(), style_lbl[:,49].ravel(),
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
            
for data in data_fewshot:
    
    print('Processing Clip %s' % data)
    
    """ Data Types """
    type = 'flat'
    
    """ Load Data """
    
    # Preprocessing in Unity converts the animation from 120 to 60 fps
    if len(data.split('/')[2].split('_')) == 3:  # Mirrored clips
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
    'Balance', 'BentForward', 'BentKnees', 'Bouncy', 'Cat', 'Chicken', 'Cool',
    'Crossover', 'Crouched', 'Dance3', 'Dinosaur', 'DragLeg', 'Drunk',
    'DuckFoot', 'Elated', 'Frankenstein', 'Gangly',
    'Gedanbarai', 'Graceful', 'Heavyset', 'Heiansyodan', 'Hobble',
    'HurtLeg', 'Jaunty', 'Joy', 'LeanRight', 'LeftHop', 'LegsApart', 'Mantis',
    'March', 'Mawashigeri', 'OnToesBentForward', 'OnToesCrouched',
    'PainfulLeftknee', 'Penguin', 'PigeonToed', 'PrarieDog', 'Quail', 'Roadrunner',
    'Rushed', 'Sneaky', 'Squirrel', 'Stern', 'Stuff',
    'SwingShoulders', 'WildArms', 'WildLegs', 'WoundedLeg',
    'Yokogeri', 'Zombie']

    phase = np.loadtxt(data.replace('.bvh', '.phase'))[::2]
    gait = np.loadtxt(data.replace('.bvh', '.gait'))[::2]
    cls = styletransfer_styles.index(data.split('/')[2].split('_')[0]) 

    """ Preprocess Data """    
    
    Pc, Xc, Yc = process_data(anim_in, anim_out, phase, gait, cls, type=type)

    if len(data.split('/')[2].split('_')) == 3:
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
        slc = slice(int(curr[0])//2-window-1, int(next[0])//2-window-1)   

        if len(data.split('/')[2].split('_')) == 3:
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

np.savez_compressed('fewshot_database3.npz', Xin=Xin, Yin=Yin, Pin=Pin, Xin_mirror=Xin_mirror, Yin_mirror=Yin_mirror, Pin_mirror=Pin_mirror)

