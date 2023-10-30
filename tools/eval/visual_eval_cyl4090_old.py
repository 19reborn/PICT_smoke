# /home/yiming/Documents/workspace/Project_PINF/mantaflow_nogui/build/manta /home/yiming/Documents/workspace/Project_PINF/pinf_clean/tools/eval/visual_eval_cyl4090.py

import os, sys, shutil, time, math, platform, datetime, inspect
import numpy as np
# import _visualize as v
from manta import *
from enum import Enum
import imageio
import subprocess
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from _tools import vel_uv2hsv, jacobian3D_np,jacobian2D_np,vor_rgb


ref_path = '/home/yiming/Documents/workspace/Project_PINF/eval_sample_data/gt/'

out_path = '/home/yiming/Documents/workspace/Project_PINF/pinf_clean/log/evaluate/' + '/cyl/v1_1030_test/'
our_path = '/home/yiming/Documents/workspace/Project_PINF/pinf_clean/log/cyl/1029_v3_no_neus_early_terminated/volumeout_080001/'

glo_path = None

# if not os.path.exists(out_path):os.mkdir(out_path)
os.makedirs(out_path, exist_ok=True)
    
den_file, vel_file = "density_high_%04d.f16.npz", "velocity_high_%04d.f16.npz"

# ref_gs = vec3(100,150,100) # 178
# our_gs = vec3(128,192,128)
ref_gs = vec3(256,256,256) # 178
our_gs = vec3(256,256,256)

# st = vec3(7,11,27)
# sz = vec3(102,156,87)
st = vec3(0,0,0)
sz = vec3(0,0,0)
pad = our_gs - sz - st
sz = [int(sz.x), int(sz.y), int(sz.z)]
st = [int(st.x), int(st.y), int(st.z)]
pad = [int(pad.x), int(pad.y), int(pad.z)]
# padding = ((0,0),(st[0],pad[0]),(st[1],pad[1]),(st[2],pad[2]),(0,0))
padding = None

def setSolver(gs, name):
    s   = FluidSolver(name=name, gridSize = gs)
    flags    = s.create(FlagGrid)
    vel      = s.create(MACGrid)
    density  = s.create(RealGrid)

    flags.initDomain(boundaryWidth=1)
    flags.fillGrid()
    setOpenBound(flags, 1,'xXyYzZ',FlagOutflow|FlagEmpty) 

    return s, [flags, density, vel]

def load_numpy(filename, grid=None, padding=None, flip=False):
    _npy = np.load(filename)
    npy = [ _npy[k] for k in _npy.files if k in ['data','den','vel','arr_0']]
    if len(npy) > 0: 
        npy = npy[0]
    else:
        npy = [ _npy[k][:,:sz[0],:sz[1],:sz[2],:] for k in ['vel_x','vel_y','vel_z']]
        if flip: npy[-1] = npy[-1] * -1.0
        npy = np.concatenate(npy,axis=-1)

    npy = np.float32(npy)

    if npy.shape[0] == 100 and npy.shape[1] > 150: # only for scalarflow, todo resolution
        npy = npy[:,:150,...]
    if padding is not None:
        # print(npy.shape)
        npy = np.pad(npy, padding, 'constant')
        # print(npy.shape)
    if flip: npy = npy[:,::-1,...]
    if grid is not None:
        if npy.shape[-1] == 3:
            copyArrayToGridMAC(npy, grid) # todo, centered,
        else:
            copyArrayToGridReal(npy, grid)

    return npy


def readData(basedir, den_file=None, vel_file=None, grids=[None,None,None], padding=None, flip=False):
    if (den_file is not None) and (grids[1] is not None):
        den_np = load_numpy(os.path.join(basedir, den_file), grids[1],padding=padding,flip=flip)
    else:
        den_np = None
    if (vel_file is not None) and (grids[2] is not None):
        vel_np = load_numpy(os.path.join(basedir, vel_file), grids[2],padding=padding,flip=flip)
    else:
        vel_np = None
    return den_np, vel_np

def saveVel(Vin, path, scale=144):
    # imageio.imwrite( path, vel_uv2hsv(Vin, scale=scale, is3D=True, logv=False)[::-1])
    imageio.imwrite( path, vel_uv2hsv(Vin[:,::-1,...], scale=scale, is3D=True, logv=False)[::-1])
    # imageio.imwrite( path, vel_uv2hsv(Vin, scale=scale, is3D=True, logv=False)[::-1,:,::-1])

def save_fig(den, vel, image_dir, den_name='den_%04d.ppm', vel_name='vel_%04d.png', t=0, is2D=False, vN=1.0):
    os.makedirs(image_dir, exist_ok = True) 
    if is2D:
        if den is not None:
            projectPpmFull( den, image_dir+ den_name% (t), 0, 1.0 )
        if vel is not None:
            saveVel(np.squeeze(vel), image_dir+ vel_name% (t))
            _, NETw = jacobian2D_np(vel)
            # imageio.imwrite(image_dir+'vor_%04d.png' % (t), vor_rgb(np.squeeze(NETw))[:,:,::-1])
            imageio.imwrite(image_dir+'vor_%04d.png' % (t), vor_rgb(np.squeeze(NETw[:,::-1,...]))[:,:,::-1])
            # imageio.imwrite(image_dir+'vor_%04d.png' % (t), vor_rgb(np.squeeze(NETw))[::-1,:,::-1])
    else:
        if den is not None:
            projectPpmFull( den, image_dir+den_name % (t), 0, 0.3 )
        if vel is not None:
            saveVel(np.squeeze(vel), image_dir+vel_name % (t))
            _, NETw = jacobian3D_np(vel)
            # saveVel(np.squeeze(NETw), image_dir+"vort_"+vel_name % (t), scale=720*vN)
            saveVel(np.squeeze(NETw), image_dir+"vort_"+vel_name % (t), scale=256)

def np_l2(np_array): return np.sum(np.square(np_array),axis=-1)

ref_s, oref_grids = setSolver(ref_gs, 'ref')
our_s, our_grids = setSolver(our_gs, 'our')

if False: # loading convex hull
    _data_path = _p +"nobackup/nerf/ex_TR20210419/mydata/synth/hull/"
    hull_full = []
    for framei in range(60,120,2): # [100]
        print(framei)
        hull_file = glo_path + "frame_%06d/volume_hull.npz"
        hull_data = load_numpy(hull_file%framei,grid=our_grids[1], padding=padding,flip=True) # shape 1,102,156,87,1
        our_grids[1].multConst(0.2)
        save_fig(our_grids[1],None, _data_path, den_name = "hull_%04d.ppm", t=framei)
        saveVisGrid(hull_data[0]*2, os.path.join(_data_path, 'hull_%04d.jpg'%framei) , drawKind.den3D, 1.0)

        hull_full.append(hull_data)
    hull_full = np.concatenate(hull_full, axis=0)
    print(hull_full.shape) # (30, 128, 192, 128, 1)
    np.savez_compressed(os.path.join(_data_path, 'hull.npz'),hull_full)
    exit()

if ref_gs.x != our_gs.x: # same res for evaluation
    newr_vel  = our_s.create(MACGrid)
    newr_den  = our_s.create(RealGrid)
    ref_grids = [our_grids[0], newr_den, newr_vel]
else:
    ref_grids = oref_grids


if glo_path is not None:
    glo_vel  = our_s.create(MACGrid)
    glo_den  = our_s.create(RealGrid)
    glo_grids = [our_grids[0], glo_den, glo_vel]

normDen = True
hullmask = False
denratio = 1.2 # for visualization

def printVelRange(gvel, name=""):
    print(name+" vel_x range", gvel[...,0].min(), gvel[...,0].mean(), gvel[...,0].max())
    print(name+" vel_y range", gvel[...,1].min(), gvel[...,1].mean(), gvel[...,1].max())
    print(name+" vel_z range", gvel[...,2].min(), gvel[...,2].mean(), gvel[...,2].max())
    gvel2 = np_l2(gvel)
    print(name+" velsq range", gvel2.min(), gvel2.mean(), gvel2.max())

# forward+backward warp diff
def warpMidTest(preV, preD, newV, newD, DtoolGrid, VtoolGrid, FtoolGrid, t_step=2):
    midV,midD = np.copy(newV), np.copy(newD)
    _preV,_preD = np.copy(preV), np.copy(preD)
    # warp pre forward, t_step/2.0
    copyArrayToGridReal(target=DtoolGrid, source=preD)
    copyArrayToGridMAC(target=VtoolGrid, source=preV*(t_step/2.0))
    advectSemiLagrange(flags=FtoolGrid, vel=VtoolGrid, grid=DtoolGrid, order=2, strength=0.8)    
    advectSemiLagrange(flags=FtoolGrid, vel=VtoolGrid, grid=VtoolGrid, order=2, strength=0.8)
    copyGridToArrayReal( source=DtoolGrid, target=_preD ) 
    copyGridToArrayMAC( source=VtoolGrid, target=_preV)
    _preV = _preV*2.0/(t_step)
    # warp new backward, t_step/2.0
    copyArrayToGridReal(target=DtoolGrid, source=newD)
    copyArrayToGridMAC(target=VtoolGrid, source=(-newV*(t_step/2.0)))
    advectSemiLagrange(flags=FtoolGrid, vel=VtoolGrid, grid=DtoolGrid, order=2, strength=0.8)
    advectSemiLagrange(flags=FtoolGrid, vel=VtoolGrid, grid=VtoolGrid, order=2, strength=0.8)
    copyGridToArrayReal( source=DtoolGrid, target=midD ) 
    copyGridToArrayMAC( source=VtoolGrid, target=midV)
    midV = -midV*2.0/(t_step)
    return (midD-_preD), (midV-_preV)

# forward warp diff
def warpTest(preV, preD, newV, newD, DtoolGrid, VtoolGrid, FtoolGrid, t_step=2):
    _preV,_preD = np.copy(preV), np.copy(preD)
    copyArrayToGridReal(target=DtoolGrid, source=preD)
    copyArrayToGridMAC(target=VtoolGrid, source=preV*t_step)
    advectSemiLagrange(flags=FtoolGrid, vel=VtoolGrid, grid=DtoolGrid, order=2, strength=0.8)    
    advectSemiLagrange(flags=FtoolGrid, vel=VtoolGrid, grid=VtoolGrid, order=2, strength=0.8)
    copyGridToArrayReal( source=DtoolGrid, target=_preD ) 
    copyGridToArrayMAC( source=VtoolGrid, target=_preV)
    _preV = _preV/(t_step)
    return np.abs(newD-_preD), (newV-_preV)


def Vmetrics(ovel=None, rvel=None, oden=None, rden=None, oVwarp=None, oDwarp=None, \
    oVwarpMid=None, oDwarpMid=None, frameHull=None, ignoreInflow=False, nameonly=False, printname=""):
    names = ["den_l2", "vel_l2", "vel_div", "warp_vell2", "warp_denl2", "warpMid_vell2", "warpMid_denl2"]
    if nameonly: return names

    l2V = np.float32(0) if any(x is None for x in [rvel, ovel]) else np_l2(rvel - ovel)
    # divV = np.float32(0) if ovel is None else np.abs(np.squeeze(divergence3D_np(ovel)))
    divV = np.float32(0) ## todo:: calculate this
    l2D = np.float32(0) if any(x is None for x in [oden, rden]) else np_l2(oden - rden)

    l2Vwarp = np.float32(0) if oVwarp is None else np_l2(oVwarp) 
    l2Dwarp = np.float32(0) if oDwarp is None else np_l2(oDwarp)

    l2VwarpMid = np.float32(0) if oVwarpMid is None else np_l2(oVwarpMid) 
    l2DwarpMid = np.float32(0) if oDwarpMid is None else np_l2(oDwarpMid)

    metrics = [l2D, l2V, divV, l2Vwarp, l2Dwarp, l2VwarpMid, l2DwarpMid]
    if ignoreInflow: # ignore inflow part
        # metrics = [(np.float32(0) if len(a.shape) == 0 else a[:,25:,:]) for a in metrics]
        # if frameHull is not None: frameHull = np.squeeze(frameHull)[:,25:,:] # todo:: why 25:?
        pass 
    hullmean = 1.0
    if frameHull is not None: hullmean = max(frameHull.mean(), 1e-6)
    if printname!= "":
        print(" ".join([printname]+[a+str(b.shape)+":"+str(b.mean()/hullmean) for a,b in zip(names, metrics)]))
    return [m.mean()/hullmean for m in metrics]


our_list, glo_list, ref_list =[], [], []
fr = 0

testWarp = True
for framei in range(15,115,10): # [100]
# for framei in range(15,25,10): # [100]
    print(framei)
    rden, rvel = readData(ref_path, den_file%framei, vel_file%framei, oref_grids)
    rden = np.reshape(rden, [int(ref_gs.z),int(ref_gs.y),int(ref_gs.x), -1])
    rvel = np.reshape(rvel, [int(ref_gs.z),int(ref_gs.y),int(ref_gs.x), -1])

    
    if our_path is not None:
        oden, ovel = readData(our_path, "d_%04d.npz"%framei, "v_%04d.npz"%framei, our_grids)
        oden = np.reshape(oden, [int(our_gs.z),int(our_gs.y),int(our_gs.x), -1])
        ovel = np.reshape(ovel, [int(our_gs.z),int(our_gs.y),int(our_gs.x), -1])

    if glo_path is not None:
        hull_data = load_numpy(glo_path+"frame_%06d/volume_hull.npz"%framei,grid=None, padding=padding,flip=True) # shape 1,102,156,87,1
        gden, gvel = readData(glo_path,"frame_%06d/density.npz"%framei, "frame_%06d/velocity.npz"%framei, glo_grids,padding=padding,flip=True)
        gden = np.reshape(gden, [int(our_gs.z),int(our_gs.y),int(our_gs.x), -1]) 
        gvel = np.reshape(gvel, [int(our_gs.z),int(our_gs.y),int(our_gs.x), -1])
        gvel *= 0.5 # scalarflow only, 2 frames

    if framei>60:
    # if True:
        rden_warpMID, rvel_warpMID = warpMidTest(_rvel_warp, _rden_warp, rvel, rden, oref_grids[1], oref_grids[2], oref_grids[0])
        rden_warp, rvel_warp = warpTest(_rvel_warp, _rden_warp, rvel, rden, oref_grids[1], oref_grids[2], oref_grids[0])
        
        if our_path is not None:
            oden_warpMID, ovel_warpMID = warpMidTest(_ovel_warp, _oden_warp, ovel, oden, our_grids[1], our_grids[2], our_grids[0])
            oden_warp, ovel_warp = warpTest(_ovel_warp, _oden_warp, ovel, oden, our_grids[1], our_grids[2], our_grids[0])
            
        if glo_path is not None:
            gden_warpMID, gvel_warpMID = warpMidTest(_gvel_warp, _gden_warp, gvel, gden, glo_grids[1], glo_grids[2], glo_grids[0])
            gden_warp, gvel_warp = warpTest(_gvel_warp, _gden_warp, gvel, gden, glo_grids[1], glo_grids[2], glo_grids[0])
               
        
    if testWarp:
        if our_path is not None:
            _oden_warp, _ovel_warp = np.copy(oden), np.copy(ovel)
        if glo_path is not None:
            _gden_warp, _gvel_warp = np.copy(gden), np.copy(gvel)
        _rden_warp, _rvel_warp = np.copy(rden), np.copy(rvel)        
        
    if (ref_gs.x != our_gs.x) and (our_path is not None):
        if framei>60:
            copyArrayToGridReal(target=oref_grids[1], source=rden_warpMID)
            copyArrayToGridMAC(target=oref_grids[2], source=rvel_warpMID)
            interpolateGrid( target=ref_grids[1], source=oref_grids[1] )
            interpolateMACGrid(target=ref_grids[2], source=oref_grids[2] )
            ref_grids[2].multConst(our_gs/ref_gs)
            rden_warpMID = np.copy(oden)
            rvel_warpMID = np.copy(ovel)
            copyGridToArrayReal(target=rden_warpMID, source=ref_grids[1])
            copyGridToArrayMAC(target=rvel_warpMID, source=ref_grids[2])

            copyArrayToGridReal(target=oref_grids[1], source=rden_warp)
            copyArrayToGridMAC(target=oref_grids[2], source=rvel_warp)
            interpolateGrid( target=ref_grids[1], source=oref_grids[1] )
            interpolateMACGrid(target=ref_grids[2], source=oref_grids[2] )
            ref_grids[2].multConst(our_gs/ref_gs)
            rden_warp = np.copy(oden)
            rvel_warp = np.copy(ovel)
            copyGridToArrayReal(target=rden_warp, source=ref_grids[1])
            copyGridToArrayMAC(target=rvel_warp, source=ref_grids[2])

        copyArrayToGridReal(target=oref_grids[1], source=rden)
        copyArrayToGridMAC(target=oref_grids[2], source=rvel)
        interpolateGrid( target=ref_grids[1], source=oref_grids[1] )
        interpolateMACGrid(target=ref_grids[2], source=oref_grids[2] )
        ref_grids[2].multConst(our_gs/ref_gs)
        rvel = np.copy(ovel)
        rden = np.copy(oden)
        copyGridToArrayReal(target=rden, source=ref_grids[1])
        copyGridToArrayMAC(target=rvel, source=ref_grids[2])
    
    if hullmask and (glo_path is not None):
        if True: save_fig(None, rvel, out_path+"oriref_", t=fr)
        if True and (our_path is not None): save_fig(None, ovel, out_path+"oriour_", t=fr, vN=1.0)
        if True: save_fig(None, gvel, out_path+"origlo_", t=fr)

        rden *= hull_data[0]
        rvel *= hull_data[0]
        gden *= hull_data[0]
        gvel *= hull_data[0]
        if our_path is not None:
            oden *= hull_data[0]
            ovel *= hull_data[0]
        if framei>60:
            if our_path is not None:
                oden_warp *= hull_data[0]; oden_warpMID *= hull_data[0]
                ovel_warp *= hull_data[0]; ovel_warpMID *= hull_data[0]
            gden_warp *= hull_data[0]; gden_warpMID *= hull_data[0]
            gvel_warp *= hull_data[0]; gvel_warpMID *= hull_data[0]
            rden_warp *= hull_data[0]; rden_warpMID *= hull_data[0]
            rvel_warp *= hull_data[0]; rvel_warpMID *= hull_data[0]
    
    # print("den means, ref",rden.mean(),"our", oden.mean(), "glo", gden.mean())
    if normDen and (our_path is not None):
        odscale = rden.mean()/oden.mean() 
        oden = oden * odscale
        if framei>60:
            oden_warp = oden_warp * odscale; oden_warpMID = oden_warpMID * odscale

    if framei>60: ## todo:: what this means?
    # if True:
        if our_path is not None:
            copyArrayToGridReal(target=our_grids[1], source=np.abs(oden_warp)*denratio*5.0)
            if True: save_fig(our_grids[1], ovel_warp, out_path+"warp_our_", t=fr) 
            copyArrayToGridReal(target=our_grids[1], source=np.abs(oden_warpMID)*denratio*5.0)
            if True: save_fig(our_grids[1], ovel_warpMID, out_path+"warpMID_our_", t=fr)           
        if glo_path is not None:
            copyArrayToGridReal(target=glo_grids[1], source=np.abs(gden_warp)*denratio*5.0)
            if True: save_fig(glo_grids[1], gvel_warp, out_path+"warp_glo_", t=fr)
            copyArrayToGridReal(target=glo_grids[1], source=np.abs(gden_warpMID)*denratio*5.0)
            if True: save_fig(glo_grids[1], gvel_warpMID, out_path+"warpMID_glo_", t=fr)
        copyArrayToGridReal(target=ref_grids[1], source=np.abs(rden_warp)*denratio*5.0)
        if True: save_fig(ref_grids[1], rvel_warp, out_path+"warp_ref_", t=fr)
        copyArrayToGridReal(target=ref_grids[1], source=np.abs(rden_warpMID)*denratio*5.0)
        if True: save_fig(ref_grids[1], rvel_warpMID, out_path+"warpMID_ref_", t=fr)

    printVelRange(rvel, "ref")
    if our_path is not None: printVelRange(ovel, "our")
    if glo_path is not None: printVelRange(gvel, "glo")
    print("ref den range", rden.min(), rden.mean(), rden.max())
    if our_path is not None: print("our den range", oden.min(), oden.mean(), oden.max())
    if glo_path is not None: print("glo den range", gden.min(), gden.mean(), gden.max())

    copyArrayToGridReal(target=ref_grids[1], source=rden)
    if our_path is not None: copyArrayToGridReal(target=our_grids[1], source=oden)
    if glo_path is not None: copyArrayToGridReal(target=glo_grids[1], source=gden)
    if True: save_fig(ref_grids[1], rvel, out_path+"ref_", t=fr)
    if our_path is not None: save_fig(our_grids[1], ovel, out_path+"our_", t=fr, vN=1.0)
    if glo_path is not None: save_fig(glo_grids[1], gvel, out_path+"glo_", t=fr)

    # with inflow:
    # uh = hull_data[0] # None 
    uh =  None 
    if our_path is not None: _our = Vmetrics(ovel, rvel, oden, rden, frameHull=uh, printname="our")[:2]
    if glo_path is not None: _glo = Vmetrics(gvel, rvel, gden, rden, frameHull=uh, printname="glo")[:2]
    _ref = Vmetrics(rvel, None, rden, None, frameHull=uh, printname="ref")[:2]
    # w.o. inflow:
    # if framei==60:
        # ovel_warp,gvel_warp,rvel_warp,oden_warp,gden_warp,rden_warp = [None]*6
        # ovel_warpMID,gvel_warpMID,rvel_warpMID,oden_warpMID,gden_warpMID,rden_warpMID = [None]*6
    ovel_warp,gvel_warp,rvel_warp,oden_warp,gden_warp,rden_warp = [None]*6
    ovel_warpMID,gvel_warpMID,rvel_warpMID,oden_warpMID,gden_warpMID,rden_warpMID = [None]*6
    if our_path is not None: _our += Vmetrics(ovel, rvel, oden, rden, ovel_warp, oden_warp, ovel_warpMID, oden_warpMID, frameHull=uh, ignoreInflow=True, printname="our")
    if glo_path is not None: _glo += Vmetrics(gvel, rvel, gden, rden, gvel_warp, gden_warp, gvel_warpMID, gden_warpMID, frameHull=uh, ignoreInflow=True, printname="glo")
    _ref += Vmetrics(rvel, None, rden, None, rvel_warp, rden_warp, rvel_warpMID, rden_warpMID, frameHull=uh, ignoreInflow=True, printname="ref")

    if our_path is not None: our_list += _our
    if glo_path is not None: glo_list += _glo
    ref_list += _ref

    if our_path is not None:
        copyArrayToGridReal(target=our_grids[1], source=np.abs(rden - oden)*denratio)
        if True: save_fig(our_grids[1], ovel - rvel, out_path+"Diff_our_", t=fr)
    if glo_path is not None:
        copyArrayToGridReal(target=glo_grids[1], source=np.abs(rden - gden)*denratio)
        if True: save_fig(glo_grids[1], gvel - rvel, out_path+"Diff_glo_", t=fr)
    fr += 1

Vname = Vmetrics(nameonly=True)
Vname = ["Inflow_" +a for a in Vname[:2]] + Vname
metricN = len(Vname)
if our_path is not None:
    our_list = np.reshape(np.float32(our_list), [-1,metricN]) # 30,N
if glo_path is not None:
    glo_list = np.reshape(np.float32(glo_list), [-1,metricN]) # 30,N
ref_list = np.reshape(np.float32(ref_list), [-1,metricN]) # 30,N

pd_dict = {}
for i in range(metricN):
    if our_path is not None: pd_dict["our_%s"%Vname[i]] = pd.Series(our_list[:,i])
    if glo_path is not None: pd_dict["glo_%s"%Vname[i]] = pd.Series(glo_list[:,i])
    if i >= 2:
        pd_dict["ref_%s"%Vname[i]] = pd.Series(ref_list[:,i])

pd_dict["mean"] = pd.Series(["ref", "our", "glo"])
for i in range(metricN):
    pd_add = [ref_list[:,i].mean()]
    print("Avg %s means, ref"%Vname[i],ref_list[:,i].mean())
    if our_path is not None: 
        pd_add += [our_list[:,i].mean()]
        print("Avg %s means, our"%Vname[i],our_list[:,i].mean())
    if glo_path is not None: 
        pd_add += [glo_list[:,i].mean()]
        print("Avg %s means, glo"%Vname[i],glo_list[:,i].mean())
    pd_dict[Vname[i]] = pd.Series(pd_add)
    
pd.DataFrame(pd_dict).to_csv(os.path.join(out_path,"eval_l2.csv"), mode='w')


if False:
    from run_nerf_helpers import FFmpegTool
    n = 1 + sum(x is not None for x in [our_path, glo_path])
    myffmpeg = FFmpegTool(os.path.join(out_path, "eval_den.mp4"), row=n, col=2)
    myffmpeg.add_image(os.path.join(out_path, 'ref_den_%04d.ppm'), stt=0, fps=15)
    myffmpeg.add_image(os.path.join(out_path, 'ref_den_%04d.ppm'), stt=0, fps=15)
    if our_path is not None:
        myffmpeg.add_image(os.path.join(out_path, 'our_den_%04d.ppm'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'Diff_our_den_%04d.ppm'), stt=0, fps=15)
    if glo_path is not None:
        myffmpeg.add_image(os.path.join(out_path, 'glo_den_%04d.ppm'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'Diff_glo_den_%04d.ppm'), stt=0, fps=15)
    myffmpeg.join_cmd()
    
    text_off = 2
    myffmpeg.add_label("ref", 2, text_off, 24)
    if our_path is not None:
        text_off += 192
        myffmpeg.add_label("our", 2, text_off, 24)
    if glo_path is not None: 
        text_off += 192
        myffmpeg.add_label("glo", 2, text_off, 24)
    myffmpeg.export(overwrite=True)

    myffmpeg = FFmpegTool(os.path.join(out_path, "eval_vel.mp4"), row=n, col=3)
    myffmpeg.add_image(os.path.join(out_path, 'oriref_vel_%04d.png'), stt=0, fps=15)
    myffmpeg.add_image(os.path.join(out_path, 'ref_vel_%04d.png'), stt=0, fps=15)
    myffmpeg.add_image(os.path.join(out_path, 'ref_vel_%04d.png'), stt=0, fps=15)
    if our_path is not None:
        myffmpeg.add_image(os.path.join(out_path, 'oriour_vel_%04d.png'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'our_vel_%04d.png'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'Diff_our_vel_%04d.png'), stt=0, fps=15)
    if glo_path is not None:
        myffmpeg.add_image(os.path.join(out_path, 'origlo_vel_%04d.png'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'glo_vel_%04d.png'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'Diff_glo_vel_%04d.png'), stt=0, fps=15)
    myffmpeg.join_cmd()
    text_off = 2
    myffmpeg.add_label("ref", 402, text_off, 24)
    if our_path is not None:
        text_off += 192
        myffmpeg.add_label("our", 402, text_off, 24)
    if glo_path is not None: 
        text_off += 192
        myffmpeg.add_label("glo", 402, text_off, 24)
    myffmpeg.export(overwrite=True)

    myffmpeg = FFmpegTool(os.path.join(out_path, "eval_vor.mp4"), row=n, col=2)
    myffmpeg.add_image(os.path.join(out_path, 'ref_vort_vel_%04d.png'), stt=0, fps=15)
    myffmpeg.add_image(os.path.join(out_path, 'ref_vort_vel_%04d.png'), stt=0, fps=15)
    if our_path is not None:
        myffmpeg.add_image(os.path.join(out_path, 'our_vort_vel_%04d.png'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'Diff_our_vort_vel_%04d.png'), stt=0, fps=15)
    if glo_path is not None: 
        myffmpeg.add_image(os.path.join(out_path, 'glo_vort_vel_%04d.png'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'Diff_glo_vort_vel_%04d.png'), stt=0, fps=15)
    myffmpeg.join_cmd()
    text_off = 2
    myffmpeg.add_label("ref", 2, text_off, 24)
    if our_path is not None:
        text_off += 192
        myffmpeg.add_label("our", 2, text_off, 24)
    if glo_path is not None: 
        text_off += 192
        myffmpeg.add_label("glo", 2, text_off, 24)
    myffmpeg.export(overwrite=True)

    myffmpeg = FFmpegTool(os.path.join(out_path, "warp.mp4"), row=n, col=2)
    myffmpeg.add_image(os.path.join(out_path, 'warp_ref_den_%04d.ppm'), stt=0, fps=15)
    myffmpeg.add_image(os.path.join(out_path, 'warp_ref_vel_%04d.png'), stt=0, fps=15)
    if our_path is not None:
        myffmpeg.add_image(os.path.join(out_path, 'warp_our_den_%04d.ppm'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'warp_our_vel_%04d.png'), stt=0, fps=15)
    if glo_path is not None: 
        myffmpeg.add_image(os.path.join(out_path, 'warp_glo_den_%04d.ppm'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'warp_glo_vel_%04d.png'), stt=0, fps=15)
    myffmpeg.join_cmd()
    text_off = 2
    myffmpeg.add_label("ref", 402, text_off, 24)
    if our_path is not None:
        text_off += 192
        myffmpeg.add_label("our", 402, text_off, 24)
    if glo_path is not None: 
        text_off += 192
        myffmpeg.add_label("glo", 402, text_off, 24)
    myffmpeg.export(overwrite=True)

    myffmpeg = FFmpegTool(os.path.join(out_path, "warpMID.mp4"), row=n, col=2)
    myffmpeg.add_image(os.path.join(out_path, 'warpMID_ref_den_%04d.ppm'), stt=0, fps=15)
    myffmpeg.add_image(os.path.join(out_path, 'warpMID_ref_vel_%04d.png'), stt=0, fps=15)
    if our_path is not None:
        myffmpeg.add_image(os.path.join(out_path, 'warpMID_our_den_%04d.ppm'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'warpMID_our_vel_%04d.png'), stt=0, fps=15)
    if glo_path is not None: 
        myffmpeg.add_image(os.path.join(out_path, 'warpMID_glo_den_%04d.ppm'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'warpMID_glo_vel_%04d.png'), stt=0, fps=15)
    myffmpeg.join_cmd()
    text_off = 2
    myffmpeg.add_label("ref", 402, text_off, 24)
    if our_path is not None:
        text_off += 192
        myffmpeg.add_label("our", 402, text_off, 24)
    if glo_path is not None: 
        text_off += 192
        myffmpeg.add_label("glo", 402, text_off, 24)
    myffmpeg.export(overwrite=True)

    if True:
        ppm_list = os.listdir(out_path)
        ppm_list = [os.remove(os.path.join(out_path, _)) for _ in ppm_list if _.endswith(".ppm") or _.endswith(".png")] 