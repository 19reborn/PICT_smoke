# /cluster/project/tang/yiming/project/mantaflow_nogui/build/manta /cluster/project/tang/yiming/project/pinf_clean/tools/eval/visual_eval_car_eular.py


import os, sys, inspect
import numpy as np
from manta import *
import imageio


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from _helpers import vel_uv2hsv, jacobian3D_np,divergence3D_np,jacobian2D_np,vor_rgb
from _helpers import FFmpegTool


ref_path = '/cluster/project/tang/yiming/dataset/pinf_gt/eval_data/gt_data/Car/np/'

# our_path = '/cluster/project/tang/yiming/project/pinf_clean/log/Car/1129_v5_larger_fine_density/volumeout_400001/'
# our_path = '/cluster/project/tang/yiming/project/pinf_clean/log/Car/1129_v3_larger_omega/volumeout_400001/'
# out_path = '/cluster/project/tang/yiming/project/pinf_clean/log/evaluate/' + '/car/1208_v1_old_data_best_full/'
# out_path = '/cluster/project/tang/yiming/project/pinf_clean/log/evaluate/' + '/car/1208_v1_old_data_best_full_use_gt_mask/'
our_path = '/cluster/project/tang/yiming/project/pinf_clean/log/Car/1208_v6_omega60_velocity/volumeout_370001/'
out_path = '/cluster/project/tang/yiming/project/pinf_clean/log/evaluate/' + '/car/1209_v1_new_data_best_full_use_gt_mask/'
glo_path = None
hull_path = None

os.makedirs(out_path, exist_ok = True)
    
den_file, vel_file = "xl_den_%04d.f16.npz", "xl_vel_%04d.f16.npz"

ref_gs = vec3(768,192,304) # 178
our_gs = vec3(384,96,154)

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

def saveVel(Vin, path, scale=368):
    # imageio.imwrite( path, vel_uv2hsv(Vin, scale=scale, is3D=True, logv=False)[::-1])
    imageio.imwrite( path, vel_uv2hsv(Vin, scale=scale, is3D=True, logv=False))



def np_l2(np_array): return np.sum(np.square(np_array),axis=-1)

ref_s, oref_grids = setSolver(ref_gs, 'ref')
our_s, our_grids = setSolver(our_gs, 'our')

oref_grids[1].load("/cluster/project/tang/yiming/dataset/pinf_gt/eval_data/gt_data/Car/manta" + "/car_fine.uni")
flagArray = np.zeros([int(ref_gs.z), int(ref_gs.y), int(ref_gs.x), 1], dtype = np.float32)
copyGridToArrayReal(target=flagArray, source=oref_grids[1])
print(flagArray.min(), flagArray.mean(), flagArray.max())
# import pdb;pdb.set_trace()
if ref_gs.x != our_gs.x: # same res for evaluation
    newr_vel  = our_s.create(MACGrid)
    newr_den  = our_s.create(RealGrid)
    ref_grids = [our_grids[0], newr_den, newr_vel]

if (ref_gs.x != our_gs.x) and (our_path is not None):
    copyArrayToGridReal(target=oref_grids[1], source=flagArray)
    interpolateGrid( target=ref_grids[1], source=oref_grids[1] )
    flagArray = np.zeros([int(our_gs.z), int(our_gs.y), int(our_gs.x), 1], dtype = np.float32)
    copyGridToArrayReal(target=flagArray, source=ref_grids[1])
# flagArray = np.ones_like(flagArray, dtype = np.float32)
bbox_min = 0.05,-0.05,0.05
bbox_max = 0.95,0.8,0.95
flagArray[:int(0.05*154),...] = 0.0
flagArray[int(0.95*154):,...] = 0.0
flagArray[:,int(0.8*96):,...] = 0.0
flagArray[:,:,:int(0.05*384),...] = 0.0
flagArray[:,:,int(0.9*384):,...] = 0.0
hull_flag = np.float32(flagArray > 1e-4)
# hull_flag = np.float32(flagArray)
flagArray[:int(0.05*154)+1,...] = 0.0
flagArray[int(0.95*154)-1:,...] = 0.0
flagArray[:,int(0.8*96)-1:,...] = 0.0
flagArray[:,:,:int(0.05*384)+1,...] = 0.0
flagArray[:,:,int(0.9*384)-1:,...] = 0.0
hull_flag1 = np.float32(flagArray > 1)
# hull_flag1 = np.float32(flagArray)
normDen = True
hullmask = True
denratio = 1.2
use_gt_mask = True

flagArray = np.ones([int(ref_gs.z), int(ref_gs.y), int(ref_gs.x), 1], dtype = np.float32)
# import pdb;pdb.set_trace()
if ref_gs.x != our_gs.x: # same res for evaluation
    newr_vel  = our_s.create(MACGrid)
    newr_den  = our_s.create(RealGrid)
    ref_grids = [our_grids[0], newr_den, newr_vel]

if (ref_gs.x != our_gs.x) and (our_path is not None):
    copyArrayToGridReal(target=oref_grids[1], source=flagArray)
    interpolateGrid( target=ref_grids[1], source=oref_grids[1] )
    flagArray = np.zeros([int(our_gs.z), int(our_gs.y), int(our_gs.x), 1], dtype = np.float32)
    copyGridToArrayReal(target=flagArray, source=ref_grids[1])
# flagArray = np.ones_like(flagArray, dtype = np.float32)
bbox_min = 0.05,-0.05,0.05
bbox_max = 0.95,0.8,0.95
flagArray[:int(0.05*154),...] = 0.0
flagArray[int(0.95*154):,...] = 0.0
flagArray[:,int(0.8*96):,...] = 0.0
flagArray[:,:,:int(0.05*384),...] = 0.0
flagArray[:,:,int(0.9*384):,...] = 0.0
hull_flag_only_bbox = np.float32(flagArray > 1e-4)
# hull_flag = np.float32(flagArray)
flagArray[:int(0.05*154)+1,...] = 0.0
flagArray[int(0.95*154)-1:,...] = 0.0
flagArray[:,int(0.8*96)-1:,...] = 0.0
flagArray[:,:,:int(0.05*384)+1,...] = 0.0
flagArray[:,:,int(0.9*384)-1:,...] = 0.0
hull_flag1_only_bbox = np.float32(flagArray > 1)

if use_gt_mask:
    hull_flag_only_bbox = hull_flag
    hull_flag1_only_bbox = hull_flag1
    
def save_fig(den, vel, image_dir, den_name='den_%04d.ppm', vel_name='vel_%04d.png', t=0, is2D=False, vN=1.0):
    # os.makedirs(image_dir, exist_ok = True) 
    if is2D:
        if den is not None:
            projectPpmFull( den, image_dir+ den_name% (t), 0, 1.0 )
        if vel is not None:
            saveVel(np.squeeze(vel), image_dir+ vel_name% (t))
            _, NETw = jacobian2D_np(vel)
            imageio.imwrite(image_dir+'vor_%04d.png' % (t), vor_rgb(np.squeeze(NETw))[::-1])
    else:
        if den is not None:
            projectPpmFull( den, image_dir+den_name % (t), 0, 4.0 )
        if vel is not None:
            saveVel(np.squeeze(vel), image_dir+vel_name % (t))
            _, NETw = jacobian3D_np(vel)
            saveVel(np.squeeze(NETw), image_dir+"vort_"+vel_name % (t), scale=720*vN)

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
    divV = np.float32(0) if ovel is None else np.abs(np.squeeze(divergence3D_np(ovel)))
    l2D = np.float32(0) if any(x is None for x in [oden, rden]) else np_l2(oden - rden)

    l2Vwarp = np.float32(0) if oVwarp is None else np_l2(oVwarp) 
    l2Dwarp = np.float32(0) if oDwarp is None else np_l2(oDwarp)

    l2VwarpMid = np.float32(0) if oVwarpMid is None else np_l2(oVwarpMid) 
    l2DwarpMid = np.float32(0) if oDwarpMid is None else np_l2(oDwarpMid)

    metrics = [l2D, l2V, divV, l2Vwarp, l2Dwarp, l2VwarpMid, l2DwarpMid]
    if ignoreInflow: # ignore inflow part
        metrics = [(np.float32(0) if len(a.shape) == 0 else a[:,25:,:]) for a in metrics]
        if frameHull is not None: frameHull = np.squeeze(frameHull)[:,25:,:]
    hullmean = 1.0
    if frameHull is not None: hullmean = max(frameHull.mean(), 1e-6)
    if printname!= "":
        print(" ".join([printname]+[a+str(b.shape)+":"+str(b.mean()/hullmean) for a,b in zip(names, metrics)]))
    return [m.mean()/hullmean for m in metrics]


our_list, glo_list, ref_list =[], [], []
fr = 0
testWarp = True
frame_num = 0

for framei in range(0,140, 1): # [100]
    print(framei)
    frame_num += 1
    if True:
        # rden, rvel = readData(ref_path, "xl_den_%04d.f16.npz"%(163+framei), "xl_vel_%04d.f16.npz"%(163+framei), oref_grids)
        rden, rvel = readData(ref_path, "xl_den_%04d.f16.npz"%(165+framei), "xl_vel_%04d.f16.npz"%(165+framei), oref_grids)
        rden = np.reshape(rden, [int(ref_gs.z),int(ref_gs.y),int(ref_gs.x), -1])
        rvel = np.reshape(rvel, [int(ref_gs.z),int(ref_gs.y),int(ref_gs.x), -1])

        
        if our_path is not None:
            oden, ovel = readData(our_path, "d_%04d.npz"%framei, "v_%04d.npz"%framei, our_grids)
            oden = np.reshape(oden, [int(our_gs.z),int(our_gs.y),int(our_gs.x), -1])
            ovel = np.reshape(ovel, [int(our_gs.z),int(our_gs.y),int(our_gs.x), -1])

        
        if (ref_gs.x != our_gs.x) and (our_path is not None):
            copyArrayToGridReal(target=oref_grids[1], source=rden)
            copyArrayToGridMAC(target=oref_grids[2], source=rvel)
            interpolateGrid( target=ref_grids[1], source=oref_grids[1] )
            interpolateMACGrid(target=ref_grids[2], source=oref_grids[2] )
            ref_grids[2].multConst(our_gs/ref_gs)
            rvel = np.copy(ovel)
            rden = np.copy(oden)
            copyGridToArrayReal(target=rden, source=ref_grids[1])
            copyGridToArrayMAC(target=rvel, source=ref_grids[2])
            rden = rden * hull_flag
            rvel = rvel * hull_flag
            
            # copyArrayToGridReal(target=ref_grids[1], source=rden)
        if hullmask:
            if True: save_fig(None, rvel, out_path+"oriref_", t=fr)
            if True and (our_path is not None): save_fig(None, ovel, out_path+"oriour_", t=fr, vN=1.0)

            # hull_data = np.float32(rden > 1e-4)
            # hull_data = np.ones_like(rden)
            # hull_data = hull_flag
            hull_data = np.float32(rden > 1e-4)


            if our_path is not None:
                if use_gt_mask:
                    oden *= hull_data
                    ovel *= hull_data
                    
                                   
                    rden *= hull_data
                    rvel *= hull_data     
                    
                else:
                    oden *= hull_flag_only_bbox
                    ovel *= hull_flag_only_bbox
                    
                    rden *= hull_flag
                    rvel *= hull_flag
            
        
        # print("den means, ref",rden.mean(),"our", oden.mean(), "glo", gden.mean())
        if normDen and (our_path is not None):
            odscale = rden.mean()/oden.mean() 
            print(rden.mean(), oden.mean(), odscale)
            oden = oden * odscale

        printVelRange(rvel, "ref")
        if our_path is not None: printVelRange(ovel, "our")
        print("ref den range", rden.min(), rden.mean(), rden.max())
        if our_path is not None: print("our den range", oden.min(), oden.mean(), oden.max())

        copyArrayToGridReal(target=ref_grids[1], source=rden)
        if our_path is not None: copyArrayToGridReal(target=our_grids[1], source=oden)
        if True: save_fig(ref_grids[1], rvel, out_path+"ref_", t=fr)
        if our_path is not None: save_fig(our_grids[1], ovel, out_path+"our_", t=fr, vN=1.0)

        # with inflow:
        
        if False:
            uh = hull_data # None 
            if our_path is not None: _our = Vmetrics(ovel, rvel, oden, rden, frameHull=uh, printname="our")[:2]
            _ref = Vmetrics(rvel, None, rden, None, frameHull=uh, printname="ref")[:2]
            # w.o. inflow:
            if framei==60:
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
    fr += 1
    

if True:
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
    # text_off = 2
    # myffmpeg.add_label("ref", 2, text_off, 24)
    # if our_path is not None:
    #     text_off += 192
    #     myffmpeg.add_label("our", 2, text_off, 24)
    # if glo_path is not None: 
    #     text_off += 192
    #     myffmpeg.add_label("glo", 2, text_off, 24)
    myffmpeg.export(overwrite=True)

    myffmpeg = FFmpegTool(os.path.join(out_path, "eval_vel.mp4"), row=n, col=3 if hullmask else 2) # if use hull_data. col should be 3
    if hullmask:
        myffmpeg.add_image(os.path.join(out_path, 'oriref_vel_%04d.png'), stt=0, fps=15)
    myffmpeg.add_image(os.path.join(out_path, 'ref_vel_%04d.png'), stt=0, fps=15)
    myffmpeg.add_image(os.path.join(out_path, 'ref_vel_%04d.png'), stt=0, fps=15)
    if our_path is not None:
        if hullmask:
            myffmpeg.add_image(os.path.join(out_path, 'oriour_vel_%04d.png'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'our_vel_%04d.png'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'Diff_our_vel_%04d.png'), stt=0, fps=15)
    if glo_path is not None:
        if hullmask:
            myffmpeg.add_image(os.path.join(out_path, 'origlo_vel_%04d.png'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'glo_vel_%04d.png'), stt=0, fps=15)
        myffmpeg.add_image(os.path.join(out_path, 'Diff_glo_vel_%04d.png'), stt=0, fps=15)
    myffmpeg.join_cmd()
    
    # text_off = 2
    # myffmpeg.add_label("ref", 402, text_off, 24)
    # if our_path is not None:
    #     text_off += 192
    #     myffmpeg.add_label("our", 402, text_off, 24)
    # if glo_path is not None: 
    #     text_off += 192
    #     myffmpeg.add_label("glo", 402, text_off, 24)
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
    # text_off = 2
    # myffmpeg.add_label("ref", 2, text_off, 24)
    # if our_path is not None:
    #     text_off += 192
    #     myffmpeg.add_label("our", 2, text_off, 24)
    # if glo_path is not None: 
    #     text_off += 192
    #     myffmpeg.add_label("glo", 2, text_off, 24)
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
    # text_off = 2
    # myffmpeg.add_label("ref", 402, text_off, 24)
    # if our_path is not None:
    #     text_off += 192
    #     myffmpeg.add_label("our", 402, text_off, 24)
    # if glo_path is not None: 
    #     text_off += 192
    #     myffmpeg.add_label("glo", 402, text_off, 24)
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
    # text_off = 2
    # myffmpeg.add_label("ref", 402, text_off, 24)
    # if our_path is not None:
    #     text_off += 192
    #     myffmpeg.add_label("our", 402, text_off, 24)
    # if glo_path is not None: 
    #     text_off += 192
    #     myffmpeg.add_label("glo", 402, text_off, 24)
    myffmpeg.export(overwrite=True)

if True:
# if False:
    ppm_list = os.listdir(out_path)
    
    # save useful images for comparison
    density_output_dir = os.path.join(out_path, "density")
    os.makedirs(density_output_dir, exist_ok=True)
    vel_output_dir = os.path.join(out_path, "velocity")
    os.makedirs(vel_output_dir, exist_ok=True)
    vor_output_dir = os.path.join(out_path, "vorticity")
    os.makedirs(vor_output_dir, exist_ok=True)
    for frame_id in range(frame_num):
        ref_den_path = os.path.join(out_path, "ref_den_%04d.ppm"%frame_id)
        imageio.imwrite(os.path.join(density_output_dir, "ref_den_%04d.png"%frame_id), imageio.imread(ref_den_path))
        
        our_den_path = os.path.join(out_path, "our_den_%04d.ppm"%frame_id)
        imageio.imwrite(os.path.join(density_output_dir, "our_den_%04d.png"%frame_id), imageio.imread(our_den_path))
        
        comparison_density = np.concatenate((imageio.imread(ref_den_path), imageio.imread(our_den_path)), axis=0)
        imageio.imwrite(os.path.join(density_output_dir, "comparison_den_%04d.png"%frame_id), comparison_density)
        
        ref_vel_path = os.path.join(out_path, "ref_vel_%04d.png"%frame_id)
        imageio.imwrite(os.path.join(vel_output_dir, "ref_vel_%04d.png"%frame_id), imageio.imread(ref_vel_path))
        
        our_vel_path = os.path.join(out_path, "our_vel_%04d.png"%frame_id)
        imageio.imwrite(os.path.join(vel_output_dir, "our_vel_%04d.png"%frame_id), imageio.imread(our_vel_path))
        
        comparison_velocity = np.concatenate((imageio.imread(ref_vel_path), imageio.imread(our_vel_path)), axis=0)
        imageio.imwrite(os.path.join(vel_output_dir, "comparison_vel_%04d.png"%frame_id), comparison_velocity)
        
        
        ref_vort_path = os.path.join(out_path, "ref_vort_vel_%04d.png"%frame_id)
        imageio.imwrite(os.path.join(vor_output_dir, "ref_vort_vel_%04d.png"%frame_id), imageio.imread(ref_vort_path))
        
        our_vort_path = os.path.join(out_path, "our_vort_vel_%04d.png"%frame_id)
        imageio.imwrite(os.path.join(vor_output_dir, "our_vort_vel_%04d.png"%frame_id), imageio.imread(our_vort_path))
        
        comparison_vorticity = np.concatenate((imageio.imread(ref_vort_path), imageio.imread(our_vort_path)), axis=0)
        imageio.imwrite(os.path.join(vor_output_dir, "comparison_vort_%04d.png"%frame_id), comparison_vorticity)
        
        
    ppm_list = [os.remove(os.path.join(out_path, _)) for _ in ppm_list if _.endswith(".ppm") or _.endswith(".png")] 