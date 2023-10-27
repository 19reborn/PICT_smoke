

import numpy as np
import sys, os
import imageio
import cv2 as cv

import subprocess

#####################################################################
# custom Logger to write Log to file
class Logger(object):
    def __init__(self, summary_dir, silent=False, fname="logfile.txt"):
        self.terminal = sys.stdout
        self.silent = silent
        self.log = open(os.path.join(summary_dir, fname), "a") 
        cmdline = " ".join(sys.argv)+"\n"
        self.log.write(cmdline) 
    def write(self, message):
        if not self.silent: 
            self.terminal.write(message)
        self.log.write(message) 
    def flush(self):
        self.terminal.flush()
        self.log.flush()

class FFmpegTool(object):
    def init_set(self, output_file, row=1, col=1, ffmpeg_path = "ffmpeg"):
        self.target_file = output_file
        self.ffmpeg_cmd = ffmpeg_path
        self.cmd = [self.ffmpeg_cmd, "-nostdin"]
        self.inputs_n = 0
        self.row = row
        self.col = col
        self.filter_cmd = ""
        self.textID = 0

    def __init__(self, output_file="", row=1, col=1, ffmpeg_path = "ffmpeg"):
        self.init_set(output_file, row, col, ffmpeg_path)

    def add_image(self, image_path, fps=60, stt=0):
        self.cmd += ["-f", "image2", "-start_number", "%d"%stt, "-framerate", "%d"%fps,
             "-i", image_path]
        self.inputs_n += 1

    def add_label(self, label, x, y, fz):
        textstr = "[v%d]"%self.textID
        cmdstr = textstr + ";" + textstr + "drawtext=text=\"" + label + "\":fontsize=%d"%fz \
            + ":box=1:boxcolor=black@0.5:boxborderw=4:x=(%d):y=(%d)"%(x,y) \
            + ":fontfile=/usr/share/fonts/truetype/lato/Lato-Regular.ttf:fontcolor=white"
        #   + ":fontfile=OpenSans.ttf:fontcolor=white"
        self.textID += 1
        self.filter_cmd += cmdstr

    
    def join_cmd(self, filter_cmd = ''):
        if filter_cmd != '':
            self.filter_cmd = filter_cmd
            return self.filter_cmd

        if self.inputs_n != (self.row * self.col ):
            print("Error")
            return ""

        vstr = []
        if self.row == 1 or self.col == 1:
            vstr += ["[%d:v]"%i for i in range(max(self.row, self.col))]
            if self.row == 1:
                vstr += ["hstack=inputs=%d"%self.col]
            else:
                vstr += ["vstack=inputs=%d"%self.row]
            
            self.filter_cmd = "".join(vstr)
        else:
            for j in range(self.row):
                c = j * self.col
                vstr += ["[%d:v]"%i for i in range(c,c+self.col)]
                vstr += ["hstack=inputs=%d"%self.col,"[c%d];"%j]
            vstr +=  ["[c%d]"%i for i in range(self.row)]
            vstr += ["vstack=inputs=%d"%self.row]
            self.filter_cmd = "".join(vstr)

        return self.filter_cmd


    def export(self, notrun = False, overwrite=False):
        if overwrite:
            self.cmd = [self.cmd[0], "-y"] + self.cmd[1:]
        if self.filter_cmd != "":
            self.cmd += ["-filter_complex", "\"%s\""%self.filter_cmd,]
        self.cmd += ["-vcodec", "libx264", "-crf", "21", "-pix_fmt", "yuv420p",
            self.target_file]
        cmd1 = " ".join(self.cmd)
        if notrun: return cmd1
        print(cmd1)
        subprocess.call(cmd1, shell=True)
        return cmd1
    
#####################################################################
# Visualization Tools

def velLegendHSV(hsvin, is3D, lw=-1, constV=255):
    # hsvin: (b), h, w, 3
    # always overwrite hsvin borders [lw], please pad hsvin before hand
    # or fill whole hsvin (lw < 0)
    ih, iw = hsvin.shape[-3:-1]
    if lw<=0: # fill whole
        a_list, b_list = [range(ih)], [range(iw)]
    else: # fill border
        a_list = [range(ih),  range(lw), range(ih), range(ih-lw, ih)]
        b_list = [range(lw),  range(iw), range(iw-lw, iw), range(iw)]
    for a,b in zip(a_list, b_list):
        for _fty in a:
            for _ftx in b:
                fty = _fty - ih//2
                ftx = _ftx - iw//2
                ftang = np.arctan2(fty, ftx) + np.pi
                ftang = ftang*(180/np.pi/2)
                # print("ftang,min,max,mean", ftang.min(), ftang.max(), ftang.mean())
                # ftang,min,max,mean 0.7031249999999849 180.0 90.3515625
                hsvin[...,_fty,_ftx,0] = np.expand_dims(ftang, axis=-1) # 0-360 
                # hsvin[...,_fty,_ftx,0] = ftang
                hsvin[...,_fty,_ftx,2] = constV
                if (not is3D) or (lw == 1):
                    hsvin[...,_fty,_ftx,1] = 255
                else:
                    thetaY1 = 1.0 - ((ih//2) - abs(fty)) / float( lw if (lw > 1) else (ih//2) )
                    thetaY2 = 1.0 - ((iw//2) - abs(ftx)) / float( lw if (lw > 1) else (iw//2) )
                    fthetaY = max(thetaY1, thetaY2) * (0.5*np.pi)
                    ftxY, ftyY = np.cos(fthetaY), np.sin(fthetaY)
                    fangY = np.arctan2(ftyY, ftxY)
                    fangY = fangY*(240/np.pi*2) # 240 - 0
                    hsvin[...,_fty,_ftx,1] = 255 - fangY
                    # print("fangY,min,max,mean", fangY.min(), fangY.max(), fangY.mean())
    # finished velLegendHSV.

def cubecenter(cube, axis, half = 0):
    # cube: (b,)h,h,h,c
    # axis: 1 (z), 2 (y), 3 (x)
    reduce_axis = [a for a in [1,2,3] if a != axis]
    pack = np.mean(cube, axis=tuple(reduce_axis)) # (b,)h,c
    pack = np.sqrt(np.sum( np.square(pack), axis=-1 ) + 1e-6) # (b,)h

    length = cube.shape[axis-5] # h
    weights = np.arange(0.5/length,1.0,1.0/length)
    if half == 1: # first half
        weights = np.where( weights < 0.5, weights, np.zeros_like(weights))
        pack = np.where( weights < 0.5, pack, np.zeros_like(pack))
    elif half == 2: # second half
        weights = np.where( weights > 0.5, weights, np.zeros_like(weights))
        pack = np.where( weights > 0.5, pack, np.zeros_like(pack))

    weighted = pack * weights # (b,)h
    weiAxis = np.sum(weighted, axis=-1) / np.sum(pack, axis=-1) * length # (b,)
    
    return weiAxis.astype(np.int32) # a ceiling is included

def vel2hsv(velin, is3D, logv, scale=None): # 2D
    fx, fy = velin[...,0], velin[...,1]
    ori_shape = list(velin.shape[:-1]) + [3]
    if is3D: 
        fz = velin[...,2]
        ang = np.arctan2(fz, fx) + np.pi # angXZ
        zxlen2 = fx*fx+fz*fz
        angY = np.arctan2(np.abs(fy), np.sqrt(zxlen2))
        v = np.sqrt(zxlen2+fy*fy)
    else:
        v = np.sqrt(fx*fx+fy*fy)
        ang = np.arctan2(fy, fx) + np.pi
    
    if logv:
        v = np.log10(v+1)
    
    hsv = np.zeros(ori_shape, np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    if is3D:
        hsv[...,1] = 255 - angY*(240/np.pi*2)  
    else:
        hsv[...,1] = 255
    if scale is not None:
        hsv[...,2] = np.minimum(v*scale, 255)
    else:
        hsv[...,2] = v/max(v.max(),1e-6) * 255.0
    return hsv


def vel_uv2hsv(vel, scale = 160, is3D=False, logv=False, mix=False):
    # vel: a np.float32 array, in shape of (?=b,) d,h,w,3 for 3D and (?=b,)h,w, 2 or 3 for 2D
    # scale: scale content to 0~255, something between 100-255 is usually good. 
    #        content will be normalized if scale is None
    # logv: visualize value with log
    # mix: use more slices to get a volumetric visualization if True, which is slow

    ori_shape = list(vel.shape[:-1]) + [3] # (?=b,) d,h,w,3
    if is3D: 
        new_range = list( range( len(ori_shape) ) )
        z_new_range = new_range[:]
        z_new_range[-4] = new_range[-3]
        z_new_range[-3] = new_range[-4]
        # print(z_new_range)
        YZXvel = np.transpose(vel, z_new_range)
        
        _xm,_ym,_zm = (ori_shape[-2]-1)//2, (ori_shape[-3]-1)//2, (ori_shape[-4]-1)//2
        
        if mix:
            _xlist = [cubecenter(vel, 3, 1),_xm,cubecenter(vel, 3, 2)]
            _ylist = [cubecenter(vel, 2, 1),_ym,cubecenter(vel, 2, 2)]
            _zlist = [cubecenter(vel, 1, 1),_zm,cubecenter(vel, 1, 2)]
        else:
            _xlist, _ylist, _zlist = [_xm], [_ym], [_zm]

        hsv = []
        for _x, _y, _z in zip (_xlist, _ylist, _zlist):
            # print(_x, _y, _z)
            _x, _y, _z = np.clip([_x, _y, _z], 0, ori_shape[-2:-5:-1])
            _yz = YZXvel[...,_x,:]
            _yz = np.stack( [_yz[...,2],_yz[...,0],_yz[...,1]], axis=-1)
            _yx = YZXvel[...,_z,:,:]
            _yx = np.stack( [_yx[...,0],_yx[...,2],_yx[...,1]], axis=-1)
            _zx = YZXvel[...,_y,:,:,:]
            _zx = np.stack( [_zx[...,0],_zx[...,1],_zx[...,2]], axis=-1)
            # print(_yx.shape, _yz.shape, _zx.shape)

            # in case resolution is not a cube, (res,res,res)
            _yxz = np.concatenate( [ #yz, yx, zx
                _yx, _yz ], axis = -2) # (?=b,),h,w+zdim,3
            
            if ori_shape[-3] < ori_shape[-4]:
                pad_shape = list(_yxz.shape) #(?=b,),h,w+zdim,3
                pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
                _pad = np.zeros(pad_shape, dtype=np.float32)
                _yxz = np.concatenate( [_yxz,_pad], axis = -3)
            elif ori_shape[-3] > ori_shape[-4]:
                pad_shape = list(_zx.shape) #(?=b,),h,w+zdim,3
                pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

                _zx = np.concatenate( 
                    [_zx,np.zeros(pad_shape, dtype=np.float32)], axis = -3)
            
            midVel = np.concatenate( [ #yz, yx, zx
                _yxz, _zx
            ], axis = -2) # (?=b,),h,w*3,3
            hsv += [vel2hsv(midVel, True, logv, scale)]
        # remove depth dim, increase with zyx slices
        ori_shape[-3] = 3 * ori_shape[-2]
        ori_shape[-2] = ori_shape[-1]
        ori_shape = ori_shape[:-1]
    else:
        hsv = [vel2hsv(vel, False, logv, scale)]

    bgr = []
    for _hsv in hsv:
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape([-1]+ori_shape[-2:])
        if is3D:
            velLegendHSV(_hsv, is3D, lw=max(1,min(6,int(0.025*ori_shape[-2]))), constV=255)
        _hsv = cv.cvtColor(_hsv, cv.COLOR_HSV2BGR)
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape(ori_shape)
        bgr += [_hsv]
    if len(bgr) == 1:
        bgr = bgr[0]
    else:
        bgr = bgr[0] * 0.2 + bgr[1] * 0.6 + bgr[2] * 0.2
    return bgr.astype(np.uint8)[::-1] # flip Y


def den_scalar2rgb(den, scale=160, is3D=False, logv=False, mix=True):
    # den: a np.float32 array, in shape of (?=b,) d,h,w,1 for 3D and (?=b,)h,w,1 for 2D
    # scale: scale content to 0~255, something between 100-255 is usually good. 
    #        content will be normalized if scale is None
    # logv: visualize value with log
    # mix: use averaged value as a volumetric visualization if True, else show middle slice

    ori_shape = list(den.shape)
    if ori_shape[-1] != 1:
        ori_shape.append(1)
        den = np.reshape(den, ori_shape)

    if is3D: 
        new_range = list( range( len(ori_shape) ) )
        z_new_range = new_range[:]
        z_new_range[-4] = new_range[-3]
        z_new_range[-3] = new_range[-4]
        # print(z_new_range)
        YZXden = np.transpose(den, z_new_range)
                
        if not mix:
            _yz = YZXden[...,(ori_shape[-2]-1)//2,:]
            _yx = YZXden[...,(ori_shape[-4]-1)//2,:,:]
            _zx = YZXden[...,(ori_shape[-3]-1)//2,:,:,:]
        else:
            _yz = np.average(YZXden, axis=-2)
            _yx = np.average(YZXden, axis=-3)
            _zx = np.average(YZXden, axis=-4)
            # print(_yx.shape, _yz.shape, _zx.shape)

        # in case resolution is not a cube, (res,res,res)
        _yxz = np.concatenate( [ #yz, yx, zx
            _yx, _yz ], axis = -2) # (?=b,),h,w+zdim,1
        
        if ori_shape[-3] < ori_shape[-4]:
            pad_shape = list(_yxz.shape) #(?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
            _pad = np.zeros(pad_shape, dtype=np.float32)
            _yxz = np.concatenate( [_yxz,_pad], axis = -3)
        elif ori_shape[-3] > ori_shape[-4]:
            pad_shape = list(_zx.shape) #(?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

            _zx = np.concatenate( 
                [_zx,np.zeros(pad_shape, dtype=np.float32)], axis = -3)
        
        midDen = np.concatenate( [ #yz, yx, zx
            _yxz, _zx
        ], axis = -2) # (?=b,),h,w*3,1
    else:
        midDen = den

    if logv:
        midDen = np.log10(midDen+1)
    if scale is None:
        midDen = midDen / max(midDen.max(),1e-6) * 255.0
    else:
        midDen = midDen * scale
    grey = np.clip(midDen, 0, 255)

    return grey.astype(np.uint8)[::-1] # flip y


#####################################################################
# Physics Tools
def jacobian3D_np(x):
    # x, (b,)d,h,w,ch
    # return jacobian and curl

    if len(x.shape) < 5:
        x = np.expand_dims(x, axis=0)
    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
    dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

    dudy = np.concatenate((dudy, np.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy = np.concatenate((dwdy, np.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

    dudz = np.concatenate((dudz, np.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
    dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
    dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = np.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
    c = np.stack([u,v,w], axis=-1)
    
    return j, c


def vor_rgb(vor, scale = 640 ):
    rgb = np.zeros(list(vor.shape[:-1]) + [3], np.uint8)
    rgb[...,0] = np.clip(-vor[...,0]*scale, 0, 255)
    rgb[...,1] = np.clip( vor[...,0]*scale, 0, 255)
    # rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)

def jacobian2D_np(x):
    dudx = x[:,:,1:,0] - x[:,:,:-1,0]
    dudy = x[:,1:,:,0] - x[:,:-1,:,0]
    dvdx = x[:,:,1:,1] - x[:,:,:-1,1]
    dvdy = x[:,1:,:,1] - x[:,:-1,:,1]
    
    dudx = np.concatenate([dudx,np.expand_dims(dudx[:,:,-1], axis=2)], axis=2)
    dvdx = np.concatenate([dvdx,np.expand_dims(dvdx[:,:,-1], axis=2)], axis=2)
    dudy = np.concatenate([dudy,np.expand_dims(dudy[:,-1,:], axis=1)], axis=1)
    dvdy = np.concatenate([dvdy,np.expand_dims(dvdy[:,-1,:], axis=1)], axis=1)

    j = np.stack([dudx,dudy,dvdx,dvdy], axis=-1)
    w = np.expand_dims(dvdx - dudy, axis=-1) # vorticity (for visualization)
    # print( w.max(), w.min(), w.mean())
    return j, w

def divergence3D_np(x): # data_format='NDHWC'
    if len(x.shape) < 5:
        x = np.expand_dims(x, axis=0)
    dudx = x[:,:-1,:-1,1:,0] - x[:,:-1,:-1,:-1,0] # left, right
    dvdy = x[:,:-1,1:,:-1,1] - x[:,:-1,:-1,:-1,1] # top, down
    dwdz = x[:,1:,:-1,:-1,2] - x[:,:-1,:-1,:-1,2] # front, bottom
    div = np.expand_dims(dudx + dvdy + dwdz, axis=-1)
    return div



