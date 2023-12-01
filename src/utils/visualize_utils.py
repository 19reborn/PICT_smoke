
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import pandas as pd

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

    grey = grey.astype(np.uint8)[::-1] # flip y

    if grey.shape[-1] == 1:
        grey = np.repeat(grey, 3, -1)

    return grey



def draw_mapping(path, data):

    plt.figure()

    N = data.shape[0]
    for i in range(N):
        x = data[i, :, 0]
        y = data[i, :, 1]
        
        # plt.plot(x, y, marker='o', linestyle='-')
        # plt.plot(x, y, linestyle='-')
        plt.plot(x, y, linestyle='--', linewidth=0.75)
        # plt.plot(x, y, linestyle='--', linewidth=0.75, marker='o')
        # plt.plot(x, y, linestyle=':')
        # plt.plot(x, y, linestyle='-.')

    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.ylim(-0.2, 0.4)

    plt.savefig(path)


def draw_mapping_3d(path, data):


    N = data.shape[0]
    fig = plt.figure(dpi = 200)
    # ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=-75, elev=25) # for car scene
    
    x_min = data[:, :, 0].min()
    x_min = min(x_min, -0.5)
    x_max = data[:, :, 0].max()
    x_max = max(x_max, 0.5)
    y_min = data[:, :, 1].min()
    y_min = min(y_min, -0.5)
    y_max = data[:, :, 1].max()
    y_max = max(y_max, 0.5)
    z_min = data[:, :, 2].min()
    z_min = min(z_min, -0.5)
    z_max = data[:, :, 2].max()
    z_max = max(z_max, 0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    for i in range(N):
        x = data[i, :, 0]
        y = data[i, :, 1]
        z = data[i, :, 2]
        
        # plt.plot(x, y, marker='o', linestyle='-')
        # plt.plot(x, y, linestyle='-')
        ax.plot(x, y, z, linestyle='--', linewidth=0.75)
        # ax.plot(x, y, z, linestyle='--', linewidth=0.75, marker='o')
        # plt.plot(x, y, linestyle=':')
        # plt.plot(x, y, linestyle='-.')

    ax.legend()
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.ylim(-0.2, 0.4)
    plt.savefig(path)
    # plt.show()
    
def draw_mapping_3d_animation(path, data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(dpi = 200)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=-75, elev=25) # for car scene
    
    M = data.shape[0] # points number
    N = data.shape[1] # frames number
    # ax = plt.axes(projection='3d')
    
    x_min = data[:, :, 0].min()
    x_min = min(x_min, -0.5)
    x_max = data[:, :, 0].max()
    x_max = max(x_max, 0.5)
    y_min = data[:, :, 1].min()
    y_min = min(y_min, -0.5)
    y_max = data[:, :, 1].max()
    y_max = max(y_max, 0.5)
    z_min = data[:, :, 2].min()
    z_min = min(z_min, -0.5)
    z_max = data[:, :, 2].max()
    z_max = max(z_max, 0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    

    # sc = ax.scatter([], [], [], s=10)


    # 轨迹存储数组
    trajectories = np.zeros_like(data)

    # 画出轨迹线
    lines = [ax.plot([], [], [], linestyle='--', linewidth=0.75)[0] for _ in range(M)]

    # 更新函数，用于在每一帧更新轨迹线的数据
    def update(frame):
        # 更新轨迹线
        for i in range(M):
            lines[i].set_data(data[i, :frame, 0], data[i, :frame, 1])
            lines[i].set_3d_properties(data[i, :frame, 2])

        return lines

    # 创建动画
    animation = FuncAnimation(fig, update, frames=N, interval=10000/N, blit=False)
    # def update(frame):
    #     sc._offsets3d = (data[frame, :, 0], data[frame, :, 1], data[frame, :, 2])
    #     return sc,

    # animation = FuncAnimation(fig, update, frames=N, interval=50, blit=True)

    # 显示动画
    # plt.savefig(path)
    animation.save(path, writer='imagemagick', fps=20)
    # plt.show()


def write_ply(points, filename, text=False):
    """
    input: Nx3, write points to filename as PLY format.
    """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)


def read_ply(file):
    # read full file and drop empty lines
    ply_df = pd.read_csv(file, header=None, engine='python')
    ply_df = ply_df.dropna()
    # extract number of vertices and faces, and row that marks the end of header
    number_vertices = int(ply_df[ply_df[0].str.contains('element vertex')][0].iloc[0].split()[2])
    number_faces = int(ply_df[ply_df[0].str.contains('element face')][0].iloc[0].split()[2])
    end_header = ply_df[ply_df[0].str.contains('end_header')].index.tolist()[0]
    # read vertex coordinates into dict
    vertex_df = pd.read_csv(file, skiprows=range(end_header + 1),
                            nrows=number_vertices, sep='\s*', header=None,
                            engine='python')
    vertex_array = np.array(vertex_df)
    # read face indices into dict
    face_df = pd.read_csv(file, skiprows=range(end_header + number_vertices + 1),
                          nrows=number_faces, sep='\s*', header=None,
                          engine='python')
    face_array = np.array(face_df.iloc[:, 1:4])
    
    return vertex_array, face_array


def draw_points(uv, image, output_dir=None):
    
    point_size = 1

    point_col   or = (0, 255, 255)
    
    thickness = 2 #  0 、4、8
    for i, coor in enumerate(uv.cpu().numpy()):
        try:
            cv.circle(image, (int(coor[0]),int(coor[1])), point_size, point_color, thickness)
        except:
            continue
    if output_dir is not None:
        cv.imwrite(output_dir,image)
        
    return image

