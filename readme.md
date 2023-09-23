## Setup
```

# build environment with python 3.7
conda create -n pinf python=3.7
conda activate pinf # all following operations are using this environment.

# if ffmpeg is not installed (test by ffmpeg -version)
conda install -c conda-forge ffmpeg 
conda install ffmpeg

# requirments
pip install -r requirments

# raymarching
cd raymarching
pip install -e .

```



## Run

Hybrid scene (default, original PINF)
kernprof -l run_pinf.py --config configs/PINF_configs/sphere.txt


Scalar scene:
- original PINF
```
CUDA_VISIBLE_DEVICES=7 kernprof -l run_pinf.py --config configs/PINF_configs/scalar.txt

Hybrid scene with neus:
```
CUDA_VISIBLE_DEVICES=7 python run_pinf.py --config configs/sphere_neus.txt
or
CUDA_VISIBLE_DEVICES=7 kernprof -l run_pinf.py --config configs/sphere_neus.txt
```

## Install problem
- Ninja is required to load C++ extensions
pip install Ninja


## EVAL
- generate mesh
```
kernprof -l -o 7.2.lprof run_pinf.py  --config configs/hybrid_neus_cuda/game_7.7_continue_train.txt --mesh_only
```

- test speed
python -m line_profiler run_pinf.py.lprof > test_speed.txt