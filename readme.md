## Setup
```

# build environment with python 3.7
conda create -n pinf python=3.7
conda activate pinf 

# if ffmpeg is not installed (test by ffmpeg -version)
conda install -c conda-forge ffmpeg 
conda install ffmpeg

# requirments
pip install -r requirments

# raymarching
cd raymarching
pip install -e .

# test environment
python env_test.py

```


## Run

### Training

Take the Cylinder scene as an example:


```
python train.py --config configs/cyl.txt
```

### Testing

```
python test.py --config configs/cyl.txt --testskip 1 --output_voxel --full_vol_output
```


### Installing problem
- Ninja is required to load C++ extensions
```
pip install Ninja
```