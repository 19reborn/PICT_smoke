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

### Training

Take the Cylinder scene as an example:


```
python train.py --config configs/cyl.txt
```


### Installing problem
- Ninja is required to load C++ extensions
pip install Ninja