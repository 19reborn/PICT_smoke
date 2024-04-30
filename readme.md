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

### Testing

```

sbatch scripts/cyl_eval.sh

python test.py --config configs/cyl.txt --testskip 1 --output_voxel --full_vol_output

# then
sbatch scripts/eval_with_gt.s

/cluster/project/tang/yiming/project/mantaflow_nogui/build/manta /cluster/project/tang/yiming/project/pinf_clean/tools/eval/visual_eval_cyl_eular.py

```

## Tunable parameters
- neus_early_terminated
for cyl and car

- density activation
for scalar, use exp


### Installing problem
- Ninja is required to load C++ extensions
pip install Ninja