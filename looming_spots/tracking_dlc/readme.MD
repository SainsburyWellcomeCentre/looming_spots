# DLC on the HPC

## Installation


Instructions for using the HPC for DLC

```ssh user@ssh.swc.ucl.ac.uk```

```ssh user@hpc-gw1```

Then activate an interactive session on a cluster node for installing DLC:

```srun -p gpu --gres=gpu:1 -n 10 --pty bash -i```

```module load /ceph/apps/ubuntu-20/modulefiles/miniconda/4.9.2```

```module load /ceph/apps/ubuntu-20/modulefiles/cuda/11.2```

```export DLClight=True```

Create a conda environment with all the code you want installed

```conda create -n dlc python=3.9```

```source activate dlc```

```pip install deeplabcut```


To test this:

```python```

```>>> import tensorflow as tf```

```>>> tf.test.is_gpu_available()```

should return True if configured correctly.

## Running scripts

To run scripts you should submit jobs via slurm.

```ssh user@ssh.swc.ucl.ac.uk```

```ssh user@hpc-gw1```

```git clone <path to github repo of your code>```

Then you will need to create a bash script to run your code as a batch job on the cluster.

```sbatch track_mouse.sh```


