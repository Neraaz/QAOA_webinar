# CUDA Quantum Setup on Bridges-2

## 1. Container Path

```bash
/ocean/containers/cuda-quantum.sif
```

## 2. Setup Interactive Session. 

Here we have requested one v100-32 gpu from GPU-shared partition. 

```bash
interact -p GPU-shared -t 120:00 -N 1 --gres=gpu:v100-32:1
```

Similarly, interactive session for CPU (RM partition).

```bash
interact -p RM -t 120:00 -N 1
```

## 3. After Node is Granted

Once your **compute node (v0xx)** (for example: v015) is allocated, use the following commands from your **local computer terminal**.

### 3.1 Check if Port 8889 is in Use

```bash
lsof -i :8889
```

### 3.2 Kill Any Process Using the Port

```bash
kill <PID>
```

### 3.3 SSH Port Forwarding to Compute Node

```bash
ssh -L 8889:localhost:8889 nnepal@bridges2.psc.edu -t ssh -L 8889:localhost:8889 v0xx
```

* The first `-L` forwards **local 8889 → login node**.
* The second `-L` forwards **login node 8889 → compute node v0xx**.
* `-t` ensures **terminal allocation** for the nested SSH.

This links the allocated `v0xx (for example: v015)` node to your local machine.

## 4. Start Jupyter Notebook on Compute Node

From the **first terminal** where the interactive session is running. `--nv` flag for NVIDIA GPU, remove flag for CPU only simulation:

For GPU usage:

```bash
apptainer shell --cleanenv --no-home --nv /ocean/containers/cuda-quantum.sif
```

For CPU only usage:

```bash
apptainer shell --cleanenv --no-home /ocean/containers/cuda-quantum.sif
```

Starts a Jupyter Notebook on a remote machine without opening a browser, listening on `0.0.0.0:8889` so you can access it locally via SSH port forwarding.

```bash
jupyter-notebook --no-browser --ip=0.0.0.0 --port=8889
```

Copy the `http://127.0.0.1:8889/tree?...` URL from the output and paste it into your **browser**.

## 5. Jupyter notebook

Open ```QAOA_MaxCut_CUDAq.ipynb``` notebook and execute shells. While using CPU or GPU, adjust backend target, accordingly.

```python
cudaq.set_target("nvidia") #single-GPU
cudaq.set_target("qpp-cpu") #CPU only, uses Q++ library for OpenMP parallelization
```

## 6. Submit job using slurm script.

We can simply submit job directly to the cluster, without using jupyter-notebook.

```bash
sbatch qaoa.job
```
One can change number of layers, graphs in maxcut.py script. `qaoa.job` is the job submission script. Finally, checkout result in slurm output (cq_....out).

## 7. Experiment with different layers, different graphs, and different backends.


### Running the tutorial on Bridges-2 OnDemand (optional)

You can also run this tutorial in a Jupyter Notebook through Bridges-2 OnDemand:

1. Start an interactive Jupyter Notebook session in OnDemand by following the steps in the [Bridges-2 User Guide](https://www.psc.edu/resources/bridges-2/user-guide#jupyter-hub).
2. When the session starts, open the tutorial notebook from the OnDemand Jupyter interface.
3. If this is your first time, run the first few cells of the notebook with the default **Python 3** kernel. They create the CUDA-Q Jupyter kernels from the container (`cuda-quantum.sif`). You only need to do this once. Then refresh the browser page so Jupyter picks up the new kernels.
4. Go to **Kernel → Change Kernel** and select **Python (CUDA-Q CPU)** or **Python (CUDA-Q GPU)**. Use the GPU kernel only in a session on a GPU partition.
