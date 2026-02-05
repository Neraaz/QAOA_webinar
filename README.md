# CUDA Quantum Setup on Bridges-2

## 1. Container Path

```bash
/ocean/containers/cuda-quantum.sif
```

## 2. Setup Interactive Session

```bash
interact -p GPU-shared -t 120:00 -N 1 --gres=gpu:v100-32:1
```

## 3. After Node is Granted

Once your **v node** is allocated, use the following commands from your **local computer terminal**.

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

This links the allocated `v0xx` node to your local machine.

## 4. Start Jupyter Notebook on Compute Node

From the **first terminal** where the interactive session is running:

```bash
apptainer shell --cleanenv --no-home --nv /ocean/containers/cuda-quantum.sif
```

```bash
jupyter-notebook --no-browser --ip=0.0.0.0 --port=8889
```

Copy the `http://127.0.0.1:8888/tree?...` URL from the output and paste it into your **browser**.

