# Vina-GPU 2.1 under WSL2

Building Vina-GPU on Windows through WSL2, with the fixes that were actually needed. Tested on Ubuntu under WSL2 with an NVIDIA RTX 4050 (6 GB).

## 1. WSL2

```powershell
wsl --install -d Ubuntu
```

If sessions drop shortly after starting, disable systemd:

```ini
# /etc/wsl.conf
[boot]
systemd=false
```

Then `wsl --shutdown` from PowerShell and reopen.

## 2. Expose the GPU via OpenCL

WSL2 surfaces the driver at `/usr/lib/wsl/lib`, but no ICD file points at it:

```bash
sudo mkdir -p /etc/OpenCL/vendors
echo "/usr/lib/wsl/lib/libOpenCL.so.1" | sudo tee /etc/OpenCL/vendors/nvidia.icd

sudo apt update
sudo apt install -y nvidia-cuda-toolkit clinfo build-essential libboost-all-dev
clinfo | grep -i "device name"    # should list the GPU
```

## 3. Build

```bash
git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git
cd Vina-GPU-2.1/AutoDock-Vina-GPU-2.1
```

Three fixes to the Makefile:

```bash
# hardcoded developer paths from the upstream repo
sed -i 's|/home/shidi/|'"$HOME"'/|g' Makefile

# boost headers live in /usr/include on Ubuntu, not the vendored path
sed -i 's|BOOST_INC_PATH=.*|BOOST_INC_PATH=/usr/include|' Makefile

# link boost_thread explicitly
sed -i 's|-lboost_system|-lboost_system -lboost_thread|' Makefile
```

The Makefile also references boost thread source files that are not present in a system boost install; remove those entries from the source list before building.

```bash
make clean && make source
./AutoDock-Vina-GPU-2-1 --help
```

## 4. Point MolGen at it

```yaml
docking:
  executable: /home/<user>/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
  use_wsl: true
  exhaustiveness: 32
  max_workers: 1
```

`use_wsl: true` runs the binary through `wsl` and rewrites `C:\path` as `/mnt/c/path`. Keep `max_workers` at 1: the GPU is the bottleneck and parallel invocations contend for it.

## Notes

- Throughput gain is largest on wide screens. For a handful of ligands, CPU Vina with high exhaustiveness is often faster overall once setup is counted.
- Vina-GPU's scores are not numerically identical to CPU Vina's. Do not mix results from both engines in one ranking.
- 6 GB VRAM constrains how much of a large library can be batched; watch for allocation failures on big runs.
