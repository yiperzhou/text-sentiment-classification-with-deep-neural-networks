## library
 pip install torchtext==0.2.3


 # request allocation resource
## srun
srun --pty -J torch --gres=gpu:1 --partition=gpu --time=5-23:59:00 --mem=30720 --ntasks=1 --cpus-per-task=8 /bin/bash -i