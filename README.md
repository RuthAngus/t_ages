# t_ages
Ages for Trevor

To run on the cluster:

```
ssh fi
<password>
<code>   

ssh rusty

cd to directory.

module load slurm

sbatch parallel.sh

tail -f slurm-xxxxxx.out

squeue -u rangus
```

parallel.sh file:

```python
#!/bin/sh

export PATH="$HOME/miniconda/bin:$PATH"
python simple_parallel.py
```


Notebooks
==========

Investigating_data
------------------
Exploring the data sent by Trevor. This is also where I plan out and test code for a parallelised cluster version.

Weakened_braking
---------------
Investigating the weird tight sequence.