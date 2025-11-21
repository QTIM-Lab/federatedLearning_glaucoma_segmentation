# SLURM Job Configurations

If you need to run these scripts on SLURM-based HPC systems, add the following SLURM headers at the beginning of the script files:

## Basic SLURM Configuration

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --job-name=your_job_name
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Optional environment setup for HPC systems
if command -v module >/dev/null 2>&1; then
  module purge
  module load python/3.10.2
fi

# Then add the standard venv activation that's already in the scripts...
```

## Pipeline-Specific SLURM Configurations

### Pipeline 1 (True FL with Gradient Accumulation)
```bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=32
#SBATCH --job-name=pipeline1_fl_rounds
#SBATCH --output=pipeline1_%j.out
#SBATCH --error=pipeline1_%j.err
```

### Pipeline 2 (Weighted FedAvg)
```bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --job-name=pipeline2_weighted_fedavg
#SBATCH --output=pipeline2_%j.out
#SBATCH --error=pipeline2_%j.err
```

### Pipeline 3 (Democratic FedAvg)
```bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --job-name=pipeline3_unweighted_fedavg
#SBATCH --output=pipeline3_%j.out
#SBATCH --error=pipeline3_%j.err
```

### Pipeline 4 (FL Fine-tuning)
```bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --job-name=pipeline4_fl_finetune
#SBATCH --output=pipeline4_%j.out
#SBATCH --error=pipeline4_%j.err
```

### Per-site Training
```bash
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --job-name=persite_training
#SBATCH --output=persite_%j.out
#SBATCH --error=persite_%j.err
```

### Central Training
```bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --job-name=central_training
#SBATCH --output=central_%j.out
#SBATCH --error=central_%j.err
```

## Resource Requirements Guidelines

- **GPU Memory**: Most scripts require at least 16GB GPU memory
- **RAM**: 32GB+ recommended for federated learning scripts
- **Time Limits**: 
  - Pipeline 1: 24-48 hours (depending on FL rounds)
  - Pipelines 2-3: 12-24 hours
  - Pipeline 4: 6-12 hours (fine-tuning)
  - Per-site: 4-8 hours per dataset
  - Central: 8-12 hours

## HPC-Specific Notes

1. **Module Loading**: Adjust module names based on your HPC system
2. **Partition Names**: Update partition names to match your cluster
3. **GPU Types**: Specify GPU types if needed (e.g., `--gres=gpu:v100:1`)
4. **Account/QOS**: Add `#SBATCH --account=your_account` if required
5. **Email Notifications**: Add `#SBATCH --mail-type=END,FAIL --mail-user=your_email@domain.com`

## Example SLURM Submission

```bash
# Add SLURM header to your script, then submit:
sbatch driver/pipeline4.sh

# Or create a wrapper script:
cat > submit_pipeline4.sh << 'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --job-name=pipeline4_fl_finetune
#SBATCH --output=pipeline4_%j.out
#SBATCH --error=pipeline4_%j.err

# Load modules if needed
module purge
module load python/3.10.2

# Run the actual script
bash driver/pipeline4.sh binrushed
EOF

sbatch submit_pipeline4.sh
```
