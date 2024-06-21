# An Investigation into Cephalometric Landmark Detection using Contemporary Deep Learning Methods

This it the official repository for the (work in progress) paper "An Investigation into Cephalometric Landmark Detection using Contemporary Deep Learning Methods" by Filipe Laitenberger, Dr. Hannah Scheuer, Priv.-Doz. Dr. Hanna Scheuer, Enno Liliental, Dr. Shaodi You, and Prof. Dr. Dr. Reinhard E. Friedrich.

## Install and Activate Conda Environment

```bash

conda env create -f environment.yml
conda activate cephalometry

```

## Reproducing the Results

All experiments can be repeated using the job scripts provided in the `jobs` directory. The scripts are named according to the experiment they run. For example, to run the Segformer on the benchmark dataset, either execute `sbatch jobs/Segformer/SegformerSmall-BenchmarkDataset.job` or run the contents of the latter file in the terminal.
