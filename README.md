# Medical Image Downsampling: LIP vs. Strided Convolution

This repository contains the experimental pipeline and code evaluating Local Importance-based Pooling (LIP) against standard strided convolutions within Feature Pyramid Networks (FPN) across chest X-ray datasets.

## 📂 Repository Structure & Experiments

Each experiment script corresponds to a specific dataset, downsampling mechanism, and model configuration:

### 🏥 NODE21 Dataset Experiments
* **`Detection_Experimet1(NODE21)`** — Evaluates **Local Importance-based Pooling (LIP) on stem max pooling** downsampling within FPN on the NODE21 dataset to measure target nodule detection performance against the baseline
* **`Detection_Experimet2(NODE21)`** — Evaluates **Local Importance-based Pooling (LIP)on stem maxpooling + layer2** downsampling within FPN on the NODE21 dataset to measure target nodule detection performance against the baseline.

### 🏥 JSRT Dataset Experiments
* **`Detection_Experimet1(JSRT)`** — Evaluates **Local Importance-based Pooling (LIP)on stem maxpooling ** downsampling within FPN on the JSRT dataset to measure target nodule detection performance against the baseline.
* **`Detection_Experimet2(jJSRT)`** — Evaluates **Local Importance-based Pooling (LIP)on stem maxpooling + layer2** downsampling within FPN on the JSRT dataset to measure target nodule detection performance against the baseline..

---

## 🚀 Running
On HPC Cluster - 
**sbatch train.slurm** for training
**sbatch test_retina.slurm** for evaluation
**sbatch test_fasterrcnn.slurm** for evaluation
