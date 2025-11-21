# Federated Learning for Optic Disc and Cup Segmentation in Glaucoma Monitoring

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation & Setup](#installation--setup)
4. [Pipeline Descriptions](#pipeline-descriptions)
5. [Running Experiments](#running-experiments)
6. [Script Options Reference](#script-options-reference)

---

## 1. Project Overview

This repository implements **Federated Learning (FL) approaches for automated optic disc and optic cup segmentation** from color fundus photographs (CFPs), enabling **privacy-preserving glaucoma assessment and monitoring** across multiple clinical sites.

### Clinical Context

**Glaucoma** is a leading cause of irreversible blindness worldwide, affecting 3.54% of the population aged 40-80 and projected to impact 111.8 million people by 2040. A key indicator of glaucoma severity is the **vertical cup-to-disc ratio (CDR)**, with ratios ≥0.6 suggestive of glaucoma. Accurate automated segmentation of the optic disc and cup enables consistent CDR calculation for diagnosis and monitoring.

### Research Objectives

This study evaluates a **federated learning framework with site-specific fine-tuning** for optic disc and cup segmentation, aiming to:
- Match central model performance while preserving patient data privacy
- Improve cross-site generalizability compared to site-specific local models
- Compare multiple FL strategies: Global Validation, Weighted Global Validation, Onsite Validation, and Fine-Tuned Onsite Validation

### Model Architecture
- **Base Model:** Mask2Former with Swin Transformer backbone
- **Pre-training:** ADE20K dataset (semantic segmentation)
- **Task:** Multi-class segmentation (background, unlabeled, optic disc, optic cup)
- **Input:** Color fundus photographs (512×512, normalized)
- **Optimizer:** AdamW
- **Loss Function:** Multi-class cross-entropy

### Datasets (9 Public Sites)

A total of **5,550 color fundus photographs** from at least **917 patients** across **7 countries** were used:

| Dataset | Total Images | Test Images | Country | Characteristics |
|---------|-------------|-------------|---------|-----------------|
| **Chaksu** | 1,345 | 135 | India | Multi-center research dataset |
| **REFUGE** | 1,200 | 120 | China | Glaucoma challenge dataset |
| **G1020** | 1,020 | 102 | Germany | Benchmark retinal fundus dataset |
| **RIM-ONE DL** | 485 | 49 | Spain | Glaucoma assessment dataset |
| **MESSIDOR** | 460 | 46 | France | Diabetic retinopathy screening |
| **ORIGA** | 650 | 65 | Singapore | Multi-ethnic Asian population |
| **Bin Rushed** | 195 | 20 | Saudi Arabia | RIGA dataset collection |
| **DRISHTI-GS** | 101 | 11 | India | Optic nerve head segmentation |
| **Magrabi** | 94 | 10 | Saudi Arabia | RIGA dataset collection |
| **Total** | **5,550** | **558** | **7 countries** | **Multi-ethnic, varied protocols** |

**Data Split:** Each dataset was divided into training (80%), validation (10%), and testing (10%) subsets. For datasets with multiple expert annotations, the STAPLE (Simultaneous Truth and Performance Level Estimation) method was used to generate consensus segmentation labels.

---

## 2. Repository Structure

```
fl-strategies-for-cup-disc-segmentation/
├── driver/                          # Main execution scripts
│   ├── centraltrain.sh             # Central model training (pooled dataset)
│   ├── persite.sh                  # Local model training (9 site-specific models)
│   ├── pipeline1.sh                # FL: Global Validation
│   ├── pipeline2.sh                # FL: Weighted Global Validation
│   ├── pipeline3.sh                # FL: Onsite Validation
│   ├── pipeline4.sh                # FL: Fine-Tuned Onsite Validation
│   └── analyze_and_plot.sh         # Statistical analysis & visualization
│
├── engine/                          # Core implementation
│   ├── train/
│   │   ├── localtraining.py        # Standard training (central/local models)
│   │   ├── pipeline1.py            # Global Validation implementation
│   │   ├── pipeline2.py            # Weighted Global Validation implementation
│   │   ├── pipeline3.py            # Onsite Validation implementation
│   │   └── pipeline4.py            # Fine-Tuned Onsite Validation implementation
│   ├── inference.py                # Model inference (multiprocess)
│   ├── evaluate.py                 # Per-sample Dice score calculation
│   ├── statistical_analysis.py     # Friedman & Wilcoxon tests
│   ├── plotting.py                 # Comprehensive visualization
│   ├── datasets.py                 # Dataset definitions
│   └── utils.py                    # Utility functions
│
├── data/                            # Raw fundus images and labels (not in repo)
│   └── {dataset}/
│       ├── images/
│       └── labels/
│
├── metadata/                        # CSV files for train/val/test splits
│   ├── combined_train.csv          # All 9 datasets merged (for central model)
│   ├── combined_val.csv
│   ├── combined_test.csv
│   ├── {dataset}_train.csv         # Per-site splits (9 datasets)
│   ├── {dataset}_val.csv
│   └── {dataset}_test.csv
│
├── models/                          # Saved model checkpoints (.pt files)
│   ├── central/                    # Central model
│   ├── persite/                    # Local models (9 site-specific)
│   │   └── {dataset}/
│   ├── pipeline1/                  # Global Validation
│   ├── pipeline2/                  # Weighted Global Validation
│   ├── pipeline3/                  # Onsite Validation
│   └── pipeline4/                  # Fine-Tuned Onsite Validation (9 models)
│       └── {dataset}/
│
├── outputs/                         # Inference predictions (colored masks)
│   └── {model_type}/
│       ├── outputs/                # PNG segmentation masks
│       └── results.csv             # Prediction metadata
│
├── scores/                          # Per-sample Dice scores for all models
│   ├── disc/
│   │   └── {dataset}.csv           # All models evaluated on each dataset
│   └── cup/
│       └── {dataset}.csv
│
├── Statistics/                      # Statistical test results
│   ├── disc/
│   │   └── {dataset}_disc_pairwise_wilcoxon.csv
│   └── cup/
│       └── {dataset}_cup_pairwise_wilcoxon.csv
│
├── plots/                           # Generated visualizations
│   ├── central_vs_local_by_dataset/
│   ├── local_vs_onsite_finetuned/
│   ├── fl_base_models_comparison/
│   ├── fl_models_vs_local/
│   ├── local_vs_central/
│   ├── onsite_finetuned_comparisons/
│   └── cross_site_performance/
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This documentation
```

**Key Directories:**
- `driver/`: Shell scripts orchestrating training, inference, evaluation, and analysis
- `engine/`: Python implementations of training algorithms and analysis tools
- `metadata/`: CSV files defining train/val/test splits for each dataset
- `models/`: Trained model checkpoints (not version-controlled due to size)
- `scores/`: Evaluation results (Dice scores) for all model-dataset combinations
- `Statistics/`: Statistical comparison results (Wilcoxon tests, p-values)
- `plots/`: Publication-ready visualizations of comparative performance

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10.2
- 80GB RAM for parallel training (Trained on NVIDIA A100-80g)

### Installation Steps

```bash
# 1. Navigate to repository
cd /path/to/fl-strategies-for-cup-disc-segmentation

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate   # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 4. Pipeline Descriptions

### Comparative Evaluation Framework

This repository implements a comparative evaluation of **six approaches** for optic disc and cup segmentation:

| Approach | Name in Results | Script |
|----------|----------------|--------|
| **Central Model** | `central` | `centraltrain.sh` | 
| **Local Models** | `{dataset}_persite` | `persite.sh` |
| **Global Validation** | `pipeline1` | `pipeline1.sh` | 
| **Weighted Global Validation** | `pipeline2` | `pipeline2.sh` |
| **Onsite Validation** | `pipeline3` | `pipeline3.sh` | 
| **Fine-Tuned Onsite Validation** | `{dataset}_fl_finetuned` | `pipeline4.sh` |

---

## 5. Running Experiments

### Complete Experimental Workflow

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run all training pipelines
./driver/centraltrain.sh
./driver/persite.sh all sequential
./driver/pipeline1.sh
./driver/pipeline2.sh
./driver/pipeline3.sh
./driver/pipeline4.sh all sequential

# 3. Run statistical analysis and generate plots
./driver/analyze_and_plot.sh
```

### Skipping Training (Use Existing Models)

```bash
# If models already exist, skip training:
./driver/centraltrain.sh --skip-training
./driver/persite.sh all --skip-training
./driver/pipeline1.sh --skip-training
./driver/pipeline2.sh --skip-training
./driver/pipeline3.sh --skip-training
./driver/pipeline4.sh all --skip-training
```

### GPU Configuration

All scripts use specific CUDA devices:
- **Pipeline 1, 2, 4, Central:** `cuda:1`
- **Pipeline 3, Per-site:** `cuda:0`

Modify `--cuda_num` in driver scripts if needed.

### Expected Runtime

| Pipeline | Training Time | Inference Time | Total |
|----------|--------------|----------------|-------|
| Central | ~8-12 hours | ~30 min | ~12.5 hours |
| Per-site (all) | ~6-8 hours (parallel) | ~4.5 hours | ~12.5 hours |
| Pipeline 1 | ~10-15 hours | ~30 min | ~15.5 hours |
| Pipeline 2 | ~8-12 hours | ~30 min | ~12.5 hours |
| Pipeline 3 | ~12-16 hours | ~30 min | ~16.5 hours |
| Pipeline 4 (all) | ~6-8 hours | ~4.5 hours | ~12.5 hours |
| **Total (all)** | **~50-70 hours** | **~10 hours** | **~80 hours** |

---

## 6. Script Options Reference

### Shell Scripts (driver/)

#### `centraltrain.sh`
Central model training on pooled dataset.

**Usage:**
```bash
./driver/centraltrain.sh [--skip-training] [--help|-h]
```

**Options:**
- `--skip-training`: Skip training and use existing model checkpoint
- `--help`, `-h`: Show help message

**Workflow:** Training → Inference → Evaluation (disc & cup)

---

#### `persite.sh`
Local model training for individual sites.

**Usage:**
```bash
./driver/persite.sh <dataset_name> [--skip-training]
./driver/persite.sh all [sequential|parallel] [--skip-training]
./driver/persite.sh --help
```

**Arguments:**
- `<dataset_name>`: One of: `binrushed`, `chaksu`, `drishti`, `g1020`, `magrabi`, `messidor`, `origa`, `refuge`, `rimone`
- `all`: Run all datasets
- `sequential` (default): Run datasets one after another
- `parallel`: Run all datasets simultaneously (requires multiple GPUs)

**Options:**
- `--skip-training`: Skip training and use existing models
- `--help`, `-h`: Show help message

**Workflow:** Training → Inference → Evaluation (disc & cup) for each dataset

---

#### `pipeline1.sh`
Federated Learning: Global Validation.

**Usage:**
```bash
./driver/pipeline1.sh [--skip-training] [--help|-h]
```

**Options:**
- `--skip-training`: Skip training and use existing model
- `--help`, `-h`: Show help message

**Workflow:** Training → Inference → Evaluation (disc & cup)

---

#### `pipeline2.sh`
Federated Learning: Weighted Global Validation.

**Usage:**
```bash
./driver/pipeline2.sh [--skip-training] [--help|-h]
```

**Options:**
- `--skip-training`: Skip training and use existing model
- `--help`, `-h`: Show help message

**Workflow:** Training → Inference → Evaluation (disc & cup)

---

#### `pipeline3.sh`
Federated Learning: Onsite Validation.

**Usage:**
```bash
./driver/pipeline3.sh [--skip-training] [--help|-h]
```

**Options:**
- `--skip-training`: Skip training and use existing model
- `--help`, `-h`: Show help message

**Workflow:** Training → Inference → Evaluation (disc & cup)

---

#### `pipeline4.sh`
Federated Learning: Fine-Tuned Onsite Validation.

**Usage:**
```bash
./driver/pipeline4.sh <dataset_name> [--skip-training]
./driver/pipeline4.sh all [sequential] [--skip-training]
./driver/pipeline4.sh all parallel [--skip-training]
./driver/pipeline4.sh --help
```

**Arguments:**
- `<dataset_name>`: One of: `binrushed`, `chaksu`, `drishti`, `g1020`, `magrabi`, `messidor`, `origa`, `refuge`, `rimone`
- `all`: Run all datasets
- `sequential` (default): Run datasets one after another
- `parallel`: Run all datasets simultaneously

**Options:**
- `--skip-training`: Skip training and use existing models
- `--help`, `-h`: Show help message

**Workflow:** Training → Inference → Evaluation (disc & cup) for each dataset

---

#### `analyze_and_plot.sh`
Statistical analysis and visualization.

**Usage:**
```bash
./driver/analyze_and_plot.sh
```

**No options.** Runs statistical analysis for both disc and cup segmentation, then generates all plots.

**Workflow:** Statistical Analysis (disc) → Statistical Analysis (cup) → Plot Generation

---

### Python Scripts (engine/)

#### `engine/train/localtraining.py`
Standard training script for central and local models.

**Options:**
- `--train_csv` (required): Path to CSV file with training data
- `--val_csv` (required): Path to CSV file with validation data
- `--csv_img_path_col` (default: `image`): Column name for image paths in CSV
- `--csv_label_path_col` (default: `label`): Column name for label paths in CSV
- `--output_directory` (default: `./outputs`): Directory for output files
- `--dataset_mean` (required): Array of float values for normalization mean (e.g., `0.768 0.476 0.290`)
- `--dataset_std` (required): Array of float values for normalization std (e.g., `0.220 0.198 0.166`)
- `--lr` (default: `0.00003`): Learning rate
- `--batch_size` (default: `16`): Batch size for training and testing
- `--jitters` (optional): Array of float jitter values: brightness, contrast, saturation, hue, probability
- `--num_epochs` (default: `50`): Maximum number of training epochs
- `--patience` (default: `5`): Early stopping patience
- `--num_val_outputs_to_save` (default: `3`): Number of validation examples to save
- `--num_workers` (default: `0`): Number of dataloader workers
- `--cuda_num` (default: `0`): CUDA device number
- `--model_dir` (optional): Directory to save model checkpoints (overrides default)

---

#### `engine/train/pipeline1.py`
Federated Learning: Global Validation implementation.

**Options:**
- `--train_csv` (required, multiple): Paths to CSV files for all training datasets
- `--val_csv` (required, multiple): Paths to CSV files for all validation datasets
- `--local_sites_training_epochs` (default: `1`): Number of epochs per site before federated averaging
- `--fl_rounds` (default: `100`): Number of federated learning rounds
- `--csv_img_path_col` (default: `image_path`): Column name for image paths
- `--csv_label_path_col` (default: `label_path`): Column name for label paths
- `--output_directory` (default: `./outputs`): Output directory
- `--dataset_mean` (default: `[0.768, 0.476, 0.289]`): Normalization mean
- `--dataset_std` (default: `[0.221, 0.198, 0.165]`): Normalization std
- `--lr` (default: `0.00003`): Learning rate
- `--batch_size` (default: `8`): Batch size
- `--jitters` (default: `[0.2, 0.2, 0.05, 0.05, 0.75]`): Data augmentation jitter values
- `--patience` (default: `7`): Early stopping patience
- `--num_val_outputs_to_save` (default: `5`): Number of validation examples to save
- `--num_workers` (default: `4`): Number of dataloader workers
- `--cuda_num` (default: `0`): CUDA device number
- `--fl_finetuned` (optional): Path to federated learning fine-tuned model
- `--fl_patience` (default: `5`): Early stopping for federated learning rounds
- `--start_fl_round` (default: `0`): Federated learning round to start from (for resuming)

---

#### `engine/train/pipeline2.py`
Federated Learning: Weighted Global Validation implementation.

**Options:**
- `--train_csv` (required, multiple): Paths to CSV files for all training datasets
- `--val_csv` (required, multiple): Paths to CSV files for all validation datasets
- `--csv_img_path_col` (default: `image_path`): Column name for image paths
- `--csv_label_path_col` (default: `label_path`): Column name for label paths
- `--output_directory` (default: `./outputs`): Output directory
- `--dataset_mean` (default: `[0.768, 0.476, 0.289]`): Normalization mean
- `--dataset_std` (default: `[0.221, 0.198, 0.165]`): Normalization std
- `--lr` (default: `0.00003`): Learning rate
- `--batch_size` (default: `8`): Batch size
- `--jitters` (default: `[0.2, 0.2, 0.05, 0.05, 0.75]`): Data augmentation jitter values
- `--num_epochs` (default: `100`): Maximum number of epochs
- `--patience` (default: `7`): Early stopping patience
- `--num_val_outputs_to_save` (default: `5`): Number of validation examples to save
- `--num_workers` (default: `0`): Number of dataloader workers
- `--cuda_num` (default: `1`): CUDA device number
- `--fl_finetuned` (optional): Path to federated learning fine-tuned model

---

#### `engine/train/pipeline3.py`
Federated Learning: Onsite Validation implementation.

**Options:**
- `--train_csv` (required, multiple): Paths to CSV files for all training datasets
- `--val_csv` (required, multiple): Paths to CSV files for all validation datasets
- `--local_sites_training_epochs` (default: `10`): Maximum epochs per site (may stop early)
- `--fl_rounds` (default: `100`): Number of federated learning rounds
- `--csv_img_path_col` (default: `image_path`): Column name for image paths
- `--csv_label_path_col` (default: `label_path`): Column name for label paths
- `--output_directory` (default: `./outputs`): Output directory
- `--dataset_mean` (default: `[0.768, 0.476, 0.289]`): Normalization mean
- `--dataset_std` (default: `[0.221, 0.198, 0.165]`): Normalization std
- `--lr` (default: `0.00003`): Learning rate
- `--batch_size` (default: `8`): Batch size
- `--jitters` (default: `[0.2, 0.2, 0.05, 0.05, 0.75]`): Data augmentation jitter values
- `--patience` (default: `7`): Local early stopping patience
- `--num_val_outputs_to_save` (default: `5`): Number of validation examples to save
- `--num_workers` (default: `0`): Number of dataloader workers
- `--fl_patience` (default: `3`): Federated learning early stopping patience
- `--cuda_num` (default: `0`): CUDA device number

---

#### `engine/train/pipeline4.py`
Federated Learning: Fine-Tuned Onsite Validation implementation.

**Options:**
- `--train_csv` (required, multiple): Paths to CSV files for training dataset
- `--val_csv` (required, multiple): Paths to CSV files for validation dataset
- `--csv_img_path_col` (default: `image_path`): Column name for image paths
- `--csv_label_path_col` (default: `label_path`): Column name for label paths
- `--output_directory` (default: `./outputs`): Output directory
- `--dataset_mean` (default: `[0.768, 0.476, 0.289]`): Normalization mean
- `--dataset_std` (default: `[0.221, 0.198, 0.165]`): Normalization std
- `--lr` (default: `0.00003`): Learning rate
- `--batch_size` (default: `8`): Batch size
- `--jitters` (default: `[0.2, 0.2, 0.05, 0.05, 0.75]`): Data augmentation jitter values
- `--num_epochs` (default: `100`): Maximum number of epochs
- `--patience` (default: `7`): Early stopping patience
- `--num_val_outputs_to_save` (default: `5`): Number of validation examples to save
- `--num_workers` (default: `0`): Number of dataloader workers
- `--cuda_num` (default: `0`): CUDA device number
- `--fl_finetuned` (optional): Path to federated learning fine-tuned model
- `--pretrained_global_model_path` (optional): Path to pretrained global model (alias for `fl_finetuned`)
- `--model_dir` (optional): Directory to save model checkpoints

---

#### `engine/inference.py`
Model inference script.

**Options:**
- `--model_path` (required): Path to trained model weights (.pt file)
- `--input_csv` (required): Path to CSV file with images to process
- `--csv_path_col_name` (required): Column name for image paths in CSV
- `--output_root_dir` (required): Root directory for outputs (CSV saved to root, images to `root_dir/outputs`)
- `--num_processes` (default: `1`): Number of parallel processes
- `--cuda_num` (default: `0`): CUDA device number

---

#### `engine/evaluate.py`
Evaluation script for calculating Dice scores.

**Options:**
- `--prediction_folder` (required): Path to folder containing prediction masks
- `--label_folder` (required): Path to folder containing ground truth labels
- `--csv_path` (required): Path to CSV file with image, label, and dataset information
- `--eval_disc` (flag): Evaluate disc segmentation (if not set, evaluates cup)
- `--cuda_num` (default: `0`): CUDA device number
- `--output_csv` (required): Path to save per-sample results CSV
- `--model_name` (required): Name of the model being evaluated (for statistical analysis)
- `--statistical_output_dir` (required): Root directory for statistical analysis CSVs (e.g., `scores`)

---

#### `engine/statistical_analysis.py`
Statistical analysis using Friedman and Wilcoxon tests.

**Options:**
- `--eval_type` (default: `cup`): Type of segmentation to analyze (`disc` or `cup`)
- `--input_dir` (optional): Custom input directory (default: `scores/disc` or `scores/cup`)
- `--output_dir` (optional): Custom output directory (default: `Statistics/disc` or `Statistics/cup`)
- `--skip-summaries` (flag): Skip generating optional summary files (only create pairwise_wilcoxon.csv files)

---

#### `engine/plotting.py`
Comprehensive plot generation.

**Options:**
- `--disc_results_dir` (optional): Directory containing disc statistical analysis results (default: `Statistics/disc`)
- `--cup_results_dir` (optional): Directory containing cup statistical analysis results (default: `Statistics/cup`)
- `--output_dir` (default: `./plots`): Output directory for generated plots

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{shrivastava2025federated,
  title={A Federated Learning-based Optic Disc and Cup Segmentation Model for Glaucoma Monitoring in Color Fundus Photographs},
  author={Shrivastava, Sudhanshu MS and Thakuria, Upasana MS and Kinder, Scott MS and Nebbia, Giacomo PhD and Zebardast, Nazlee MD MPH and Baxter, Sally L. MD MSc and Xu, Benjamin MD PhD and Alryalat, Saif Aldeen MD and Kahook, Malik MD and Kalpathy-Cramer, Jayashree PhD and Singh, Praveer PhD},
  year={2025},
  affiliation={1Ophthalmology, University of Colorado Anschutz Medical Campus, Aurora, Colorado; 2Massachusetts Eye and Ear Infirmary, Harvard Medical School, Massachusetts, United States; 3Division of Ophthalmology Informatics and Data Science, Viterbi Family Department of Ophthalmology and Shiley Eye Institute, University of California, San Diego, CA, USA; 4Division of Biomedical Informatics, Department of Medicine, University of California, San Diego, CA, USA; 5Roski Eye Institute, Keck School of Medicine, University of Southern California, Los Angeles, CA, USA}
}
```

**Authors:**
- Shrivastava, Sudhanshu MS¹
- Thakuria, Upasana MS¹
- Kinder, Scott MS¹
- Nebbia, Giacomo PhD¹
- Zebardast, Nazlee MD MPH²
- Baxter, Sally L. MD, MSc³,⁴
- Xu, Benjamin MD PhD⁵
- Aldeen Alryalat, Saif Aldeen MD¹
- Kahook, Malik MD¹
- Kalpathy-Cramer, Jayashree PhD¹
- Singh, Praveer PhD¹

**Affiliations:**
1. Ophthalmology, University of Colorado Anschutz Medical Campus, Aurora, Colorado
2. Massachusetts Eye and Ear Infirmary, Harvard Medical School, Massachusetts, United States
3. Division of Ophthalmology Informatics and Data Science, Viterbi Family Department of Ophthalmology and Shiley Eye Institute, University of California, San Diego, CA, USA
4. Division of Biomedical Informatics, Department of Medicine, University of California, San Diego, CA, USA
5. Roski Eye Institute, Keck School of Medicine, University of Southern California, Los Angeles, CA, USA

**Corresponding Author:**  
Praveer Singh, PhD  
Department of Ophthalmology  
University of Colorado Anschutz Medical Campus  
1675 Aurora Ct, Aurora, CO 80045  
Email: Praveer.Singh@cuanschutz.edu

---

## Acknowledgments

This research was conducted at the University of Colorado Anschutz Medical Campus, Department of Ophthalmology, in collaboration with:
- Massachusetts Eye and Ear Infirmary, Harvard Medical School
- Viterbi Family Department of Ophthalmology and Shiley Eye Institute, UC San Diego
- Division of Biomedical Informatics, UC San Diego
- Roski Eye Institute, Keck School of Medicine, USC

**Public Datasets Used:**
- Bin Rushed, Magrabi, MESSIDOR (RIGA collection)
- Chákṣu database (Manipal Academy of Higher Education)
- DRISHTI-GS (Indian Institute of Technology)
- G1020 (German benchmark dataset)
- ORIGA (Singapore Eye Research Institute)
- REFUGE (glaucoma challenge dataset)
- RIM-ONE DL (Spanish multi-center study)

We acknowledge the contributors and institutions that made these datasets publicly available for research.

---

## License

This code is provided for research purposes. Please refer to individual dataset licenses for data usage restrictions.

---

*Documentation Version: 2.0 | Last Updated: 2025-11-14*
