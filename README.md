# FORCE Dataset

The FORCE dataset provides a comprehensive set of human-object interaction scenarios, emphasizing intuitive physics. Unlike existing datasets, FORCE features detailed physical interactions with objects of varying resistances, such as pushing, pulling, and carrying. This dataset aims to bridge the gap in understanding the physical dynamics of human-object interactions.

## Directory Structure

* `assets/` Contains scanned 3D models of all objects used in the dataset.
* `force_npz/` Includes human motion in SMPL format, along with the corresponding object motion data (pose and translation).
* `annotations_final/` Provides the final annotations for all interactions.
* `lib/` Contains utility libraries for processing the dataset.
* `visualize_force_dataset.py` A Python script to visualize and explore the dataset.
* `betas.pkl` SMPL body shape parameters.

## Dataset Overview

### Dataset Preview
![Dataset Preview](assets/dataset_preview.gif)

<!-- A detailed overview of the dataset can be found in `media/dataset.mp4`. -->

### Features
* Physical interactions between humans and objects with various resistance levels.
* Actions include pushing, pulling, and carrying objects.

## Getting Started

### Prerequisites
1. Install the required dependencies using a package manager of your choice (e.g., pip or conda).
2. Run the visualization script:
```bash
python visualize_force_dataset.py
```

## Usage
* Load data from the `force_npz/` directory to access SMPL and object motion data.
* Use the `visualize_force_dataset.py` script to visualize interactions.

## Contact

If you have any questions or need assistance with the dataset, feel free to contact:
**Xiaohan Zhang** 📧 Email: xzhang@mpif-inf.mpg.de

## Citation

If you use the FORCE dataset in your research, please cite:

```bibtex
@inproceedings{zhang2024force,
    title = {FORCE: Dataset and Method for Intuitive Physics Guided Human-object Interaction},
    author = {Zhang, Xiaohan and Bhatnagar, Bharat Lal and Starke, Sebastian and Petrov, Ilya A. and 
              Guzov, Vladimir and Dhamo, Helisa and Pérez Pellitero, Eduardo and Pons-Moll, Gerard},
    booktitle = {International Conference on 3D Vision (3DV)},
    month = {March},
    year = {2025},
}
```

Thank you for your interest in the FORCE dataset!