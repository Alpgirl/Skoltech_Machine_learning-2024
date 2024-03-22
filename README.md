# Skoltech Machine Learning Course Project 2024

Welcome to our Machine Learning course project repository! This project is part of the 2024 curriculum at Skoltech, aimed at exploring innovative solutions and advancing the field of machine learning.

## Project Overview

This project is focused on creating an advanced approach for the accurate segmentation and classification of brain tumors, utilizing a dataset called “Brain MRI segmentation” for segmentation task and then train classification on  "Brain Tumor Classification" dataset that are filled with magnetic resonance (MR) images of the brain. Our goal is to accurately segment and classify various types of tumors through the application of various deep learning models, including CNN (Convolutional Neural Networks), U-Net, ResNet, and others. This endeavor seeks to leverage the capabilities of these models to improve the precision and efficiency of brain tumor diagnosis.

## Team Members

| Name              | Role                | Contact Information |
|-------------------|---------------------|---------------------|
| Hasaan Maqsood    | Project Lead        | [Email](Hasaan.Maqsood@skoltech.ru) |
| Inna Larina       | CV Engineer         | [Email](inna.larina@skoltech.ru) |
| Iana Kulichenko   | Data Scientist      | [Email](Iana.Kulichenko@skoltech.ru) |

## Table of Contents

- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#Team-Member)
- [License](#license)


# Project File Structure

This repository supports classification and segmentation machine learning tasks, alongside providing presentation and report documentation. Below is a detailed overview of the repository's structure.

## Repository Structure

```bash
├── classification
│ ├── classification.ipynb          # Jupyter notebook with the model training code for classification.
│ └── classification dataset         # Directory containing the dataset for classification tasks.
│    ├── Training                    # Training dataset for the classification model.
│    └── Testing                     # Testing dataset for the classification model.
├── segmentation
│ ├── segmentation.ipynb            # Jupyter notebook with the model training code for segmentation.
│ ├── segm_model.pth                # Saved model weights for the segmentation model.
│ └── segmentation dataset          # Directory containing the dataset for segmentation tasks.
├── Presentation                     # Directory or file with presentation materials.
├── Report                           # Directory or file with detailed project report.
├── Readme.md                        # Overview and guide for using this repository.
└── requirements.txt                 # Required libraries to run the notebooks.

```
Provide step-by-step instructions on how to install the project.
## Installation

```bash
git clone [https://your.project.repo.link.here](https://github.com/Hasaanmaqsood/Skoltech_Machine_learning-2024.git)
cd Skoltech_Machine_learning-2024
pip install -r requirements.txt

```
# Datasets 
## Here How you can download Segmentation Dataset
```bash
# Downloading dataset from kaggle
#upload kaggle json
from google.colab import files
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p /content
!unzip /content/lgg-mri-segmentation.zip -d /content/dataset
```
## Team Member

<a href="https://github.com/Hasaanmaqsood/Skoltech_Machine_learning-2024/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Hasaanmaqsood/Skoltech_Machine_learning-2024"/>
</a>







