# GANsDTA
## Description
Drug-Target binding affinity prediction using Generative Adversarial Networks (GANs) is an innovative approach in computational drug discovery. GANs are a class of artificial intelligence models that consist of two neural networks, a generator and a discriminator, which work together in a game-like manner. In this context, the generator generates potential drug molecules, while the discriminator assesses their binding affinity to specific target proteins. The GANs are trained on known drug-target interaction data, enabling them to learn and generate new drug-like molecules that have a higher probability of binding strongly to the target. By leveraging GANs for this prediction task, researchers aim to accelerate the drug development process by identifying potential drug candidates with enhanced binding affinity, ultimately improving the chances of success in drug discovery and optimization.

## Table of Contents
[Datasets](#datasets)  
[Model Architecture](#ModelArchitecture)  
[Installation](#Installation)  
[Usage](#Usage)  
[Training](#training)  
[Results](#results)  
[Contribution](#contribution)  
[License](#License)  
[Acknowledgments](#Acknowledgments)  

# Datasets:
### Dataset Source:
The data files were downloaded from xyz.com  
### Description:
The Davis dataset is a comprehensive collection of molecular interactions, consisting of 68 distinct drugs and 442 target proteins. The dataset includes pairwise affinities, which are quantified using Kd values, representing the kinase dissociation constant. These Kd affinity values offer insights into the binding abilities of drugs to their corresponding target proteins. However, the range of affinities spans a wide spectrum, from as low as 0.016 to as high as 10000. Due to the considerable variation in affinities, it can sometimes impact the accuracy of predictive models. To mitigate this issue and enhance the performance of our experiments, we transform the Kd values into logspace, resulting in pKd values. This logarithmic transformation compresses the range of affinities, making them more manageable and conducive to accurate predictions in our drug discovery research. The utilization of pKd values allows us to better understand and model drug-target interactions, ultimately facilitating the development of novel and effective therapeutics.  
### Preprocessing:
During the preprocessing stage, the raw dataset is cleaned and formatted to ensure consistency and remove any irrelevant information. Additionally, data normalization is performed to bring features to a similar scale, enabling more effective model training and analysis.
### Dataset Size:
The dataset comprises a total of 300,056 records, with 25,000 records allocated for training purposes, leaving the remaining 275,056 records for testing and evaluation.  
### Sample Entry:
+ **Sample ID:** 12345  
+ **Drug SMILES:** CC(=O)Nc1cnc2[nH]cnc2c1N  
+ **Target Protein Sequence:** MGGKQDKIYLVLENGKTLKFPMILYGMLVYKLLNKFRNEEYDVLDKILEKKDGNFIMKVKNGKLCDLFIFSKKDINPN  
+ **Affinity Value (pKd):** 7.82  

+ All the data presented in the CSV file

## Model Architecture
![Model](model.jpg)

# Source codes:
+ run.py: This script trains the model

# Requirements
You'll need to run the following commands in order to run the codes
```sh
+ conda env create -f env.yml
```
it will download all the required libraries

Or install Manually...
```sh
conda create -n GANsDTA python=3.8
conda activate GANsDTA
+ python 3.8.11
+ conda install -y -c conda-forge matplotlib
+ pip install torch
```
## Train the model
Running
```sh
conda activate GANsDTA
python run.py
```
The run.py script is designed to train the model using a specific dataset and make predictions based on the actual affinity between drugs and target proteins. Additionally, it has the capability to generate novel drugs that closely resemble those with known affinities.
