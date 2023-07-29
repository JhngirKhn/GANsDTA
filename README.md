# GANsDTA
Drug-Target binding affinity prediction using Generative Adversarial Networks (GANs) is an innovative approach in computational drug discovery. GANs are a class of artificial intelligence models that consist of two neural networks, a generator and a discriminator, which work together in a game-like manner. In this context, the generator generates potential drug molecules, while the discriminator assesses their binding affinity to specific target proteins. The GANs are trained on known drug-target interaction data, enabling them to learn and generate new drug-like molecules that have a higher probability of binding strongly to the target. By leveraging GANs for this prediction task, researchers aim to accelerate the drug development process by identifying potential drug candidates with enhanced binding affinity, ultimately improving the chances of success in drug discovery and optimization.

# Resources:

The data files were downloaded from https://github.com/hkmztrk/DeepDTA/tree/master/data
+ For the DAVIS dataset: test_fold_setting1.txt, train_fold_setting1.txt, Y, ligands_can.txt, and proteins.   txt are located in the data/davis/folds/ directory.
+ For the KIBA dataset: test_fold_setting1.txt, train_fold_setting1.txt, Y, ligands_can.txt, and proteins.txt are located in the data/kiba/folds/ directory.


# Source codes:
+ create_data.py: This script generates data in PyTorch format
+ utils.py: The utils.py module provides utility functions and classes used by other scripts in the codebase. One such class is TestbedDataset, which is used by create_data.py to create data in PyTorch format.
+ training.py: train the GraphVAEDTA model and make prediction.
+ models.py: recieving graph data as a input for drug while sequence data for protein
+ test.py: The script test.py is utilized to assess the performance of our saved models.

# Step-by-step running:

# Requirements
You'll need to run following commands in order to run the codes.
+ conda env create -f environment.yml
it will download all the required libraries

````
Or intall Manually..

conda create -n GraphVAEDTA python=3.8
conda activate GraphVAEDTA
+ python 3.8.11
+ conda install -y -c conda-forge rdkit
+ conda install pytorch torchvision cudatoolkit -c pytorch
+ pip install torch-cluster==1.6.0+pt112cu102
+ pip install torch-scatter==2.1.0+pt112cu102
+ pip install torch-sparse==0.6.16+pt112cu102
+ pip install torch-spline-conv==1.2.1+pt112cu102
+ pip install torch-geometric==2.2.0



## Create data in pytorch format
Running
```sh
conda activate GraphVAEDTA
python create_data.py
```
The create_data.py script generates four CSV files: kiba_train.csv, kiba_test.csv, davis_train.csv, and davis_test.csv. These files are saved in the data/ folder and can be used as input to generate PyTorch-formatted data. These files are in turn input to create data in pytorch format,
stored at data/processed/, consisting of  kiba_train.pt, kiba_test.pt, davis_train.pt, and davis_test.pt.

## Train the model
To train a model using training data.

```sh
conda activate GraphVAEDTA
python training.py 0 0
```

where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively;
 the second argument is for the index of the Cuda , 0/1/2/3 for cuda:0, cuda:1, cuda:2, or cuda:3, respectively;

This returns the model and result log files for the achieving the best MSE for testing data throughout the training.
