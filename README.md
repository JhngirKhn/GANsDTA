# GANsDTA
Drug-Target binding affinity prediction using Generative Adversarial Networks (GANs) is an innovative approach in computational drug discovery. GANs are a class of artificial intelligence models that consist of two neural networks, a generator and a discriminator, which work together in a game-like manner. In this context, the generator generates potential drug molecules, while the discriminator assesses their binding affinity to specific target proteins. The GANs are trained on known drug-target interaction data, enabling them to learn and generate new drug-like molecules that have a higher probability of binding strongly to the target. By leveraging GANs for this prediction task, researchers aim to accelerate the drug development process by identifying potential drug candidates with enhanced binding affinity, ultimately improving the chances of success in drug discovery and optimization.

# Resources:

The data files were downloaded from xyz.com
+ All the data presented in the CSV file


# Source codes:
+ run.py: This script trains the model

# Requirements
You'll need to run the following commands in order to run the codes.
+ conda env create -f env.yml
it will download all the required libraries

Or install Manually...
conda create -n GANsDTA python=3.8
conda activate GANsDTA
+ python 3.8.11
+ conda install -y -c conda-forge matplotlib
+ pip install torch

## Train the model
Running
```sh
conda activate GANsDTA
python run.py
```
The run.py script is designed to train the model using a specific dataset and make predictions based on the actual affinity between drugs and target proteins. Additionally, it has the capability to generate novel drugs that closely resemble those with known affinities.
