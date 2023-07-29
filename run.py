import numpy as np
import torch.nn.functional as F
import csv
import random
import pandas as pd
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.text import Tokenizer # The Tokenizer class provided by Keras allows you to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary)
from tensorflow.keras.preprocessing.sequence import pad_sequences # pad_sequences is a function from the Keras preprocessing module that can be used to pad sequences to the same length.
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm


batch_size = 256
latent_size = 100
ephocs =10
lr = 0.001

# device = get_defult_device()
# path = "dta.csv"

# data = []

# with open(path) as csvfile:
#   reader = csv.reader(csvfile)
#   next(reader)
#   for row in reader:
#     triplet = []
#     triplet.append(row[0])
#     data.append(triplet)
# len(data)


# smiles = []

# random.shuffle(data)

# for triplet in range(len(data)):
#   smiles.append(data[triplet][0])


# tokenizar_smiles = Tokenizer(char_level= True)
# tokenizar_smiles.fit_on_texts(smiles)

# word_index_smiles = tokenizar_smiles.word_index
# vocab_size_smiles = len(word_index_smiles)

# smiles = tokenizar_smiles.texts_to_sequences(smiles)
# padded_smiles = pad_sequences(smiles, truncating="post", padding="post", maxlen=85)


# train_smiles = np.array(smiles)
# smiles = torch.tensor(train_smiles)
# train_loader = DataLoader(smiles, batch_size=batch_size, shuffle=True)



path = "dta.csv"
Data_set = pd.read_csv(path)

data = []

with open(path) as csvfile:
  reader = csv.reader(csvfile)
  next(reader)
  for row in reader:
    triplet = []
    triplet.append(row[0])
    triplet.append(row[1])
    triplet.append(float(row[2]))
    data.append(triplet)
len(data)
# print(data)

smiles = []
protein = []
labels = []
random.shuffle(data)

for triplet in range(len(data)):
  smiles.append(data[triplet][0])
  protein.append(data[triplet][1])
  labels.append(data[triplet][2])


split = int(0.9 * len(smiles))
train_smiles = smiles[:split]
test_smiles = smiles[split:]
train_proteins = protein[:split]
test_proteins = protein[split:]
train_labels = labels[:split]
test_labels = labels[split:]

#Tokenize_smiles

#train
tokenizar_smiles = Tokenizer(char_level= True)
tokenizar_smiles.fit_on_texts(train_smiles)

word_index_smiles = tokenizar_smiles.word_index
vocab_size_smiles = len(word_index_smiles)

train_sequence_smiles = tokenizar_smiles.texts_to_sequences(train_smiles)
train_padded_smiles = pad_sequences(train_sequence_smiles, truncating="post", padding="post", maxlen=85)

#test
test_sequence_smiles = tokenizar_smiles.texts_to_sequences(test_smiles)
test_padded_smiles = pad_sequences(test_sequence_smiles, truncating="post", padding="post", maxlen=85)



#Tokenize_proteins

#train
tokenizar_proteins = Tokenizer(char_level= True)
tokenizar_proteins.fit_on_texts(train_proteins)

word_index_proteins = tokenizar_proteins.word_index
vocab_size_proteins = len(word_index_proteins)

train_sequence_proteins = tokenizar_proteins.texts_to_sequences(train_proteins)
train_padded_proteins = pad_sequences(train_sequence_proteins, truncating="post", padding="post", maxlen=1200)

#test
test_sequence_proteins = tokenizar_proteins.texts_to_sequences(test_proteins)
test_padded_proteins = pad_sequences(test_sequence_proteins, truncating="post", padding="post", maxlen=1200)


train_smiles_array = np.array(train_padded_smiles)
test_smiles_array = np.array(test_padded_smiles)
train_proteins_array = np.array(train_padded_proteins)
test_proteins_array = np.array(test_padded_proteins)
train_labels_array = np.array(train_labels, dtype="float32")
test_labels_array = np.array(test_labels, dtype="float32")


# Convert the data to tensors
train_smiles_array = torch.tensor(train_smiles_array)
train_proteins_array = torch.tensor(train_proteins_array)
train_labels_array = torch.tensor(train_labels_array)
test_smiles_array = torch.tensor(test_smiles_array)
test_proteins_array = torch.tensor(test_proteins_array)
test_labels_array = torch.tensor(test_labels_array)

# Create a TensorDataset from the train data
train_dataset = TensorDataset(train_smiles_array, train_proteins_array, train_labels_array)

# Create a DataLoader from the train dataset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create a TensorDataset from the test data
test_dataset = TensorDataset(test_smiles_array, test_proteins_array, test_labels_array)

# Create a DataLoader from the test dataset
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

discriminator = nn.Sequential(
      nn.Conv1d(in_channels=128, out_channels=32 ,kernel_size=7, stride=1), 
      nn.ReLU(),
      nn.Conv1d(32, 32* 2, 7, 1),   
      nn.ReLU(),
      nn.Conv1d(32 * 2, 32 * 3, 7, 1),
      nn.ReLU(),
      nn.Conv1d(32 * 3, 1, 7, 1),
      nn.Sigmoid()
)
discriminator = discriminator.cuda()

genarator = nn.Sequential(
      nn.ConvTranspose1d(latent_size, 32 * 3, 7, 1, 0),
      nn.ReLU(),
      nn.ConvTranspose1d(32 * 3, 32 *2 , 7, 1, 0),
      nn.ReLU(),
      nn.ConvTranspose1d(32*2, 32 * 1, 7, 1, 0),
      nn.ReLU(),
      nn.ConvTranspose1d(32 * 1, 128, 7, 1, 0)
)

genarator = genarator.cuda()

def train_generator(opt_g):

    #clear genarator
    opt_g.zero_grad()

    #generate Fake images
    latent = torch.randn(batch_size, latent_size, 1,1)
    fake_smiles = genarator(latent)

    #try to fool the discriminator

    preds = discriminator(fake_smiles)
    targets = torch.ones(batch_size, 1)
    loss = F.binary_cross_entropy(preds, targets)

    #Update generator wieghts
    loss.backword()
    opt_g.step()

    return loss.item()


def train_discriminator(smiles, opt_d):
   
   opt_d.zero_grad()
   #pass real data through discriminator
   real_preds = discriminator(smiles)
   real_target = torch.ones(smiles.size(0), 1)
   real_loss = F.binary_cross_entropy(real_preds, real_target)
   real_score = torch.mean(real_preds).item()

   #Genarate Fake Smiles
   latent = torch.randn(batch_size, latent_size, 1, 1)
   fake_smiles = genarator(latent)


   #pass fake Smiles through Discriminator
   fake_target = torch.zeros(fake_smiles.size(0), 1)
   fake_preds = discriminator(fake_smiles)

   fake_loss = F.binary_cross_entropy(fake_preds, fake_target)
   fake_score = torch.mean(fake_preds).item()

   #update discriminator wieghts

   loss = real_loss + fake_loss
   loss.backward()
   opt_d.step()

   return loss.item(), real_score, fake_score

def Embadding(train_smiles_array):
  embad = nn.Embedding(vocab_size_smiles, 128)
  smiles = embad(train_smiles_array)
  return smiles

# def fit(ephocs, lr):
torch.cuda.empty_cache()

    
#losses and score
losses_g = []
losses_d = []
real_scores = []
fake_scores = []

#create optemizers
opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
opt_g = torch.optim.Adam(genarator.parameters(), lr=lr, betas=(0.5, 0.999))


for epoch in range(ephocs):
  for train_smiles_array,_,_ in tqdm(train_dataloader):
    smiles = Embadding(train_smiles_array)
        # 
        #train discremenator
    loss_d, real_score, fake_score = train_discriminator(smiles, opt_d)

        #train generator
    loss_g = train_generator(opt_g)

        #record loss & score

    losses_g.append(loss_g)
    losses_d.append(loss_d)
    real_scores.append(real_score)
    fake_scores.append(fake_score)


        #log losses and score (last bactch)

    print("Epochs [{}/{}], loss_g: {:4f}, loss_d, real_score: {:4f}, fake_score: {:4f}",(ephocs+1, epoch, loss_g, loss_d, real_score, fake_score))

            # return losses_d, losses_g, real_scores, fake_scores
        

# history = fit(ephocs, lr)


# losses_g, losses_d, real_score, fake_score = history





