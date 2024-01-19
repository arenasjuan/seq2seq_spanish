import os
import random
import requests
from zipfile import ZipFile
import pathlib
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

spacy.prefer_gpu()

# Download SpaCy models
# spacy.cli.download("es_core_news_lg")
# spacy.cli.download("en_core_web_lg")

spacy_es = spacy.load('es_core_news_lg')
spacy_en = spacy.load('en_core_web_lg')

# Download and prepare the Anki dataset
url = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
zip_file_path = "spa-eng.zip"

# Check if the file already exists
if not os.path.exists(zip_file_path):
    print("\nDownloading the dataset...")
    response = requests.get(url)
    with open(zip_file_path, 'wb') as file:
        file.write(response.content)
    print("\nDownload complete.")
else:
    print("\nDataset already downloaded.")

with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(".")

text_file = pathlib.Path("spa-eng/spa.txt")

text_pairs = []
with open(text_file, encoding='utf-8') as file:
    lines = file.read().split("\n")[:-1]

for line in lines:
    eng, spa = line.split("\t")
    spa = "[start] " + spa + " [end]"
    text_pairs.append((spa, eng))

# Tokenization
def tokenize_es(text):
    return [tok.text for tok in spacy_es.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Building vocab
def build_vocab(tokenizer, iterator):
    vocab = build_vocab_from_iterator(map(tokenizer, iterator), specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

# Split the data
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

# Load data
def load_data(pairs):
    for es, en in pairs:
        yield es, en

train_data = list(load_data(train_pairs))
valid_data = list(load_data(val_pairs))
test_data = list(load_data(test_pairs))

# Build vocabularies
spanish = build_vocab(tokenize_es, (es for es, _ in train_data))
english = build_vocab(tokenize_en, (en for _, en in train_data))

# DataLoader Collate Function
def collate_fn(batch):
    es_batch, en_batch = [], []
    for es_text, en_text in batch:
        es_batch.append(torch.tensor([spanish[token] for token in tokenize_es(es_text)], dtype=torch.long))
        en_batch.append(torch.tensor([english[token] for token in tokenize_en(en_text)], dtype=torch.long))
    es_batch = pad_sequence(es_batch, padding_value=spanish['<pad>'])
    en_batch = pad_sequence(en_batch, padding_value=english['<pad>'])
    return es_batch, en_batch

# DataLoader
BATCH_SIZE = 32
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs).squeeze(0)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        hidden, cell = self.encoder(source)

        x = target[0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

# Training hyperparameters
num_epochs = 100
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(spanish)
input_size_decoder = len(english)
output_size = len(english)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024  # Needs to be the same for both RNN's
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard to get nice loss plot
writer = SummaryWriter(f"runs/loss_plot")
step = 0

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = spanish['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("seq2seq_checkpoint.pth.tar"), model, optimizer)

# Example Spanish sentence for translation
sentence = "Un gran tiro de caballos arrastra hasta la orilla un barco con varios hombres a bordo."

for epoch in range(num_epochs):
    print(f"\n[Epoch {epoch + 1} / {num_epochs}]")

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(model, sentence, spacy_es, spanish, english, device, max_length=50)
    print(f"Translated example sentence: \n {translated_sentence}")

    model.train()

    for batch_idx, (inp_data, target) in enumerate(train_loader):
        inp_data, target = inp_data.to(device), target.to(device)

        # Forward prop
        output = model(inp_data, target)

        # Reshape for loss function
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    if batch_idx % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item()}")

score = bleu(test_data[1:100], model, spanish, english, device)
print(f"Bleu score {score*100:.2f}")
