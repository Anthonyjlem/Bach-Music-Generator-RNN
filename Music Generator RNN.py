from pretty_midi_extra import piano_roll_to_pretty_midi
from torch import nn
import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F

# define the RNN

class BachRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        
        self.lstm = nn.LSTM(len(tokens), n_hidden, n_layers, dropout=0.5, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(n_hidden, len(tokens))
    
    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = x.contiguous().view(-1, self.n_hidden) # stack up LSTM outputs
        x = self.fc(x)
        return x, hidden
    
    def init_hidden(self, batch_size):
        # Create two new tensors n_layers x batch_size x n_hidden initialized to zero for hidden and cell state
        # of LSTM
        weight = next(self.parameters()).data
        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden

# define a function to generate batches

def get_batches(data, batch_size, seq_length):
    """Returns a generator that returns batches of size batch_size x seq_length from data."""
    batch_size_total = batch_size*seq_length
    n_batches = len(data)//batch_size_total # number of batches to make from data
    data = data[:n_batches*batch_size_total] # reduce data so can make whole number of batches
    data = data.reshape((batch_size, -1)) # reshape data to have batch_size rows
    # generate batch data, x, and labels
    for seq in range(0, data.shape[1], seq_length):
        x = data[:, seq:seq+seq_length]
        labels = np.zeros_like(x) # labels should be x but shifted one over in the future
        try:
            labels[:, :-1], labels[:, -1] = x[:, 1:], data[:, seq+seq_length]
        except IndexError:
            labels[:, :-1], labels[:, -1] = x[:, 1:], data[:, 0]
        yield x, labels

# define function to one-hot encode data

def one_hot_encode(data, n_labels):
    one_hot = np.zeros((data.size, n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), data.flatten()] = 1 # set correct columns in one_hot to 1
    one_hot = one_hot.reshape((*data.shape, n_labels)) # the * unrolls the list
    return one_hot

# define a function to make predictions

def predict(net, note, h=None, top_k=None):
    """Given a note, predicts a note. Returns the predicted note and hidden state."""
    x = np.array([[notes2int[note]]])
    x = one_hot_encode(x, len(unique_notes))
    data = torch.from_numpy(x)
    if train_on_gpu:
        data = data.cuda()
    h = tuple([each.data for each in h]) # detach hidden state from history
    out, h = net(data, h)
    # get probabilities of notes
    prob = F.softmax(out, dim=1).data
    if train_on_gpu:
        data = data.cuda()
    # get top probability notes
    if top_k is None:
        top_note = np.arange(len(unique_notes))
    else:
        prob, top_note = prob.topk(top_k)
        if train_on_gpu:
            top_note = top_note.cpu().numpy().squeeze()
        else:
            top_note = top_note.numpy().squeeze()
    if train_on_gpu:
        prob = prob.cpu().numpy().squeeze()
    else:
        prob = prob.numpy().squeeze()
    note = np.random.choice(top_note, p=prob/prob.sum())
    return int2notes[note], h    

# get data

raw_data = pretty_midi.PrettyMIDI("bach_sheep_may_safely_graze.mid")
piano_data = raw_data.instruments[0] # get piano data
data = piano_data.get_piano_roll()

# convert data into a dictionary

# find the start of the song
start = 0
# Go through each index of each note. If the sum in that index is 0, move on until the start of the song
for frame in range(len(data[0])):
    total = 0
    for note_data in data:
        total += note_data[frame]
    if total != 0:
        start = frame
        break
        
time_to_notes = {}
        
for frame in range(len(data[0][start:])):
    notes = []
    for note_index, note_data in enumerate(data):
        if note_data[frame] != 0:
            notes.append(note_index)
    # if the sum is 0, append e
    if np.sum(notes) == 0:
        time_to_notes[frame] = 'e' # e for empty
    # if sum is not 0, find notes that are played and add them into the dictionary of index:[notes played]
    else:
        time_to_notes[frame] = notes

# tokenize data

notes = []
for i in range(len(time_to_notes)):
    notes.append(time_to_notes[i])

unique_notes = tuple(set(map(tuple, notes))) # set of all unique note groupings
int2notes = dict(enumerate(unique_notes)) # dictionary to convert integers to note groupings
notes2int = {note: i for i, note in int2notes.items()} # dictionary to convert note groups to integers
encoded = np.array([notes2int[note] for note in map(tuple, notes)])

# check if GPU available

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("Training on GPU")
else:
    print("Training on CPU")

# instantiate the RNN

net = BachRNN(unique_notes)
if train_on_gpu:
    net.cuda()

# train the model

batch_size = 10
sequence_length = 50
epochs = 40

net.train()

opt = torch.optim.RMSprop(net.parameters(), lr=0.001) # RMSprop good for RNN
criterion = nn.CrossEntropyLoss()

val_idx = int(len(encoded)*0.9) # index for validation dataset
n_chars = len(unique_notes)
valid_loss_min = np.Inf

for e in range(epochs):
    h = net.init_hidden(batch_size) # initialize the hidden state
    data, val_data = encoded[:val_idx], encoded[val_idx:] # make validation and training datasets
    for x, y in get_batches(data, batch_size, sequence_length):
        # one-hot encode data and convert to tensors
        x = one_hot_encode(x, len(unique_notes))
        data, targets = torch.from_numpy(x), torch.from_numpy(y)
        if train_on_gpu:
            data, targets = data.cuda(), targets.cuda()
        h = tuple([each.data for each in h]) # create new variables for hidden state to prevent backprop through entire training history
        net.zero_grad()
        output, h = net(data, h)
        loss = criterion(output, targets.view(batch_size*sequence_length).long())
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5) # help prevent exploding gradient in RNNs and LSTMs
        opt.step()
        
    # calculate validation loss
    val_h = net.init_hidden(batch_size)
    val_losses = []
            
    net.eval()
            
    for x, y in get_batches(val_data, batch_size, sequence_length):
        # one-hot encode data and convert to tensors
        x = one_hot_encode(x, len(unique_notes))
        data, targets = torch.from_numpy(x), torch.from_numpy(y)
        if train_on_gpu:
            data, targets = data.cuda(), targets.cuda()
        val_h = tuple([each.data for each in val_h])
        output, val_h = net(data, val_h)
        val_loss = criterion(output, targets.view(batch_size*sequence_length).long())
        val_losses.append(val_loss.item())

    net.train()

    print("Epoch:", e+1, "\tLoss:", loss.item(), "\tVal Loss:", np.mean(val_losses))

    if np.mean(val_losses) < valid_loss_min:
        print("Saving model...")
        checkpoint = {"n_hidden": net.n_hidden,
                      "n_layers": net.n_layers,
                      "state_dict": net.state_dict(),
                      "tokens": unique_notes}
        with open("music_generator_rnn.net", "wb") as f:
            torch.save(checkpoint, f)
        valid_loss_min = val_loss

# load best RNN

with open("music_generator_rnn.net", "rb") as f:
    checkpoint = torch.load(f)
    
net = BachRNN(checkpoint["tokens"], n_hidden=checkpoint["n_hidden"], n_layers=checkpoint["n_layers"])
net.load_state_dict(checkpoint["state_dict"]) 

# generate a new song about 30 seconds long (about 3000 frames)

if train_on_gpu:
    net.cuda()

net.eval()

note_stability = 1 # determines the minimum duration of a note
song = [np.random.choice(unique_notes)]*note_stability
h = net.init_hidden(1)
for frame in range(3000):
    note, h = predict(net, song[-1], h, top_k=5)
    notes = [note]*note_stability
    song = song + notes

# convert song to piano roll

new_piano_roll = np.ndarray((128, 3000))

for frame in range(len(new_piano_roll[0])):
    for note_index in range(len(new_piano_roll)):
        if note_index in song[frame]:
            new_piano_roll[note_index][frame] = 100 # 100 for 100% volume
        else:
            new_piano_roll[note_index][frame] = 0 # note is not being played

# convert piano roll to PrettyMIDI object

song_pretty_midi = piano_roll_to_pretty_midi(new_piano_roll)

# export new song as a midi file

song_pretty_midi.write("new_song.mid")
