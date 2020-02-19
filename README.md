# Bach-Music-Generator-RNN
This RNN was trained on "Sheep May Safely Graze," by J.S. Bach, and produces new 30 second long midi files.

The code in the module "pretty_midi_extra.py" is copied from https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py. In accordance with their citation policy, it is cited as follows: Colin Raffel and Daniel P. W. Ellis. Intuitive Analysis, Creation and Manipulation of MIDI Data with pretty_midi. In Proceedings of the 15th International Conference on Music Information Retrieval Late Breaking and Demo Papers, 2014.

## Notes
As the network is trained and has lower training and validation losses, it is apparent that it converges towards producing tunes very similar to those found in "Sheep May Safely Graze." It would be interesting to train the network on several pieces written by Bach to prevent it from simply replicating portions of "Sheep May Safely Graze."

An interesting hyperparameter I introduced was "note_stability," which determines the minimum duration of a note. This is static, and the problem is that in the music produced, while it prevents the song from rapidly changing, it also creates very long notes that do not change. In the future, it would be nice to train the RNN to learn note durations, or to adjust the parameter so that it only lengthens the note if it would have been changed too quickly. 

Useful tips for making this and RNNs in general (as found here: https://github.com/karpathy/char-rnn#tips-and-tricks):
* If your training loss is much lower than validation loss then this means the network might be overfitting. Solutions to this are to decrease your network size, or to increase dropout. For example you could try dropout of 0.5 and so on.
* If your training/validation loss are about equal then your model is underfitting. Increase the size of your model (either number of layers or the raw number of neurons per layer)
* I would advise that you always use n_layers of either 2/3

Examples for deciding number of factors (also from link above):
* I have a 100MB dataset and I'm using 150K parameters. My data size is significantly larger (100 mil >> 0.15 mil), so I expect to heavily underfit. I am thinking I can comfortably afford to make n_hidden larger.
* I have a 10MB dataset and running a 10 million parameter model. I'm slightly nervous and I'm carefully monitoring my validation loss. If it's larger than my training loss then I may want to try to increase dropout a bit and see if that helps the validation loss.
* In general, parameters and size of dataset should be of same order of magnitude 

## Dependencies
The program requires the following:
* Pretty Midi
* Numpy
* PyTorch
