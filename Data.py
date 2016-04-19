#from fuel.datasets.youtube_audio import YouTubeAudio
import random
from audio_tools import soundsc
import matplotlib as mpl
from scipy.io import wavfile
mpl.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import scipy.io.wavfile as wave

class Data:

    def __init__(self, mb_size, seq_length, audio_path):
        self.mb_size = mb_size
        self.seq_length = seq_length

        #data = YouTubeAudio('XqaJ2Ol5cC4')
        #stream = data.get_example_stream()
        #it = stream.get_epoch_iterator()
        #seq = next(it)

        #self.total_seq_length = seq[0].shape[0]

        #self.seq = seq[0]

        #self.seq = wave.read('1k_rate.wav')[1]
        self.seq = wave.read(audio_path)[1]

        self.total_seq_length = self.seq.shape[0]

        number_bins = 8000
        self.bins = np.linspace(-8000.0, 8000.0, number_bins)

        #print self.total_seq_length
        #print self.seq.min()
        #print self.seq.max()

    '''
        Pick a random location, get sequence of seq_length
    '''
    def getExample(self):
        startingPoint = random.randint(0, self.total_seq_length - self.seq_length - 1)
        return self.seq[startingPoint : startingPoint + self.seq_length]

    def digitize(self, x):
        return np.digitize(x, self.bins) + 1

    def dedigitize(self, dig):
        return np.rint(self.bins[dig - 1]) - 1

    def getBatch(self):
        exampleLst = []
        for j in range(0, self.mb_size):
            exampleLst.append(self.getExample())
        return np.vstack(exampleLst)

    def saveExampleWav(self, x_gen, name):
        assert x_gen.ndim == 1
        #wave.write("plots/" + name + ".wav", 1000, x_gen)
        wavfile.write("plots2/" + name + ".wav", 1000, soundsc(x_gen))

    def saveExample(self, x_gen, name):

        assert x_gen.ndim == 1

        plt.plot(x_gen)

        imgLoc = "plots2/" + name + ".png"

        plt.ylim(-6000,6000)

        plt.savefig(imgLoc)

        plt.clf()

if __name__ == "__main__":
    d = Data(mb_size = 2, seq_length = 10)

    x = d.getBatch()

    print "original", x.tolist()
    print "new", d.digitize(x).tolist()
    print "recon", d.dedigitize(d.digitize(x)).tolist()

    print "diff", np.sum(np.square(d.dedigitize(d.digitize(x)) - x))


