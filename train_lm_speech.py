#from lm import train
from lm_music import train

def main(job_id, params):
    print params
    validerr = train(
        saveto=params['model'][0],
        reload_=params['reload'][0],
        dim_word=params['dim_word'][0],
        dim=params['dim'][0],
        n_words=params['n-words'][0],
        decay_c=params['decay-c'][0],
        lrate=params['learning-rate'][0],
        optimizer=params['optimizer'][0],
        maxlen=200,
        batch_size=32,
        valid_batch_size=32,
        validFreq=100,
        dispFreq=10,
        saveFreq=100,
        sampleFreq=100,
        dataset='/data/lisatmp4/anirudhg/ptb/train.txt',
        valid_dataset='/data/lisatmp4/anirudhg/ptb/valid.txt',
        dictionary='/data/lisatmp4/anirudhg/ptb/dictionary_pbt.pkl',
        use_dropout=params['use-dropout'][0])
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['/Tmp/anirudhg/Speech/1459958720.model.npz'],
        'dim_word': [1024],
        'dim': [1024],
        'n-words': [8000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
