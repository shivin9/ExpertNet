import numpy as np
import torch
from utils import DATA_DIR, BASE_DIR
from sklearn.model_selection import train_test_split 
import math

def pad_sents(sents, pad_token, N_FEATS=7, END_T=-1):
    j = 0
    sents_padded = []
    max_length = max([len(_) for _ in sents])
    for i in sents:
        padded = list(i) + [pad_token]*(max_length-len(i))
        padded = np.array(np.stack(padded, axis=0), dtype='float')
        padded = padded[:,:N_FEATS][:END_T]
        sents_padded.append(padded)
    return np.array(sents_padded, dtype='float')



def get_ts_datasets(args, r_state=0):
    DATASET = args.dataset
    train_x = np.load(DATA_DIR + '/' + DATASET + '/train.npy', allow_pickle=True)
    test_x = np.load(DATA_DIR + '/' + DATASET + '/test.npy', allow_pickle=True)

    train_y = np.load(DATA_DIR + '/' + DATASET + '/train_y.npy', allow_pickle=True)
    test_y = np.load(DATA_DIR + '/' + DATASET + '/test_y.npy', allow_pickle=True)

    train_x_len = np.load(DATA_DIR + '/' + DATASET + '/train_x_len.npy', allow_pickle=True)
    test_x_len = np.load(DATA_DIR + '/' + DATASET + '/test_x_len.npy', allow_pickle=True)

    X = np.hstack([train_x, test_x])
    y = np.hstack([train_y, test_y])
    lens = np.hstack([train_x_len, test_x_len])

    train_x, test_x, train_y, test_y, train_x_len, test_x_len = train_test_split(X, y, lens, random_state=r_state)
    train_x, dev_x, train_y, dev_y, train_x_len, dev_x_len = train_test_split(train_x, train_y, train_x_len, random_state=r_state)

    return (train_x, train_x_len, train_y), (dev_x, dev_x_len, dev_y), (test_x, test_x_len, test_y)


def batch_iter(x, y, lens, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(x) / batch_size) 
    index_array = list(range(len(x)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size] #  fetch out all the induces
        
        examples = []
        for idx in indices:
            examples.append((x[idx], y[idx],  lens[idx]))
       
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
    
        batch_x = [e[0] for e in examples]
        batch_y = [e[1] for e in examples]
#         batch_name = [e[2] for e in examples]
        batch_lens = [e[2] for e in examples]
       

        yield batch_x, batch_y, batch_lens

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask