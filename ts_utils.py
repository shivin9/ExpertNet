import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, label_binarize
from utils import DATA_DIR, BASE_DIR
from sklearn.model_selection import train_test_split 
from utils import color
from matplotlib import pyplot as plt
import math
import umap

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

    scale = StandardScaler()
    _ = scale.fit(np.nan_to_num(np.concatenate(train_x)))

    for idx in range(len(train_x)):
        train_x[idx] = torch.Tensor(scale.transform(np.nan_to_num(train_x[idx])))

    for idx in range(len(test_x)):
        test_x[idx] = torch.Tensor(scale.transform(np.nan_to_num(test_x[idx])))

    train_x, dev_x, train_y, dev_y, train_x_len, dev_x_len = train_test_split(train_x, train_y, train_x_len, random_state=r_state)

    return (train_x, train_x_len, train_y), (dev_x, dev_x_len, dev_y), (test_x, test_x_len, test_y), scale


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
        # batch_name = [e[2] for e in examples]
        batch_lens = [e[2] for e in examples]
       

        yield indices, batch_x, batch_y, batch_lens


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


def get_embeddings(model, X, args):
    batch_size = args.batch_size
    batch_num = math.ceil(len(X) / batch_size) 
    index_array = list(range(len(X)))
    z, q = [], []

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size] #  fetch out all the induces
        
        examples = []
        for idx in indices:
            examples.append(x[idx])
           
        batch_x = [e[0] for e in examples]
        q_batch, z_batch = model(torch.FloatTensor(batch_x).to(args.device), output="latent")
        z.append(z_batch.detach().numpy())
        q.append(q_batch.detach().numpy())

    return np.concatenate(z, axis=0), np.concatenate(q, axis=0)


def plot_data(X, y, cluster_ids, args, e):
    reducer = umap.UMAP(random_state=42)
    X2 = reducer.fit_transform(X)
    c_clusters = [color[int(cluster_ids[i])] for i in range(len(cluster_ids))]
    c_labels = [color[int(y[i])] for i in range(len(cluster_ids))]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Clusters vs Labels')
    ax1.scatter(X2[:,0], X2[:,1], color=c_clusters)
    ax2.scatter(X2[:,0], X2[:,1], color=c_labels)
    # fig.savefig(BASE_DIR + "/figures/" + args.dataset + "_e" + str(e) + ".png")
    plt.show()


def plot(z_train, q_train, y_train, args, X_test=None, y_test=None, labels=None, epoch=0):
    # idx = torch.Tensor(np.random.randint(0,len(X_train), int(0.1*len(X_train)))).type(torch.LongTensor).to(device)
    idx = range(int(0.2*len(z_train)))
    z_train = z_train[idx]
    y_train = y_train[idx]

    if labels is not None:
        cluster_id_train = labels[idx]
    else:
        cluster_id_train = torch.argmax(q_train[idx], axis=1)

    plot_data(z_train, y_train, cluster_id_train, args, epoch)
