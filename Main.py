import umap
import struct
import numpy as np
import matplotlib.pyplot as plt

fnames_data = ['data/MNIST/train-images.idx3-ubyte', 'data/MNIST/t10k-images.idx3-ubyte', 'data/FasionMNIST/train-images-idx3-ubyte', 'data/FasionMNIST/t10k-images-idx3-ubyte']
fnames_labels = ['data/MNIST/train-labels.idx1-ubyte', 'data/MNIST/t10k-labels.idx1-ubyte', 'data/FasionMNIST/train-labels-idx1-ubyte', 'data/FasionMNIST/t10k-labels-idx1-ubyte']

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

data = []
for f in fnames_data:
    data.append(read_idx(f))
data = np.concatenate(data)
data = data.reshape((data.shape[0],-1))

labels = []
for f in fnames_labels:
    labels.append(read_idx(f))
labels[2] = labels[2] + 10
labels[3] = labels[3] + 10
labels = np.concatenate(labels)

embedding = umap.UMAP(n_neighbors=10,
                      min_dist=0.1,
                      metric='correlation').fit_transform(data)

group = labels
group[group>9] = 10
group[group<10] = 0

group[group<10] = 0
group[group>9] = 1
fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(embedding[ix,0], embedding[ix,1], label = g, s = 100)
ax.legend()
plt.show()

np.savetxt(r'fitted_embeddings/nn10_md0.1_corr.csv', np.concatenate([embedding, np.expand_dims(labels,axis=1)],axis=1), delimiter=',')
