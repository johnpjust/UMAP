import umap
import struct
import numpy as np
import matplotlib.pyplot as plt
import r_pca
import sklearn
import scipy.io
import pickle
import glob

## MNIST vs fashion MNIST
fnames_data = [r'C:\Users\just\Downloads\public_datasets/MNIST/train-images.idx3-ubyte', r'C:\Users\just\Downloads\public_datasets/MNIST/t10k-images.idx3-ubyte', r'C:\Users\just\Downloads\public_datasets/FasionMNIST/train-images-idx3-ubyte', r'C:\Users\just\Downloads\public_datasets/FasionMNIST/t10k-images-idx3-ubyte']
fnames_labels = [r'C:\Users\just\Downloads\public_datasets/MNIST/train-labels.idx1-ubyte', r'C:\Users\just\Downloads\public_datasets/MNIST/t10k-labels.idx1-ubyte', r'C:\Users\just\Downloads\public_datasets/FasionMNIST/train-labels-idx1-ubyte', r'C:\Users\just\Downloads\public_datasets/FasionMNIST/t10k-labels-idx1-ubyte']

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

# data_ = data[np.arange(0,data.shape[0], 10)]
# data_=(data_ - data_.mean(axis=0))/data_.std(axis=0)
# ## robust PCA
# rpca = r_pca.R_pca((data_ - data_.mean(axis=0))/data_.std(axis=0))
# L, S = rpca.fit(max_iter=10000, iter_print=100)
# plt.figure();plt.scatter(L[:,0], L[:,1])
# plt.figure();plt.scatter(S[:,0], S[:,1])
# rpca.plot_fit([2,2])
# plt.show()

arr = np.arange(data.shape[0])
np.random.shuffle(arr)
data = data[arr,]
labels = labels[arr]
pca=sklearn.decomposition.PCA(n_components=10)
pca.fit(data)
temp = pca.transform(data)
# plt.figure();plt.scatter(temp[:,0], temp[:,1])

np.savetxt(r'C:\Users\just\Desktop\MNIST_Fashion_PCA.csv', np.concatenate([temp,np.expand_dims(labels,axis=1)], axis=1), delimiter=',')

fnames_cifar = glob.glob(r'C:\Users\just\Downloads\cifar-10-python.tar\cifar-10-python\cifar-10-batches-py\train')
## CIFAR10 vs SVHN
cifar10=[np.load(f, allow_pickle=True, encoding='latin1') for f in fnames_cifar]

mat = scipy.io.loadmat('file.mat')