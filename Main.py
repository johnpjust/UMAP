import umap
import struct
import numpy as np
import matplotlib.pyplot as plt
import r_pca
import sklearn
import scipy.io
import pickle
import glob

############################## MNIST vs fashion MNIST #############################################
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

########################### CIFAR10 vs SVHN ##################################################
fnames_cifar = glob.glob(r'C:\Users\just\Downloads\public_datasets\cifar-10-python\cifar-10-batches-py\train\*')
cifar10=[np.load(f, allow_pickle=True, encoding='latin1') for f in fnames_cifar]
cifardata = np.concatenate([a['data'] for a in cifar10])
cifarlabels = np.expand_dims(np.concatenate([a['labels'] for a in cifar10]), axis=1)

svhn = scipy.io.loadmat(r'C:\Users\just\Downloads\public_datasets\SVHN.mat')
svhndata = np.moveaxis(svhn['X'],3,0)
svhndata = np.reshape(svhndata, (svhndata.shape[0],-1))
# svhnlabels = svhn['y']

data = np.concatenate([cifardata, svhndata], axis=0)
labels = np.concatenate([cifarlabels, svhn['y']+10], axis=0)

arr = np.arange(data.shape[0])
np.random.shuffle(arr)
data = data[arr,]
labels = labels[arr]

## UMAP
embedding = umap.UMAP(n_neighbors=10,
                      min_dist=0.1,
                      metric='correlation').fit_transform(data)

np.savetxt(r'C:\Users\just\Desktop\cifar_svhn_UMAP_embeddings.csv', np.concatenate([embedding,labels], axis=1), delimiter=',')

## PCA
pca=sklearn.decomposition.PCA(n_components=10)
pca.fit(data)
temp = pca.transform(data)

np.savetxt(r'C:\Users\just\Desktop\cifar_svhn_PCA.csv', np.concatenate([temp,labels], axis=1), delimiter=',')