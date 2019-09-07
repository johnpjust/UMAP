import os
import json
import pprint
import datetime
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from bnaf import *
from optim.lr_scheduler import *
import glob
import random
import struct
import functools
import sklearn.decomposition
import scipy.stats


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def load_data_wPCA(args):
    fnames_data = [r'C:\Users\justjo\Downloads\public_datasets/MNIST/train-images.idx3-ubyte',
                   r'C:\Users\justjo\Downloads\public_datasets/MNIST/t10k-images.idx3-ubyte',
                   r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/train-images-idx3-ubyte',
                   r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/t10k-images-idx3-ubyte']
    # fnames_labels = [r'C:\Users\justjo\Downloads\public_datasets/MNIST/train-labels.idx1-ubyte',
    #                  r'C:\Users\justjo\Downloads\public_datasets/MNIST/t10k-labels.idx1-ubyte',
    #                  r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/train-labels-idx1-ubyte',
    #                  r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/t10k-labels-idx1-ubyte']

    data = []
    for f in fnames_data:
        data.append(read_idx(f))
    data = np.concatenate(data)
    data = data.reshape((data.shape[0], -1))

    # labels = []
    # for f in fnames_labels:
    #     labels.append(read_idx(f))
    # labels[2] = labels[2] + 10
    # labels[3] = labels[3] + 10
    # labels = np.concatenate(labels)

    fmnist_data = data[-70000:,:]
    # fmnist_labels = labels[-70000:]
    m = np.mean(fmnist_data)
    s = np.std(fmnist_data)
    arr = np.arange(fmnist_data.shape[0])
    np.random.shuffle(arr)
    fmnist_data = (fmnist_data[arr,] - m) / s
    # pca = sklearn.decomposition.PCA(n_components=fmnist_data.shape[1])
    pca = sklearn.decomposition.PCA(n_components=args.n_comp_pca)
    pca.fit(data)

    datatrain = pca.transform(fmnist_data)
    mpca = np.mean(datatrain, axis=0)
    spca = np.std(datatrain, axis=0)

    datatrain = (datatrain-mpca)/spca
    # dist = scipy.stats.johnsonsu.fit(np.random.choice(np.concatenate(datatrain), size=1000000, replace=False))
    data_all = pca.transform((data-m)/s)
    data_all = (data_all - mpca)/spca

    if args.johnsonsu:
        dist = (-0.001832479991053192, 1.4168126381734334, -0.0031200079512223155, 1.0809622235738585) ##1MM samples
        datatrain = np.arcsinh((datatrain - dist[-2]) / dist[-1]) * dist[1] + dist[0]
        data_all = np.arcsinh((data_all - dist[-2]) / dist[-1]) * dist[1] + dist[0]

    return data_all, datatrain


def img_preprocessing(img):
    return tf.minimum(tf.maximum(img + tf.random.uniform(img.shape, -0.00390625, 0.00390625),-1), 1) ## add noise

def img_preprocessing2(img):
    tf.random.set_seed(None)
    return img + tf.random.normal(img.shape, 0, 0.005) ## add noise -- based on stdev of 1 for all features

def load_dataset(args):

    tf.random.set_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    if args.train == 'pca':
        data_test, data_train = load_data_wPCA(args)
        data_val = data_train[-10000:,]
        data_train = data_train[:-10000,:]
    elif args.train == 'train':
        fnames_data = [r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/train-images-idx3-ubyte',
                   r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/t10k-images-idx3-ubyte']
        fnames_test = [r'C:\Users\justjo\Downloads\public_datasets/MNIST/train-images.idx3-ubyte',
                   r'C:\Users\justjo\Downloads\public_datasets/MNIST/t10k-images.idx3-ubyte']
        data_train = read_idx(fnames_data[0])
        data_train = data_train.reshape((data_train.shape[0], -1))/128 - 1

        data_val = read_idx(fnames_data[1])
        data_val = data_val.reshape((data_val.shape[0], -1))/128 - 1
        data_test = np.concatenate([read_idx(fnames_test[0]), read_idx(fnames_test[1])])
        data_test = data_test.reshape((data_test.shape[0], -1)) / 128 - 1
    else:
        fnames_data = ['data/MNIST/train-images.idx3-ubyte', 'data/MNIST/t10k-images.idx3-ubyte', 'data/FasionMNIST/train-images-idx3-ubyte', 'data/FasionMNIST/t10k-images-idx3-ubyte']
        data = []
        for f in fnames_data:
            data.append(read_idx(f))
        data = np.concatenate(data)
        data = data.reshape((data.shape[0], -1))/128 - 1
        data_train = data
        data_val = []
        data_test = []

    if args.add_noise:
        img_preprocessing_ = img_preprocessing2
    else:
        img_preprocessing_ = img_preprocessing

    dataset_train = tf.data.Dataset.from_tensor_slices(tf.constant(data_train, dtype=tf.float32))#.float().to(args.device)
    # dataset_train = dataset_train.shuffle(buffer_size=data_train.shape[0]).map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)
    dataset_train = dataset_train.shuffle(buffer_size=data_train.shape[0]).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)

    dataset_valid = tf.data.Dataset.from_tensor_slices(tf.constant(data_val, dtype=tf.float32))#.float().to(args.device)
    # dataset_valid = dataset_valid.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)
    dataset_valid = dataset_valid.batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)

    dataset_test = tf.data.Dataset.from_tensor_slices(tf.constant(data_test, dtype=tf.float32))#.float().to(args.device)
    dataset_test = dataset_test.batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)

    args.n_dims = data_train.shape[1]

    return dataset_train, dataset_valid, dataset_test

def create_model(args, verbose=False):

    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)

    tf.random.set_seed(args.manualSeedw)
    np.random.seed(args.manualSeedw)

    dtype_in = tf.float32

    g_constraint = lambda x: tf.nn.relu(x) + 1e-6 ## for batch norm
    flows = []
    for f in range(args.flows):
        #build internal layers for a single flow
        layers = []
        for _ in range(args.layers - 1):
            layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
                                       args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in))
            layers.append(Tanh(dtype_in=dtype_in))

        flows.append(
            BNAF(layers = [MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in), Tanh(dtype_in=dtype_in)] + \
               layers + \
               [MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims, dtype_in=dtype_in)], \
             res=args.residual if f < args.flows - 1 else None, dtype_in= dtype_in
             )
        )
        ## with batch norm example
        # for _ in range(args.layers - 1):
        #     layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
        #                                args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in))
        #     layers.append(CustomBatchnorm(gamma_constraint = g_constraint, momentum=args.momentum))
        #     layers.append(Tanh(dtype_in=dtype_in))
        #
        # flows.append(
        #     BNAF(layers = [MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in), CustomBatchnorm(gamma_constraint = g_constraint, momentum=args.momentum), Tanh(dtype_in=dtype_in)] + \
        #        layers + \
        #        [CustomBatchnorm(scale=False, momentum=args.momentum), MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims, dtype_in=dtype_in)], \
        #      res=args.residual if f < args.flows - 1 else None, dtype_in= dtype_in
        #      )
        # )

        if f < args.flows - 1:
            flows.append(Permutation(args.n_dims, 'flip'))

        model = Sequential(flows)#, dtype_in=dtype_in)
        # params = np.sum(np.sum(p.numpy() != 0) if len(p.numpy().shape) > 1 else p.numpy().shape
        #              for p in model.trainable_variables)[0]
    
    # if verbose:
    #     print('{}'.format(model))
    #     print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params,
    #         NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims))

    # if args.save and not args.load:
    #     with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
    #         print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params,
    #             NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims), file=f)
    
    return model

def load_model(args, root, load_start_epoch=False):
    # def f():
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))
    # root.restore(os.path.join(args.load or args.path, 'checkpoint'))
    # if load_start_epoch:
    #     args.start_epoch = tf.train.get_global_step().numpy()
    # return f

# @tf.function
def compute_log_p_x(model, x_mb):
    ## use tf.gradient + tf.convert_to_tensor + tf.GradientTape(persistent=True) to clean up garbage implementation in bnaf.py
    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = tf.reduce_sum(tfp.distributions.Normal(tf.zeros_like(y_mb), tf.ones_like(y_mb)).log_prob(y_mb), axis=-1)#.sum(-1)
    return log_p_y_mb + log_diag_j_mb

# @tf.function
def train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, args):
    
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        # t = tqdm(data_loader_train, smoothing=0, ncols=80)
        train_loss = []
        
        for x_mb in data_loader_train:
            with tf.GradientTape() as tape:
                loss = - tf.reduce_mean(compute_log_p_x(model, x_mb)/args.n_dims) # + args.regL2*tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in model.trainable_weights])) #negative -> minimize to maximize liklihood
                # loss = -tfp.stats.percentile(compute_log_p_x(model, x_mb), 50)
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss.append(loss)
            # if args.tensorboard:
            #     tf.summary.scalar('loss/train', loss, tf.compat.v1.train.get_global_step())


        train_loss = np.mean(train_loss)
        # train_loss = np.median(train_loss)
        validation_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb)/args.n_dims) for x_mb in data_loader_valid])
        test_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb) / args.n_dims) for x_mb in data_loader_test])
        # validation_loss = - np.median([np.median(compute_log_p_x(model, x_mb)) for x_mb in data_loader_valid])


        # print('Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f}'.format(
        #     epoch + 1, args.start_epoch + args.epochs, train_loss, validation_loss))


        stop = scheduler.on_epoch_end(epoch = epoch, monitor=validation_loss)

        tf.compat.v1.train.get_global_step().assign_add(1)
        if args.tensorboard:
            tf.summary.scalar('loss/validation', validation_loss,tf.compat.v1.train.get_global_step())
            tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
            tf.summary.scalar('loss/test', test_loss, tf.compat.v1.train.get_global_step())


        if stop:
            break

    validation_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb)/args.n_dims) for x_mb in data_loader_valid])
    test_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb)/args.n_dims) for x_mb in data_loader_test])

    # validation_loss = - np.median([np.median(compute_log_p_x(model, x_mb)) for x_mb in data_loader_valid])
    # test_loss = - np.median([np.median(compute_log_p_x(model, x_mb)) for x_mb in data_loader_test])

    print('###### Stop training after {} epochs!'.format(epoch + 1))
    print('Validation loss: {:4.3f}'.format(validation_loss))
    print('Test loss:       {:4.3f}'.format(test_loss))
    
    if args.save:
        with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
            print('###### Stop training after {} epochs!'.format(epoch + 1), file=f)
            print('Validation loss: {:4.3f}'.format(validation_loss), file=f)
            print('Test loss:       {:4.3f}'.format(test_loss), file=f)

class parser_:
    pass

def main():
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    # tf.compat.v1.enable_eager_execution(config=config)

    # tf.config.experimental_run_functions_eagerly(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    args = parser_()
    args.device = '/gpu:0'  # '/gpu:0'
    args.dataset = 'corn'  # 'gq_ms_wheat_johnson'#'gq_ms_wheat_johnson' #['gas', 'bsds300', 'hepmass', 'miniboone', 'power']
    args.learning_rate = np.float32(1e-2)
    args.batch_dim = 100
    args.clip_norm = 0.1
    args.epochs = 5000
    args.patience = 10
    args.cooldown = 10
    args.decay = 0.5
    args.min_lr = 5e-4
    args.flows = 6
    args.layers = 1
    args.hidden_dim = 12
    args.residual = 'gated'
    args.expname = ''
    args.load = ''#r'checkpoint/corn_layers1_h12_flows6_gated_2019-09-05-00-08-31'
    args.save = True
    args.tensorboard = 'tensorboard'
    args.early_stopping = 15
    args.maxiter = 5000
    args.factr = 1E1
    args.regL2 = 0.0001
    args.regL1 = -1
    args.manualSeed = None
    args.manualSeedw = None
    args.momentum = 0.9  ## batch norm momentum
    args.prefetch_size = 1  # data pipeline prefetch buffer size
    args.parallel = 16  # data pipeline parallel processes
    args.train = 'train'  # 'train'
    args.johnsonsu = True
    args.n_comp_pca = 350
    args.add_noise = True

    args.path = os.path.join('checkpoint', '{}{}_layers{}_h{}_flows{}{}_{}'.format(
        args.expname + ('_' if args.expname != '' else ''),
        args.dataset, args.layers, args.hidden_dim, args.flows, '_' + args.residual if args.residual else '',
        str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

    print('Loading dataset..')

    data_loader_train, data_loader_valid, data_loader_test = load_dataset(args)

    if args.save and not args.load:
        print('Creating directory experiment..')
        os.mkdir(args.path)
        with open(os.path.join(args.path, 'args.json'), 'w') as f:
            json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

    print('Creating BNAF model..')
    with tf.device(args.device):
        model = create_model(args, verbose=True)

    ### debug
    # data_loader_train_ = tf.contrib.eager.Iterator(data_loader_train)
    # x = data_loader_train_.get_next()
    # a = model(x)

    ## tensorboard and saving
    writer = tf.summary.create_file_writer(os.path.join(args.tensorboard, args.load or args.path))
    writer.set_as_default()

    tf.compat.v1.train.get_or_create_global_step()

    global_step = tf.compat.v1.train.get_global_step()
    global_step.assign(0)

    root = None
    args.start_epoch = 0

    print('Creating optimizer..')
    with tf.device(args.device):
        optimizer = tf.optimizers.Adam()
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.compat.v1.train.get_global_step())

    if args.load:
        load_model(args, root, load_start_epoch=True)

    print('Creating scheduler..')
    # use baseline to avoid saving early on
    scheduler = EarlyStopping(model=model, patience=args.early_stopping, args=args, root=root)

    with tf.device(args.device):
        train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, args)

if __name__ == '__main__':
    main()

##"C:\Program Files\Git\bin\sh.exe" --login -i

#### tensorboard --logdir=C:\Users\justjo\PycharmProjects\UMAP\tensorboard\checkpoint
## http://localhost:6006/

