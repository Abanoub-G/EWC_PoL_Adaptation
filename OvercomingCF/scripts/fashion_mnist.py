import numpy as np
from urllib import request
from os import path
from scipy.io import loadmat
import tarfile
import gzip
import pickle

# filename = [
#     ["n_mnist_awgn", "mnist-with-awgn.gz"],
#     ["n_mnist_motion", "mnist-with-motion-blur.gz"],
#     ["n_mnist_contrast", "mnist-with-reduced-contrast-and-awgn.gz"],
# ]
filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

filename2 = [
    ["training_images", "fashion_train-images-idx3-ubyte.gz"],
    ["test_images", "fashion_t10k-images-idx3-ubyte.gz"],
    ["training_labels", "fashion_train-labels-idx1-ubyte.gz"],
    ["test_labels", "fashion_t10k-labels-idx1-ubyte.gz"]
]


def download_file(url, filename):
    opener = request.URLopener()
    opener.addheader('User-Agent', 'Mozilla/5.0')
    opener.retrieve(url, filename)


def download_fashion_mnist():
    if path.exists("fashion_t10k-labels-idx1-ubyte.gz") and path.exists("fashion_t10k-images-idx3-ubyte.gz") and path.exists("fashion_train-labels-idx1-ubyte.gz") and path.exists("fashion_train-images-idx3-ubyte.gz"):
        print('Original files from website already downloaded!')
    else:
        try:
            base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

            for name in filename:
                print("Downloading " + name[1] + "...")
                print("Please wait for me... :)")
                download_file(base_url + name[1], "fashion_"+name[1])
            print("Download complete.")

        except:
            print("Can find files in website please check manually this website to download files: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com")

    # if path.exists("fashion_mnist_dataset/t10k-labels-idx1-ubyte.gz") and path.exists("fashion_mnist_dataset/t10k-images-idx3-ubyte.gz") and path.exists("fashion_mnist_dataset/train-labels-idx1-ubyte.gz") and path.exists("fashion_mnist_dataset/train-images-idx3-ubyte.gz"):
    #     print("N-MNIST dataset zip files are already unziped")

    # elif path.exists("mnist-with-awgn.gz") and path.exists("mnist-with-motion-blur.gz") and path.exists("mnist-with-reduced-contrast-and-awgn.gz"):
    #     print("Unzipping N-MNIST dataset files")
    #     opened_tar = tarfile.open("mnist-with-awgn.gz")
    #     opened_tar.extractall()

    #     opened_tar = tarfile.open("mnist-with-motion-blur.gz")
    #     opened_tar.extractall()

    #     opened_tar = tarfile.open("mnist-with-reduced-contrast-and-awgn.gz")
    #     opened_tar.extractall()

    #     print("Done unzipping N-MNIST dataset files")
    # else:
    #     print("Zipped N-MNIST dataset files do not exist, please download them.")


def save_fashion_mnist():
    fashion_mnist = {}
    for name in filename2[:2]:
        with gzip.open(name[1], 'rb') as f:
            tmp = np.frombuffer(f.read(), np.uint8, offset=16)
            fashion_mnist[name[0]] = tmp.reshape(-1, 1, 28, 28).astype(np.float32) / 255
    for name in filename2[-2:]:
        with gzip.open(name[1], 'rb') as f:
            fashion_mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("fashion_mnist.pkl", 'wb') as f:
        pickle.dump(fashion_mnist, f)
    print("Save complete.")

# def load(noise_type):  
#     if noise_type == "gn":   
#         data = loadmat("mnist-with-awgn.mat") # gausian noise

#     if noise_type == "blur": 
#         data = loadmat("mnist-with-motion-blur.mat")

#     if noise_type == "contrast":
#         data = loadmat("mnist-with-reduced-contrast-and-awgn.mat")

#     train_X = data["train_x"]
#     train_Y = data["train_y"]
#     test_X = data["test_x"]
#     test_Y = data["test_y"]

#     train_X = train_X.astype('float32')
#     train_X = train_X.reshape((train_X.shape[0], 1, 28, 28)) 
#     train_X = train_X / 255 # normalise data 
#     test_X = test_X.astype('float32')
#     test_X = test_X.reshape((test_X.shape[0], 1, 28, 28))
#     test_X = test_X / 255 # normalise data 

#     train_Y_temp = []
#     test_Y_temp = []

#     for j in range(len(train_Y)):
#         train_Y_temp.append(np.where(train_Y[j] == 1)[0][0])

#     for j in range(len(test_Y)):
#         test_Y_temp.append(np.where(test_Y[j] == 1)[0][0])

#     train_Y = np.array(train_Y_temp, dtype=np.uint8) 
#     test_Y = np.array(test_Y_temp, dtype=np.uint8) 

#     dataset = {}
#     dataset["training_images"] = train_X
#     dataset["training_labels"] = train_Y
#     dataset["testing_images"]  = test_X
#     dataset["testing_labels"]  = test_Y


#     return dataset["training_images"], dataset["training_labels"], \
#            dataset["testing_images"], dataset["testing_labels"],\
#            dataset

def init():
    # Check if already downloaded:
    if path.exists("fashion_mnist.pkl"):
        print('Files already downloaded!')
    else:  # Download Dataset
        download_fashion_mnist()
        save_fashion_mnist()

    # MNIST(path.join('data', 'fashion_mnist'), download=True)


def load():
    with open("fashion_mnist.pkl", 'rb') as f:
        fashion_mnist = pickle.load(f)

        print("fashion_mnist = ", fashion_mnist)
        dataset = {}
        dataset["training_images"] = fashion_mnist["training_images"]
        dataset["training_labels"] = fashion_mnist["training_labels"]
        dataset["testing_images"]  = fashion_mnist["test_images"]
        dataset["testing_labels"]  = fashion_mnist["test_labels"]
    return fashion_mnist["training_images"], fashion_mnist["training_labels"], \
           fashion_mnist["test_images"], fashion_mnist["test_labels"], dataset 


# def init():
#     download_n_mnist()


if __name__ == '__main__':
    init()