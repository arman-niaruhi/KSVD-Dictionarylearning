import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import orthogonal_mp_gram
from matplotlib import pyplot as plt
import cv2
from net import ApproximateKSVD
import math
import time




'''
    Choose a set of training images that are similar to the noisy image you want to denoise.
    Extract small image patches from the training images.
    Use CDL to learn a dictionary of filters that can efficiently represent the patches in a sparse manner.
    Apply the learned dictionary to the noisy image patches to obtain a sparse representation of each patch.
    Threshold the sparse representations to remove noise.
    Reconstruct the denoised image by applying the learned dictionary to the thresholded sparse representations.

Here are more detailed steps:

    Load the noisy image and convert it to grayscale if it is in color.
    Choose the size of the patches you want to use for training the CDL algorithm. A common patch size is 8x8 pixels.
    Extract small patches from the training images and stack them into a matrix X.
    Normalize each patch to have zero mean and unit variance.
    Use CDL to learn a dictionary D of filters that can be used to efficiently represent the patches in a sparse manner.
                        This can be done using algorithms such as K-SVD or online dictionary learning.
    Extract patches from the noisy image and stack them into a matrix Y.
    Normalize each patch in Y to have zero mean and unit variance.
    Use the learned dictionary D to obtain a sparse representation of each patch in Y by solving an optimization problem that seeks to minimize 
            the difference between the patch and its representation as a linear combination of the filters in D subject to a sparsity constraint.
    Threshold the sparse representations to remove noise by setting to zero the coefficients that are below a certain threshold. 
                A common thresholding method is the soft-thresholding operator.
    Reconstruct the denoised image by applying the learned dictionary D to the thresholded sparse representations.

'''


class Dictionary_Learning():

    # Feed the input and do some preprocessing steps
    def input_image(self,img):
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # Load the noisy image
        image = F.to_tensor(img).unsqueeze(0).type(dtype)
        # image = F.to_tensor(Image.open('arman.jpg')).unsqueeze(0).type(dtype)
        return image


    # Patchify the image to the patches
    def patchify_the_image(self, image, patch_size = 8):

        # Extract patches from the image
        patches = torch.nn.functional.unfold(image, kernel_size = patch_size, stride = patch_size)

        # Normalize the patches
        eps = 1e-6
        mean = torch.mean(patches, dim=1, keepdim=True)
        std = torch.std(patches, dim=1, keepdim=True)
        patches = (patches - mean) / (std + eps)

        return patches, mean, std



    # Image reconstruction
    def image_reconstruction(self, reconstructed_patches, std, mean, patch_size, image, eps= 1e-6):

        # Unnormalize the image
        reconstructed_patches = reconstructed_patches * (std+eps) + mean
        reconstructed_patches = torch.clamp(reconstructed_patches, 0, 1)


        reconstructed_patches = reconstructed_patches.view(1, patch_size*patch_size,-1)
        output_size = (image.shape[2], image.shape[3])

        reconstructed_image = torch.nn.functional.fold(reconstructed_patches, output_size = output_size ,kernel_size=patch_size, stride=patch_size)

        return reconstructed_image


    # Display the original and reconstructed images
    def plot_images(self, image, reconstructed_image, number_of_atoms):

        img_original = image.squeeze().numpy() * 255
        img_reconstructed = reconstructed_image.squeeze().numpy() * 255


        psnr = cv2.PSNR(img_original, img_reconstructed)
        # print(f"'psnr' between original image and reconstructed image: {psnr} ")

        fig = plt.figure(figsize=(10, 7))
        # ax = fig.subplots()
        # setting values to rows and column variables
        rows = 1
        columns = 2

        # Adds a subplot at the 1st position 
        fig.add_subplot(rows, columns, 1)
        # showing image
        plt.imshow(img_original, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title("Original Image")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)
        
        # showing image
        plt.imshow(img_reconstructed, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title("Reconstructed Image")
        end_time = time.time()

        runtime =  end_time - start_time
        fig.text(0.4, 0.80, f"PSNR : {str(psnr)}" )
        fig.text(0.4, 0.77, f"Time : {str(runtime)}")
        fig.text(0.4, 0.74, f"Dictionary size = {str(number_of_atoms)}" )

        # ax.text(2, 6, psnr, fontsize=15)
        fig.savefig(f'{number_of_atoms} atoms_sample.jpg',bbox_inches='tight', dpi=150)
        plt.show()


    # Entire process in just on function
    def sparse_coding_with_ksvd(self, image, number_of_atoms = 100, patch_size = 8, num_iter_for_Dictionary = 10):
        global start_time 
        start_time = time.time()
        x, y = image.size
        ratio = x/y
        image = image.resize((math.floor(400*ratio), 400))
        image = self.input_image(image)
        patches, mean, std = self.patchify_the_image(image=image, patch_size=patch_size)
        # D = torch.randn(number_of_atoms, (patch_size**2))
        patches = patches.squeeze().T

        D_size = patch_size*patch_size

        dic = ApproximateKSVD(n_components=65, transform_n_nonzero_coefs=D_size)
        dic.fit(patches)
        gamma = dic.transform(patches)
        D = dic.components_
        reconstructed_patches = torch.mm(gamma,dic.components_).T
    
        reconstructed_image = self.image_reconstruction(reconstructed_patches=reconstructed_patches, std=std,mean=mean, patch_size=patch_size, image=image)

        # Call the image saver methode
        self.plot_images(image=image, reconstructed_image=reconstructed_image, number_of_atoms = number_of_atoms)
        


def main():
    NUMBER_OF_ATOMS = 100
    PATCH_SIZE = 8
    NUM_ITERATION_FOR_DICTIONARY = 10

    # Give input image and for preprocessing and other steps
    IMAGE = Image.open('Untitled3.jpeg').convert('L')

    # Learning the dictionary 
    DL = Dictionary_Learning()

    # Using the Dictionary from previous step and try to fit the best sparse representation
    DL.sparse_coding_with_ksvd(image = IMAGE, number_of_atoms = NUMBER_OF_ATOMS,
                                patch_size = PATCH_SIZE, num_iter_for_Dictionary = NUM_ITERATION_FOR_DICTIONARY)


if __name__ == "__main__":
    main()



    



