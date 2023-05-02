
import torch
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import orthogonal_mp_gram
import numpy as np
import scipy as sp

class ApproximateKSVD(object):
    def __init__(self, n_components, max_iter=2, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements
        max_iter:
            Maximum number of iterations
        tol:
            tolerance for error
        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target
        """
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_non_zero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            I = gamma[:, j] > 0
            if torch.sum(I) == 0:
                continue

            D[j, :] = 0
            g = gamma[I, j].unsqueeze(dim=1)
            r = X[I, :] - torch.mm(gamma[I, :],D)
            d = torch.mm(g.T,r)
            d /= torch.norm(d)
            g = torch.mm(r,d.T)
            D[j, :] = d
            gamma[I, j] = g.T

        return D, gamma

    def _initialize(self, X):
        if min(X.shape) < self.n_components:
                D = torch.rand(self.n_components, X.shape[1])
        else:
            X = X.numpy()
            u, s, vt = sp.sparse.linalg.svds(X.T, k=self.n_components)
            D = np.dot(np.diag(s), vt)
            D = torch.tensor(D)
        D /= torch.norm(D, dim=1, keepdim=True)
        return  D

    def _transform(self, D, X):
        gram = torch.mm(D,D.T)
        Xy = torch.mm(D,(X.T))

        n_nonzero_coefs = self.transform_non_zero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])


        return torch.tensor(orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T)

    def fit(self, X):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        D = self._initialize(X)
        for i in tqdm(range(self.max_iter)):
            gamma = self._transform(D, X)
            e = torch.norm(X - torch.mm(gamma, D))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self

    def transform(self, X):
        return self._transform(self.components_, X)