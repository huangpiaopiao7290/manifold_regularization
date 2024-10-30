## @auther: pp
## @date: 2024/10/22
## @description:


import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy import sparse

import torch.nn as nn

class LapSVM(nn.Module):
    def __init__(self, options):
        super(LapSVM, self).__init__()
        self.options = options

        self.features_labeled = None

    def fit(self,features_labeled, labels, features_unlabeled):
        """

        :param features_labeled: 带标签数据的特征
        :param labels: 带标签数据的标签
        :param features_unlabeled: 无标签数据的特征
        """
        self.features_labeled = np.vstack([features_labeled, features_unlabeled])
        labels = np.diag(labels)

        # 计算邻接矩阵
        if self.options['neighbor_mode']=='connectivity':
            matrix_w = kneighbors_graph(self.X, self.opt['n_neighbor'], mode='connectivity',include_self=False)
            matrix_w = (((matrix_w + matrix_w.T) > 0) * 1)
        elif self.opt['neighbor_mode']=='distance':
            matrix_w = kneighbors_graph(self.X, self.opt['n_neighbor'], mode='distance',include_self=False)
            matrix_w = matrix_w.maximum(matrix_w.T)
            matrix_w = sparse.csr_matrix((np.exp(-matrix_w.data**2/4/self.opt['t']),
                                    matrix_w.indices,matrix_w.indptr),shape=(self.X.shape[0],
                                                                 self.X.shape[0]))
        else:
            raise Exception()


        # 计算Laplace matrix
        graph_laplace = sparse.diags(np.array(matrix_w.sum(0))[0]).tocsr() - matrix_w

        # Computing K with k(i,j) = kernel(i, j)
        matrix_kernel = self.options['kernel_function'](self.X,self.X,**self.opt['kernel_parameters'])

        labels_nums = features_labeled.shape[0]
        unlabeled_nums = features_unlabeled.shape[0]
        # Creating matrix J [I (labels_nums x labels_nums), 0 (l x (labels_nums + unlabeled_nums))]
        matrix_j = np.concatenate([np.identity(labels_nums),
                            np.zeros(labels_nums * unlabeled_nums).reshape(labels_nums, unlabeled_nums)],
                            axis=1)

        # Computing "almost" alpha
        almost_alpha = np.linalg.inv(2 * self.opt['gamma_A'] * np.identity(labels_nums + unlabeled_nums)
                                     + ((2 * self.opt['gamma_I']) / (labels_nums + unlabeled_nums) ** 2)
                                     * graph_laplace.dot(matrix_kernel)).dot(matrix_j.T).dot(labels)

        # Computing matrix q
        matrix_q = labels.dot(matrix_j).dot(matrix_kernel).dot(almost_alpha)
        matrix_q = (matrix_q+matrix_q.T) / 2

        # recycle memory
        del matrix_w, graph_laplace, matrix_kernel, matrix_j

