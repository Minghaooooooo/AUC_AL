import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel as RBF, cosine_similarity
from sklearn.cluster import KMeans
import random
import time
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.metrics import jaccard_score
from scipy import stats

class dpmeans:

    def __init__(self, X):
        # Initialize parameters for DP means

        self.K = 1
        self.K_init = 10
        self.d = X.shape[1]
        self.z = np.mod(np.random.permutation(X.shape[0]), self.K) + 1
        self.mu = np.random.standard_normal((self.K, self.d))
        self.sigma = 1
        self.nk = np.zeros(self.K)
        self.pik = np.ones(self.K) / self.K

        # init mu
        self.mu = np.array([np.mean(X, 0)])

        # init lambda
        self.Lambda = self.kpp_init(X, self.K_init)

        self.max_iter = 100
        self.obj = np.zeros(self.max_iter)
        self.em_time = np.zeros(self.max_iter)

    def kpp_init(self, X, k):
        # k++ init
        # lambda is max distance to k++ means

        [n, d] = np.shape(X)
        mu = np.zeros((k, d))
        dist = np.inf * np.ones(n)

        mu[0, :] = X[np.ceil(np.random.rand() * n - 1).astype(int), :]
        for i in range(1, k):
            D = X - np.tile(mu[i - 1, :], (n, 1))
            dist = np.minimum(dist, np.sum(D * D, 1))
            idx = np.where(np.random.rand() < np.cumsum(dist / float(sum(dist))))
            mu[i, :] = X[idx[0][0], :]
            Lambda = np.max(dist)

        return Lambda

    def fit(self, X):

        obj_tol = 1e-3
        max_iter = self.max_iter
        [n, d] = np.shape(X)

        obj = np.zeros(max_iter)
        em_time = np.zeros(max_iter)
        print('running dpmeans...')

        for iter in range(max_iter):
            tic = time.time()
            dist = np.zeros((n, self.K))

            # assignment step
            for kk in range(self.K):
                Xm = X - np.tile(self.mu[kk, :], (n, 1))
                dist[:, kk] = np.sum(Xm * Xm, 1)

            # update labels
            dmin = np.min(dist, 1)
            self.z = np.argmin(dist, 1)
            idx = np.where(dmin > self.Lambda)

            if (np.size(idx) > 0):
                self.K = self.K + 1
                self.z[idx[0]] = self.K - 1  # cluster labels in [0,...,K-1]
                self.mu = np.vstack([self.mu, np.mean(X[idx[0], :], 0)])
                Xm = X - np.tile(self.mu[self.K - 1, :], (n, 1))
                dist = np.hstack([dist, np.array([np.sum(Xm * Xm, 1)]).T])

            # update step
            self.nk = np.zeros(self.K)
            for kk in range(self.K):
                self.nk[kk] = self.z.tolist().count(kk)
                idx = np.where(self.z == kk)
                self.mu[kk, :] = np.mean(X[idx[0], :], 0)

            self.pik = self.nk / float(np.sum(self.nk))

            # compute objective
            for kk in range(self.K):
                idx = np.where(self.z == kk)
                obj[iter] = obj[iter] + np.sum(dist[idx[0], kk], 0)
            obj[iter] = obj[iter] + self.Lambda * self.K

            # check convergence
            if (iter > 0 and np.abs(obj[iter] - obj[iter - 1]) < obj_tol * obj[iter]):
                print('converged in %d iterations\n' % iter)
                break
            em_time[iter] = time.time() - tic
        # end for
        self.obj = obj
        self.em_time = em_time
        return self.z, obj, em_time

def random_sampling(self, probs, n_instances):
    # random select
    total_instances = len(self.pool_dataset)
    indices = list(range(total_instances))
    random.shuffle(indices)
    selected_indices = indices[:n_instances]
    return selected_indices


def multiple_margin(self, probs, n_instances):
    # multiple margin
    # Calculate the number of positive labels in new_training_data
    avg_num_pos_labels = torch.mean(torch.sum(self.train_dataset.y, dim=1)).item()
    print(avg_num_pos_labels)
    # Calculate the top m margins for each instance0
    margins, _ = torch.topk(probs, int(avg_num_pos_labels), dim=1)
    margins_diff = margins[:, :-1] - margins[:, 1:]
    # Calculate the mean margin across labels for each instance
    xi = int(avg_num_pos_labels / 2)
    # Create a tensor representing the indices of margins_diff
    indices = torch.arange(margins_diff.size(1), dtype=torch.float).unsqueeze(0)
    # Compute the power term for each element
    power_term = xi - indices
    # Compute margins_diff multiplied by e to the power of (xi - index)
    margins_diff_exp = margins_diff * torch.exp(power_term)
    print('margin_diff_exp.shape', margins_diff_exp.shape)
    sum_margin = torch.sum(margins_diff_exp, dim=1)
    print('mean_margin', sum_margin.shape)
    # Select the instances with the highest mean margin
    _, selected_indices = torch.topk(sum_margin, n_instances, largest=False)
    # _, selected_indices = torch.topk(margin, n_instances, largest=True)
    return selected_indices


def test11(self, probs, n_instances):

    ctraining = self.train_origin
    ctraining1 = np.array(ctraining)
    ctraining2 = ctraining1 * 10
    cpredit = self.pool_origin
    cpredit1 = np.array(cpredit)
    cpredit2 = cpredit1 * 10
    print(cpredit2.shape)
    cresult = RBF(cpredit2, ctraining2)
    cresult2 = np.sum(cresult, axis=1)
    # cresult2 = np.argmin(cresult, axis=1)
    # np.savetxt('cresult2.csv', cresult2, delimiter=',', fmt='%f')

    cresult3 = torch.tensor(cresult2)
    a = cresult3.size()
    a = a[0]
    value21, cresult5 = torch.topk(cresult3, k=int(a/20), largest=False)
    value22, cresult6 = torch.topk(cresult3, k=int(a /20), largest=True)
    interval1=value21[-1]
    interval1=interval1.numpy()
    interval2 = value22[-1]
    interval2=interval2.numpy()
    #cresult6 = np.array(cresult5)
    #variance,mean=torch.var_mean(cresult3, dim=0, keepdim=True)
    #interval = stats.norm.interval(0.8, mean, variance)
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    #temp31 = torch.sum(temp1, dim=1)
    #a = temp31.size()
    #a = a[0]
    #temp21 = temp2 / a
    #testmin = torch.argmin(temp21)
    #print(f"Min index in pooling {testmin}")
    _, collum = torch.topk(temp2, k=1, largest=False)
    c = collum
    c = c.cpu()
    c = c.numpy()
    c1 = c[0]
    print(f"Min label in pooling {c1}")
    value, row1 = torch.sort(temp1[:, c1], dim=0)
    size2=row1.size()
    size2 = size2[0]
    list=[]
    for i in range(size2):
        indextemp1=row1[i]
        cvalue1=cresult3[indextemp1]
        cvalue2=cvalue1.numpy()
        if cvalue2 > interval1 :
            indextemp2=indextemp1.cpu()
            indextemp3=indextemp2.numpy()
            list.append(indextemp3)
            if len(list)>=n_instances:
                print("selection breaking ")
                break
        else:
            print(f"not selected {indextemp1} value {cvalue2}")
    list1=np.array(list)
    print("cc")
    return list1


def test13(self, probs, n_instances):

    ctraining = self.train_origin
    ctraining1 = np.array(ctraining)
    ctraining2 = ctraining1 * 10
    cpredit = self.pool_origin
    cpredit1 = np.array(cpredit)
    cpredit2 = cpredit1 * 10

    cresult = RBF(cpredit2, ctraining2)
    cresult2 = np.sum(cresult, axis=1)
    cresult3 = torch.tensor(cresult2)
    a = cresult3.size()
    a = a[0]
    value21, cresult5 = torch.topk(cresult3, k=int(a/5), largest=True)
    value22, cresult6 = torch.topk(cresult3, k=int(a /5), largest=True)
    interval1=value21[-1]
    interval1=interval1.numpy()
    interval2 = value22[-1]
    interval2=interval2.numpy()
    #cresult6 = np.array(cresult5)
    #variance,mean=torch.var_mean(cresult3, dim=0, keepdim=True)
    #interval = stats.norm.interval(0.8, mean, variance)
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    #temp31 = torch.sum(temp1, dim=1)
    #a = temp31.size()
    #a = a[0]
    #temp21 = temp2 / a
    #testmin = torch.argmin(temp21)
    #print(f"Min index in pooling {testmin}")
    _, collum = torch.topk(temp2, k=1, largest=False)
    c = collum
    c = c.cpu()
    c = c.numpy()
    c1 = c[0]
    print(f"Min label in pooling {c1}")
    value, row1 = torch.sort(temp1[:, c1], dim=0)
    size2=row1.size()
    size2 = size2[0]
    list=[]
    for i in range(size2):
        indextemp1=row1[i]
        cvalue1=cresult3[indextemp1]
        cvalue2=cvalue1.numpy()
        if cvalue2 < interval1 :
            indextemp2=indextemp1.cpu()
            indextemp3=indextemp2.numpy()
            list.append(indextemp3)
            if len(list)>=n_instances:
                print("selection breaking ")
                break
        else:
            print(f"not selected {indextemp1} value {cvalue2}")
    list1=np.array(list)
    print("cc")
    return list1


def test15(self, probs, n_instances):

    ctraining = self.train_origin
    ctraining1 = np.array(ctraining)
    ctraining2 = ctraining1 * 10
    cpredit = self.pool_origin
    cpredit1 = np.array(cpredit)
    cpredit2 = cpredit1 * 10

    cresult = cosine_similarity(cpredit2, ctraining2)
    cresult2 = np.max(cresult, axis=1)
    cresult3 = torch.tensor(cresult2)
    a = cresult3.size()
    a = a[0]
    value21, cresult5 = torch.topk(cresult3, k=int(a/20), largest=True)
    value22, cresult6 = torch.topk(cresult3, k=int(a /20), largest=True)
    interval1=value21[-1]
    interval1=interval1.numpy()
    interval2 = value22[-1]
    interval2=interval2.numpy()
    #cresult6 = np.array(cresult5)
    #variance,mean=torch.var_mean(cresult3, dim=0, keepdim=True)
    #interval = stats.norm.interval(0.8, mean, variance)
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    #temp31 = torch.sum(temp1, dim=1)
    #a = temp31.size()
    #a = a[0]
    #temp21 = temp2 / a
    #testmin = torch.argmin(temp21)
    #print(f"Min index in pooling {testmin}")
    _, collum = torch.topk(temp2, k=1, largest=False)
    c = collum
    c = c.cpu()
    c = c.numpy()
    c1 = c[0]
    print(f"Min label in pooling {c1}")
    value, row1 = torch.sort(temp1[:, c1], dim=0)
    size2=row1.size()
    size2 = size2[0]
    list=[]
    for i in range(size2):
        indextemp1=row1[i]
        cvalue1=cresult3[indextemp1]
        cvalue2=cvalue1.numpy()
        if cvalue2 < interval1 :
            indextemp2=indextemp1.cpu()
            indextemp3=indextemp2.numpy()
            list.append(indextemp3)
            if len(list)>=n_instances:
                print("selection breaking ")
                break
        else:
            print(f"not selected {indextemp1} value {cvalue2}")
    list1=np.array(list)
    print("cc")
    return list1


def test17(self, probs, n_instances): #label kmeans and average
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    temp31 = torch.sum(temp1, dim=1)
    a = temp31.size()
    a = a[0]
    temp21 = temp2 / a
    testmin = torch.argmin(temp21)
    print(f"Min index in pooling {testmin}and value {temp2[testmin]}")
    temp22=temp21.cpu()
    temp23 = temp22.numpy()

    a1 = np.array(temp22)
    a22 = np.expand_dims(a1, 0)
    a2 = a22.T
    dp = dpmeans(a2)
    dp.fit(a2)
    k = dp.K
    print(f"K means size {k}")
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(a2)
    cc1 = kmeans.cluster_centers_
    labels = kmeans.labels_
    labelmin = np.argmin(cc1)
    index2 = np.where(labels == labelmin)[0]
    print(f"K means index {index2} and value {temp21[index2]}")
    size11 = index2.size
    value11, cresult11 = torch.topk(temp21, size11, largest=False)

    list2=[]
    for i in range(size11):
        tempindex11=cresult11[i]
        value, row1 = torch.sort(temp1[:, tempindex11], dim=0)
        if row1[0] not in list2:
            tempindex12=row1[0].cpu()
            tempindex13 = tempindex12.numpy()
            list2.append(tempindex13)
        if len(list2) >= n_instances:
            print("selection breaking ")
            break
        if i >= n_instances:
            i = 0
    list3 = np.array(list2)
    print(f"return index {list3}")
    return list3

# Category Vector Inconsistency and Ranking of Scores(CVIRS)
# From paper "Effective active learning strategy for multi-label learning"
# implemented by Minghao Li
def CVIRS_sampling(self, pred_scores, n_instances):
    # Convert lists to tensors
    Y_tensor = self.train_dataset.get_y()  # torch.tensor(Y_in_train, dtype=torch.float32)
    pred_scores_tensor = torch.tensor(pred_scores, dtype=torch.float32)
    # Calculate M_phi
    M_phi = torch.abs(2 * pred_scores_tensor - 1)
    # print('M_score:', M_phi)
    unlabeled_samples_number = M_phi.size(0)
    labels_number = M_phi.size(1)
    # M_phi_transposed = M_phi.transpose(0, 1)
    # print(M_phi_transposed)
    # Sort M_phi along the second dimension
    M_phi_sorted, indices = torch.sort(M_phi, dim=0)
    # print(indices)
    # print(M_phi_sorted)
    # Initialize a tensor for ranks with the same shape as M_phi
    ranks = torch.zeros_like(M_phi, dtype=torch.long)
    # Assign ranks based on the sorted indices
    for i in range(M_phi.size(0)):
        ranks[indices[i, :], torch.arange(M_phi.size(1))] = i
    # Initialize Borda count scores
    S = ((unlabeled_samples_number - (ranks + 1)) / (unlabeled_samples_number * labels_number)).sum(dim=1)

    # print('Length of unlabeled_samples: ', unlabeled_samples_number)
    # print('Length of labels:', labels_number)
    # print('ranks:', ranks)
    # print('uncertainty S_score:', S)

    def h_2(x1, x2):
        if not (0 <= x1) or not (0 <= x2):
            raise ValueError("Probabilities a and b must be between 0 and 1.")

        if x1 == 0 or x2 == 0:
            return 0  # Joint entropy is 0 if any probability is 0 (assuming log_2(0) is 0)

        return - (x1 * math.log2(x1) + x2 * math.log2(x2))

    def h_4(a, b, c, d):
        if not (0 <= a) or not (0 <= b) or not (0 <= c) or not (0 <= d):
            raise ValueError("Probabilities a and b must be between 0 and 1.")

        if a + d == 0 and b + c == 0:
            return 0  # Joint entropy is 0 if any probability is 0 (assuming log_2(0) is 0)

        elif a + d != 0 and b + c == 0:
            return (a + d) / labels_number * h_2(a / (a + d), d / (a + d))

        else:
            # print('h_4 1st term:', h_2((b + c) / labels_number, (a + d) / labels_number))
            # print('h_4 2nd term:', (b+c)/labels_number*h_2(b/(b + c), c/(b+c)))
            # print('h_4 3rd term:', (a+d)/labels_number*h_2(a/(a+d), d/(a+d)))
            return (h_2((b + c) / labels_number, (a + d) / labels_number) + (b + c) / labels_number * h_2(b / (b + c),
                    c / (b + c)) + (a + d) / labels_number * h_2(a / (a + d), d / (a + d)))

    # Convert pred_scores_tensor to binary predictions (0 or 1)
    pred_binary = (pred_scores_tensor >= 0.5).float()
    print('predict binary results: ', pred_binary)
    d_score = []
    for i, pred_row in enumerate(pred_binary):
        d_score_item = []
        for j, Y_row in enumerate(Y_tensor): \
                # Compute the number of occurrences for each combination
            count_11_a = ((pred_row == 1) & (Y_row == 1)).sum()
            count_01_b = ((pred_row == 0) & (Y_row == 1)).sum()
            count_10_c = ((pred_row == 1) & (Y_row == 0)).sum()
            count_00_d = ((pred_row == 0) & (Y_row == 0)).sum()
            # print("Count of a(1,1):", count_11_a)
            # print("Count of b(0,1):", count_01_b)
            # print("Count of c(1,0):", count_10_c)
            # print("Count of d(0,0):", count_00_d)

            if count_11_a + count_00_d == 0 and count_01_b + count_10_c != 0:
                d_score_item.append(torch.tensor(1))
                continue
            d_n = ((2 * h_4(count_11_a, count_01_b, count_10_c, count_00_d) - h_2(
                (count_11_a + count_01_b) / labels_number, (count_10_c + count_00_d) / labels_number) -
                    h_2((count_10_c + count_00_d) / labels_number, (count_11_a + count_01_b) / labels_number)) / h_4(
                count_11_a, count_01_b, count_10_c, count_00_d))
            # print('d_n: ', d_n)
            # print('score 1st term:', (2 * h_4(a, b, c, d))/ h_4(a, b, c, d))
            # print('score 2st term:', (- h_2((a + b) / labels_number, (c + d) / labels_number))/h_4(a, b, c, d))
            # print('score 3st term:', (- h_2((c+d)/labels_number, (a+b)/labels_number)) / h_4(a, b, c, d))
            d_score_item.append(d_n)
        combined_tensor = torch.stack(d_score_item)
        d_score.append(torch.mean(combined_tensor))
    d_score_tensor = torch.stack(d_score)
    result = d_score_tensor + S
    _, selected_indices = torch.topk(result, n_instances, largest=False)
    # _, selected_indices = torch.topk(margin, n_instances, largest=True)
    return selected_indices

