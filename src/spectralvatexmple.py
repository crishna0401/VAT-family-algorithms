from spectralvat import specvat
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

################### simple guassian data creation ###############
data, labels = make_blobs(n_samples=1000, centers=3, n_features=200,random_state=4, cluster_std=1)

# illustrate data
plt.scatter(data[:,0], data[:,1], s=10, color='red')

############## applying specvat ################################
sv = specvat(data,cp=10,ns=400,use_cosine=False)




############# Experimenting with various 'k' ####################
k=9
ground_truth_clusters = len(np.unique(labels))

rs = sv.dissimilarity(k)
rv1 = sv.vat(rs)
rv2 =  sv.ivat(rs)

############ Plotting reordered dissimilarity matrices #########

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(rv1[0], cmap='gray')
plt.title('VAT')
plt.subplot(1, 2, 2)
plt.imshow(rv2[0], cmap='gray')
plt.title('iVAT')


############ Predicted labels from VAT ############################
cut = rv1[-1]
I=rv1[2]
vat_pred = sv.compute_pred(cut,I,gt_clusters=ground_truth_clusters)

############ Predicted labels from iVAT ############################
cut = rv2[-1]
I=rv2[2]
ivat_pred = sv.compute_pred(cut,I,gt_clusters=ground_truth_clusters)


print(sv.compute_accuracy(vat_pred,labels))
print(sv.compute_accuracy(ivat_pred,labels))
