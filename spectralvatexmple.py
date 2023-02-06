from spectralvat import specvat
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

################### simple guassian data creation###############
centers = [(-15, -5), (0, 0), (5, 15)]
data, labels = make_blobs(n_samples=10000, centers=centers, shuffle=False, random_state=42)

# illustrate data
plt.scatter(data[:,0], data[:,1], s=10, color='red')


############## applying specvat ################################
k=2  # have to change this value to get multiple plots
sv = specvat(data,k,cp=10,ns=400,use_cosine=True)
rv1 = sv.vat()
rv2 =  sv.ivat()

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
vat_pred = sv.compute_pred(cut,I,gt_clusters=3)

############ Predicted labels from iVAT ############################
cut = rv2[-1]
I=rv2[2]
ivat_pred = sv.compute_pred(cut,I,gt_clusters=3)


print(sv.compute_accuracy(vat_pred,labels))
