import numpy as np

train_data=np.load('training_images.npy')
train_labels=np.load('training_labels.npy')
sample_num=24

small_train_data=np.zeros([sample_num*10,128])
small_train_labels=np.zeros([sample_num*10])

count_m=np.zeros([10])

count=0
for i in range(60000):

    l=train_labels[i]

    if count_m[l]<sample_num:
    	small_train_data[count,:]=train_data[i]
    	small_train_labels[count]=l
    	count_m[l]=count_m[l]+1
    	count=count+1

print(count)
print(small_train_labels)

np.save("small_training_images.npy", small_train_data)
np.save("small_training_labels.npy", small_train_labels)