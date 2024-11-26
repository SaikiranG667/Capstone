from sklearn.datasets import load_files       
from keras.utils import to_categorical
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
##from Pillow import load_img

tar=10
path='./dataset/'


# define function to load train, test, and validation datasets
def load_data(path):
    data=load_files(path)
    files=np.array(data['filenames'])
    targets=to_categorical(np.array(data['target']), tar)
    return files, targets


# load train, test, and validation datasets
train_files,train_targets= load_data(path)
##test_files=train_files
##test_targets=train_targets
train_files, test_files, train_targets, test_targets = train_test_split(
    train_files,train_targets, test_size=0.2, random_state=42)
# get the burn classes
# We only take the characters from a starting position to remove the path
burn_classes = [item[10:-1] for item in sorted(glob("./dataset/*/"))]
print(burn_classes)


# print statistics about the dataset
print('There are %d total categories.' % len(burn_classes))
print('There are %s total  images.\n' % len(np.hstack([train_files, test_files])))
print('There are %d training images.' % len(train_files))
print('There are %d test images.'% len(test_files))


from tensorflow.keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path,width=224,height=224):
    
    # loads RGB image as PIL.Image.Image type
##    img=image.load_img(img_path, target_size=(width, height))
    img=image.load_img(img_path, target_size=(width, height))    
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)    
    x=image.img_to_array(img)
    
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensor=[path_to_tensor(img_path, width, height) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensor)



# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255




                
import keras
import timeit

# graph the history of model.fit
def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show() 


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True


from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense , Activation
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

img_width,img_height=224,224
batch_size=64
epochs=30
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 3
lr = 0.0004
                 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import callbacks
import time
#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(tar, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

hist=model.fit(train_tensors, train_targets ,validation_split=0.3, epochs=epochs, batch_size=batch_size)


show_history_graph(hist)





##test_loss, test_acc = model.evaluate(test_tensors, train_targets)
test_loss, test_acc = model.evaluate(test_tensors, test_targets)


y_pred=model.predict(test_tensors)


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import roc_curve



# Calculate ROC curve from y_test and pred

from sklearn.metrics import roc_curve
fpr ,tn, thresholds = roc_curve(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)
accuracycnn = accuracy_score(np.argmax(test_targets, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)

print('Accuracy=' +str(accuracycnn*100))


fpr ,tpr, thresholds = roc_curve(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)
Sensitivity= tpr / (tpr+fpr)

print('Sensitivity='+str(Sensitivity[1]))
precisioncnn = precision_score(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)

print('precision='+str(precisioncnn))

fpr ,tpr, thresholds = roc_curve(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)
f1scorecnn = f1_score(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)

print('f1-score='+str(f1scorecnn))

fpr ,tpr, thresholds = roc_curve(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)
recallscorecnn = recall_score(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)

print('recall-score='+str(recallscorecnn))

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(np.argmax(test_targets, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)
# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')
# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')
# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')
# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('DNN True Positive Rate')
plt.xlabel('DNN False Positive Rate')
plt.show()





model.save('trained_model_DNN1.h5')
print('Model saved')





