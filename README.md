# Gesture_Recognition
Hand Gesture Recognition using Deep Learning Framework

# Problem Statement

The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:
Thumbs up:  Increase the volume
Thumbs down: Decrease the volume
Left swipe: 'Jump' backwards 10 seconds
Right swipe: 'Jump' forward 10 seconds  
Stop: Pause the movie

# Training and Validation Images
Each video is a sequence of 30 frames (or images)

The data is in a zip file. The zip file contains a 'train' and a 'val' folder with two CSV files for the two folders. These folders are in turn divided into subfolders where each subfolder represents a video of a particular gesture. Each subfolder, i.e. a video, contains 30 frames (or images). Note that all images in a particular video subfolder have the same dimensions but different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 (depending on the webcam used to record the videos). Hence, you will need to do some pre-processing to standardise the videos. 

Each row of the CSV file represents one video and contains three main pieces of information - the name of the subfolder containing the 30 images of the video, the name of the gesture and the numeric label (between 0-4) of the video.

https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL

# Two Architectures: 3D Convs and CNN-RNN Stack

## CONV3D model
### 1st layer
model.add(Conv3D(8, (3, 3, 3), activation="relu",name="conv1", 
                     input_shape=(x,y,z,3),
                     data_format="channels_last",
                     padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(2, 2, 2), name="pool1"))

### 2nd layer
model.add(Conv3D(16, (3, 3, 3), activation="relu",name="conv2",padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(2, 2, 2), name="pool2"))

### 3rd layer
model.add(Conv3D(32, (1, 3, 3), activation="relu",name="conv3", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(2, 2, 2), name="pool3"))

### 4th layer
model.add(Conv3D(64, (1, 3, 3), activation="relu",name="conv4", padding="same",))
model.add(Dropout(0.25))
model.add(MaxPooling3D(pool_size=(2, 2, 2), name="pool4"))

### flatten and put a fully connected layer
model.add(Flatten())
model.add(Dense(256, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.5))

### Output layer
model.add(Dense(num_classes, activation='softmax', name='output'))

# CNN-RNN stack
### input layer
input_tensor = Input(shape=(x, y, z, 3))

### Get pre-trained model: vgg16 
vgg_layers = VGG16(weights='imagenet', include_top=False)
    # freeze the layers in base model
for layer in vgg_layers.layers:
    layer.trainable = False
    # create a VGG model selecting all layers   
vgg_model = Model(inputs=vgg_layers.input, outputs=vgg_layers.output)

### Adding Time Distributed wrapper on top of resnet model and passing the input tensor
time_distributed_layer= TimeDistributed(vgg_model)(input_tensor)

### Average pooling layer
avg_pool_layer= TimeDistributed(GlobalAveragePooling2D())(time_distributed_layer)

### Flatten before connecting to GRU
flatten_layer = TimeDistributed(Flatten())(avg_pool_layer)
flatten_dropped_out=Dropout(0.5)(flatten_layer)

### reshape the output of time distributed layer to be fed into LSTM or GRU
#tensor_size=np.prod(time_distributed_layer.get_shape().as_list()[2:]) 
#reshape_layer = Reshape(target_shape=(x,tensor_size))(time_distributed_layer)
         
### GRU layer
gru_out = GRU(128, return_sequences=False, dropout=0.5)(flatten_dropped_out)

### Fully connected Dense Layer
#fc_out = Dense(256, activation="relu")(gru_out)
#fc_dropped_out=Dropout(0.5)(fc_out)
#bn_layer=BatchNormalization()(fc_dropped_out)

### Output layer
output = Dense(num_classes, activation='softmax')(gru_out)

### final Model
model2 = Model(inputs=input_tensor, outputs=output)

# Experiments
Experiment Number	Model	Result 	Decision + Explanation
1.	Conv3D 	Accuracy: 0.15	Increase the number of layers to 4 followed by 2 fully connected layer.
2.	Conv3D	Accuracy: 0.27	Decrease of filter size in the direction of time in lower layers, so that more features are extracted in that direction.
3.	Conv3D	Accuracy: 0.40	Add augmented training images by flipping horizontally and reversing the label. 
4.	Conv3D	Accuracy: 0.76	Transform the training image by removing background using skin color detection technique.
5. 	Conv3D	Accuracy: 0.77	Very minimal increase in accuracy but good increase in training time. Thus No effect of background removal observed.
6.	Conv3D	Accuracy: 0.81	Create training batches on every epoch to choose stratified samples from each class. Thus removing data imbalance during training.
7.	Conv3D	Accuracy : 0.83	Keep the batch size to 40 and epochs to 30.
8.	ConvLSTM (Conv2d with Resnet50 pre-trained weights + GRU)	Accuracy: 0.16	The training accuracy increase gradually reaching to 0.96 but the validation accuracy remains 0.16. This is surely due to over fitting.
9.	ConvLSTM (Conv2d with Resnet50 pre-trained weights + GRU)	Accuracy: 0.19	Added dropouts, regularization and batch normalization to get rid of over fitting.
10.	ConvLSTM (Conv2d with Resnet50 pre-trained weights + GRU)	Accuracy: 0.23	Removed dropouts from GRU layer .Reduced the batch size to 20. Seems using Resnet the model is learning to classify images by face of person and other static features. The sequential features are lost somehow.
11.	ConvLSTM 
(Conv2d with VGG16 pre-trained weights + GRU)	Accuracy: 0.73	VGG16 is better feature extractor than Resnet50. Accuracy improved significantly. But still the model is overfitting. Train accuracy being 99%
12.	ConvLSTM 
(Conv2d with VGG16 pre-trained weights + GRU)	Accuracy: 0.73	Remove last 10 layer of VGG16 before feeding into the GRU. 

`Final Model	Conv3D	Accuracy : 0.83	The model .h5 file is less 11 MB and number of trainable parameters is also less.`

