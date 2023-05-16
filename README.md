

# Deep-Food
### Food image classification model using EfficientNetB0 

#### View Web app [here](https://nagarajrpoojari-deep-food-app-mouwg8.streamlit.app/)

Deep-Food is a Convolutional neural network model for multiclass image classification . It uses 
[EfficientNetB0](https://arxiv.org/abs/1905.11946) pre trained model and 
achieves  accuracy of __80%__ .

###

## Architecture

<img src="https://user-images.githubusercontent.com/116948655/235911765-af644520-6188-4b81-ad3f-f2a6fcbd1dd6.png"  height="200" >

### EfficientNetB0
<img src="https://iq.opengenus.org/content/images/2022/11/Architecture-of-EfficientNet-B0-with-MBConv-as-Basic-building-blocks.png"  height="200">

### Download dataset
```python
!wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
!tar xzvf food-101.tar.gz
```
### Create model
```python
from tensorflow.keras import layers

input_shape= (224,224,3)
base_model=tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable=False
inputs=layers.Input(input_shape, name='input_layer')
x=base_model(inputs, training=False)
x=layers.GlobalAveragePooling2D(name='Global_avg_pooling_2D')(x)
x=layers.Dense(101)(x)
output = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x) 
model_1=tf.keras.Model(inputs,output)
model_1.compile(loss='sparse_categorical_crossentropy' , optimizer='adam', metrics=['accuracy'])

```
### Load weights
```python
model_1.load_weights('food_vision_fine_tune_checkpoint')
```

## Sample tests

![now](https://user-images.githubusercontent.com/116948655/235911294-f925e8d1-99d5-4ad6-8766-7edaa4f65fd0.png)









