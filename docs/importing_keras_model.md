KotlinDL is a great library that can help you embed models that have been trained in Python with [Keras](https://keras.io) into JVM applications.

In this tutorial, we’ll explain how to save your Keras model so that it's compatible with KotlinDL, 
specify which architectures are currently supported, and demonstrate how to load and run an inference with such a model from your JVM project. 
  
### Supported architectures
KotlinDL supports a limited number of deep learning architectures. As the project evolves, we will expand the list of supported architectures.

Please check an up-to-date [list of supported layers](../README.md#Limitations).

### Saving a trained Keras model 
For this tutorial, we'll train a simple convolutional neural network that can classify images 
from the CIFAR-10 dataset in Python and then load it with KotlinDL.

---
**About CIFAR-10**

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. 
Here are some examples: 

![](images/cifar.png)

--- 
 
Here's how we define the convolutional neural network in Keras that we will train:
```python
model = models.Sequential(
[
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])
```

And here's the model's architecture:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928     
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                65600     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 122,570
Trainable params: 122,570
Non-trainable params: 0
```

You can find a Jupyter notebook with the Python code used to train this model [here](CIFAR-10.ipynb). 

Once this model has been trained, it's important to save it properly, so we can load it from KotlinDL:
```python
model.save('keras-cifar-10/weights', save_format='h5')

model_json = model.to_json()
with open("keras-cifar-10/model.json", "w") as json_file:
    json_file.write(model_json)
```

We need to save two things:
1) **model weights** – This is the result of the training. Make sure to specify `save_format='h5'`, which will save the model to disk in the HDF5 format as opposed to the default SavedModel format. 
KotlinDL requires the HDF5 format to load the model’s weights properly.

2) **model architecture** – Save this as a JSON file. 

### Loading the model and running the inference task
Once you have the HDF5 and JSON model files, you can load and use the model from your project. 
Note that any preprocessing that has been done to the images before training the model also needs to be done before running the inference task.

In this case we only normalized the pixel values before training, 
and KotlinDL provides a convenient method for reading an image straight into a normalized array, 
so you don't have to do it manually:  `ImageConverter.toNormalizedFloatArray()`. 

```kotlin
val labelsMap = mapOf(
    0 to "airplane",
    1 to "automobile",
    2 to "bird",
    3 to "cat",
    4 to "deer",
    5 to "dog",
    6 to "frog",
    7 to "horse",
    8 to "ship",
    9 to "truck"
)

val imageArray = ImageConverter.toNormalizedFloatArray(File(PATH_TO_IMAGE))

fun main() {
    val modelConfig = File(PATH_TO_MODEL_JSON)
    val weights = File(PATH_TO_WEIGHTS)

    val model = Sequential.loadModelConfiguration(modelConfig)

    model.use {
        it.compile(Adam(), Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, Metrics.ACCURACY)

        it.loadWeights(HdfFile(weights))

        val prediction = it.predict(imageArray)
        println("Predicted label is: $prediction. This corresponds to class ${labelsMap[prediction]}.")
    }
}
```

As you can see, loading and using a model trained with Keras is quite simple. 
If the architecture is supported, all you need to do is to save the weights and JSON configuration after training, 
load it from KotlinDL, and make sure to preprocess your data again before running the inference task.



  
