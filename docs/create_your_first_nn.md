In this tutorial, you'll learn how you can create your first neural network with Kotlin DL. We'll guide you through the 
basic building blocks and concepts you'll need to get started with Deep Learning. 
This tutorial does not require any previous experience with Deep Learning. 

## Image Classification

For the purposes of this tutorial, we'll be using the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist)
which you can think of a "Hello World" dataset in Deep Learning. 
This dataset contains 70000 images that represent 10 different categories of clothing items. 
Each example is a 28x28 grayscale image, associated with a single label from 10 classes. Here are the labels and how 
they map to actual classes: 

| Label        | Class           | 
| ------------- |:-------------:| 
| 0      | T-shirt/top | 
| 1      | Trouser |
| 2      | Pullover |
| 3      | Dress |
| 4      | Coat |
| 5      | Sandal |
| 6      | Shirt |
| 7      | Sneaker |
| 8      | Bag |
| 9      | Ankle boot |

And this is what the images themselves look like: 
![](images/fashion-mnist-sprite.png)

In this tutorial we are going to build and train a neural network that will be able to classify images like these 
into given categories. 

## Building a Neural Network

Before our neural network can learn from the data, we need to define the structure of the neural network itself. 
The basic building block of a neural network is a **layer**. To define a neural network, you need to describe 
what layers it should consist of. The goal of those layers is to capture some data representation aspects during 
the training phase. With enough data, layers, and training time, neural networks are capable of capturing enough 
complexity to result in great performance on many tasks. 

Here, our neural network will consist of a few simple layers in a sequence:  

```kotlin
private val model = Sequential.of(
    Input(28,28),
    Flatten(),
    Dense(300, Activations.Relu),
    Dense(100, Activations.Relu),
    Dense(10, Activations.Linear)
)
```

First, we define what kind of input will be passed to this neural network. The images in the dataset are greyscale, 
so we don't need to worry about multiple color channels, and they are 28 x 28 pixels. Thus, an input image can be 
represented as a two-dimensional array with 28 x 28 elements (with values ranging from 0 to 255).

The first layer is **Flatten()**. This layer simply reformats the data - it transforms the two-dimensional
 input array into a one-dimensional array with 784 elements (28 * 28 = 784).
 
Next, we've got a sequence of three **Dense()** layers. Dense layers are the most common layers in all kinds
 of neural network architectures. They are also called fully connected layers because they connect every neuron 
 in one layer to every neuron in the next layer.
 
For each Dense layer we have specified its size. The first one has 300 neurons (or nodes), the second one has 100 neurons, 
and the last one has 10 neurons (remember, we have 10 different classes to predict). 

In each Dense layer we have also explicitly specified an activation function to be used. 
Without activation functions a neural network would simply be passing data through linear nodes and would not be performing 
any better than a simple linear regression model. Adding non-linear activation functions to the layers allows neural networks
to capture data's complexity. There are different kinds of such functions. Here we've picked 
[ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) as one of the most common choices for an activation function.    

## Compiling a neural network 
In the previous step we have defined the neural network's structure. In the next step, we need to decide *how* it will be trained - 
what optimization algorithm will be used, what do we want to optimize for, and how do we evaluate progress. This is defined 
in the *compile* step:

```kotlin
    model.compile(Adam(), Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, Metrics.ACCURACY)
```

* [Adam](https://arxiv.org/abs/1412.6980) is an optimization algorithm that our neural network will use to update the weights in its nodes as it trains.
* Loss function is used to measure model's accuracy during the training. This is what you optimize for to improve your model's performance. 
Here, the last layer of the model gives us 10 numbers - each representing a probability of the example belonging to each class. 
Softmax Crossentropy with logits measures the probability error. 
* Metrics allow you to monitor the training and evaluation of the model. Accuracy simply represents the number of correct 
predicitons out of all predictions made.  

At this point you can call `summary()` method to get a printout of the neural network's architecture. 

```kotlin
    model.summary()
```

```
INFO  api.core.Sequential - =================================================================
INFO  api.core.Sequential - Model: Sequential
INFO  api.core.Sequential - _________________________________________________________________
INFO  api.core.Sequential - Layer (type)                 Output Shape              Param #   
INFO  api.core.Sequential - =================================================================
INFO  api.core.Sequential - flatten_1(Flatten)           [784]                     0
INFO  api.core.Sequential - _________________________________________________________________
INFO  api.core.Sequential - dense_2(Dense)               [300]                     235500
INFO  api.core.Sequential - _________________________________________________________________
INFO  api.core.Sequential - dense_3(Dense)               [100]                     30100
INFO  api.core.Sequential - _________________________________________________________________
INFO  api.core.Sequential - dense_4(Dense)               [10]                      1010
INFO  api.core.Sequential - _________________________________________________________________
INFO  api.core.Sequential - =================================================================
INFO  api.core.Sequential - Total trainable params: 266610
INFO  api.core.Sequential - Total frozen params: 0
INFO  api.core.Sequential - Total params: 266610
INFO  api.core.Sequential - =================================================================

```
Great! You have defined the structure of your first neural network, what optimization algorithm will be used during its training, 
and what you are optimizing for. 

In the next tutorial, you'll learn how you can train this model on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  