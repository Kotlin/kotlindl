In this tutorial, we'll create a neural network with Kotlin DL: you'll learn about basic building blocks and concepts
you need to be familiar with to get started with Deep Learning. In the following tutorials you will learn how to 
[train this neural network](training_a_model.md), and how then 
[use the resulting model to generate predictions](loading_trained_model_for_inference.md). No previous deep learning 
experience is required to follow these tutorials. 

## Image Classification

One of the most common Deep Learning applications is image classification. In this tutorial, we'll be using 
the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) which you can think of as a "Hello World" 
of image classification. 
This dataset contains 70000 images that represent 10 different categories of clothing items. Each example is a 
28x28 grayscale image, associated with a single label from 10 classes. Here are the labels and how 
they map to the actual classes: 

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

Here's what the images themselves look like:
 
![](images/fashion-mnist-sprite.png)

We are going to build and train a neural network that will be able to classify images like these 
into given categories. If you're going to follow this tutorial along, you'll need to download the dataset - you can find the 
download links [here](https://github.com/zalandoresearch/fashion-mnist). 

## Building a Neural Network

Let's start by defining the structure of our neural network itself. The basic building block of a neural network is 
a **layer**. To define a neural network, we need to describe what layers it should consist of. 
 
The goal of the layers is to capture some data representation aspects during the training phase. With enough data, 
layers, and training iterations, neural networks are capable of capturing enough complexity to result in great 
performance on many tasks. 

The neural network that we will define here is called *Multilayer Perceptron (MLP)*. It is one of the simplest neural 
networks that consists of multiple fully-connected layers. This means that each neuron in any given layer receives 
outputs from all the neurons in the previous layer, and sends its own output to all neurons in the next layer.

Here's how we define a neural network that consists of a few simple layers in a sequence:  

```kotlin
val model = Sequential.of(
    Input(28,28,1),
    Flatten(),
    Dense(300),
    Dense(100),
    Dense(10)
)
```

Quite simple, right? Let's take a closer look at what we have defined here. 
First, we specify the kind of input we will pass to this neural network. We have images that are 28 x 28 pixels, and only 
have one color channel, thus the input will be an array of size 28 x 28 x 1. 

The first layer is `Flatten()`. This layer simply reformats the data - it transforms the three-dimensional
 input array into a one-dimensional array with 784 elements (28 * 28 * 1 = 784).

Next, we've got a sequence of three `Dense()` layers. Dense layers, also called fully-connected layers, are the most 
common layers in all kinds of neural network architectures. For each Dense layer we have specified its size. 
The first one has 300 neurons, the second one has 100 neurons, and the last one has 10 neurons (remember, we have 10 
different classes to predict). 

---
**NOTE**

In this case we only specified the size of each Dense layer. There are many other layer parameters that you can tweak. 
Kotlin DL comes with sensible defaults where possible to help you get started quickly. Here are the defaults that are 
used for `Dense` layers: 
* `activation = Activations.Relu`: Simply put, in a neuron, first a weighted sum of inputs plus bias is calculated, 
then an activation function is applied. Finally, this output is passed further. Most activation functions are differentiable 
 and add non-linearity. Non-linearity allows deep learning networks capture data's complexity. There are different 
 non-linear activation functions, however, [ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) is one 
 of the most commonly used ones.   
* `kernelInitializer = HeNormal()`, `biasInitializer = HeUniform()`: Before a neural network is trained, all the 
parameters need to be initialized. While it can be done with random numbers or zeros, these default initializer algorithms 
offer a more optimal way of doing so.  
 
**If you have used a framework like Keras before, note that the defaults may differ in Kotlin DL.**  

--- 

## Compiling a neural network 
In the previous step we have defined the neural network's structure. Now we need to decide *how* it will be trained - 
what optimization algorithm will be used, what do we want to optimize for, and how do we evaluate progress. We will 
describe this in the *compile* method:
    
```kotlin
    model.use{
            it.compile(Adam(), Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, Metrics.ACCURACY)
            // next step here is training the model: this is described in the next tutorial
            // ...
        }
```

* `model` is an AutoCloseable object, so you can leverage the `use` construct. If not, do not forget to call `model.close()`. 
* [Adam](https://arxiv.org/abs/1412.6980) is an optimization algorithm that our neural network will use to update the weights in its nodes as it trains.
* Loss function is used to measure model's accuracy during the training. This is what we optimize for to improve the model's performance. 
Here, the last layer of the model gives us 10 numbers - each representing a probability of the example belonging to each class. 
*Softmax Crossentropy with logits* measures the probability error. 
* Metrics allow you to monitor the training and evaluation of the model. *Accuracy* simply represents the percentage of correct 
predictions out of all predictions made.  

At this point we can call `summary()` method to get a printout of the neural network's architecture. 

```kotlin
    it.summary()
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
Great! We have defined the structure of the neural network, what optimization algorithm will be used during its training, 
and what we will be optimizing for. 

In the next tutorial, you'll learn how you can train this model on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  