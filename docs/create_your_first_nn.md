In this tutorial, we'll use Kotlin DL to create a neural network, and you'll learn about the basic building blocks and concepts needed to get started with Deep Learning. 
In subsequent tutorials, you will learn how to [train this neural network](training_a_model.md) and [use the resulting model to generate predictions](loading_trained_model_for_inference.md). 
No previous deep learning experience is required to get started with these tutorials.

## Image Classification

One of the most common applications of Deep Learning is image classification. 
In this tutorial, we'll be using the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), which you can think of as an image classification "Hello World." 
This dataset contains 70,000 images that fall into ten different categories of clothing items. 
Each example is a 28x28 grayscale image associated with a single label that matches one of the 10 classes. 
Here are the labels and how they map to the actual classes:

| Label |    Class    | 
|-------|:-----------:| 
| 0     | T-shirt/top | 
| 1     |  Trousers   |
| 2     |  Pullover   |
| 3     |    Dress    |
| 4     |    Coat     |
| 5     |   Sandals   |
| 6     |    Shirt    |
| 7     |  Sneakers   |
| 8     |     Bag     |
| 9     | Ankle boots |

For this example, we are planning to use a numeric representation for string categories, called string-to-number encoding. 
We'll later use this to get human-readable predictions:

```kotlin
val stringLabels = mapOf(0 to "T-shirt/top",
        1 to "Trousers",
        2 to "Pullover",
        3 to "Dress",
        4 to "Coat",
        5 to "Sandals",
        6 to "Shirt",
        7 to "Sneakers",
        8 to "Bag",
        9 to "Ankle boots"
)
```
When working with other datasets, you may need to look up this mapping in a metadata file that comes with a dataset. 
You may even need to encode it yourself if the original data contains the string representation of classes instead of integers.

Here's what the images themselves look like:
 
![](images/fashion-mnist-sprite.png)

We will build and train a neural network that will classify images like these into given categories. 
If you're going to follow along with this tutorial, you'll need to download the dataset, which you can find [here](https://github.com/zalandoresearch/fashion-mnist).

## Building a Neural Network

Let's start by defining the structure of our neural network. The basic building block of a neural network is a **layer**. 
So, to define a neural network, we need to describe what layers it should consist of. 
 
The goal of the layers is to capture aspects of data representations during the training phase. 
With enough data, layers, and training iterations, 
neural networks can capture enough complexity to deliver outstanding results on many tasks.

The neural network that we will define here is called _Multilayer Perceptron (MLP)_. 
Consisting of multiple fully connected layers, it is one of the simplest neural networks. 
Each neuron in a given layer receives outputs from all the neurons in the previous layer and sends its output to all the neurons in the next layer.

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
First, we specify the kind of input we will pass to this neural network. 
We have images that are 28 x 28 pixels and only have one color channel. 
Thus, the input will be an array of size 28 x 28 x 1.

The first layer is `Flatten()`. It simply reformats the data, 
transforming the three-dimensional input array into a one-dimensional array 
with 784 elements (28 * 28 * 1 = 784).

Next, we've got a sequence of three `Dense()` layers. 
Dense layers, also called fully connected layers, 
are the most common layers in all kinds of neural network architectures. 
For each Dense layer, we have specified its size. 
The first one has 300 neurons, the second one has 100 neurons, 
and the last one has 10 neurons (remember, we have 10 different classes to predict).

---
**NOTE**

In this case, we only specified the size of each Dense layer. 
There are many other layer parameters that you can tweak. 
Kotlin DL comes with sensible defaults where possible to help you get started quickly. 
Here are the defaults that are used for `Dense` layers: 
* `activation = Activations.Relu`Simply put, in a neuron, 
a weighted sum of the inputs plus bias is calculated, then an activation function is applied. 
This output is then passed along. 
Most activation functions are differentiable and add non-linearity, which allows deep learning networks to capture the complexity of a dataset. 
There’s a variety of non-linear activation functions, but [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) is one of the most commonly used ones. 
* `kernelInitializer = HeNormal()`, `biasInitializer = HeUniform()`– Before a neural network is trained, all the parameters need to be initialized. 
While this can be done with random numbers or zeros, these default initializer algorithms accelerate the training of the neural network.  
 
**If you have used a framework like Keras before, note that the defaults may be different in Kotlin DL.**  

--- 

## Compiling a neural network 
In the previous step, we defined the structure of our neural network. 
Next, we need to decide how it will be trained. What optimization algorithm will be used? 
What do we want to optimize for? And how will we evaluate progress? 
We will provide this information in the `compile()` method:
    
```kotlin
model.use {
    it.compile(
        optimizer = Adam(),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )

    // next step here is training the model: this is described in the following tutorial
    // ...
}
```

* `model`  is an AutoCloseable object, allowing you to leverage the `use` construct. If you don’t want to use the `use` Kotlin construction, do not forget to call `model.close()`. 
* [Adam](https://arxiv.org/abs/1412.6980) is the optimization algorithm that our neural network will use to update the weights in its nodes as it trains.
* The loss function is used to optimize your model. 
This is the function that will be minimized by the optimizer. 
Here, the last layer of the model gives us 10 numbers, each representing the probability of the example belonging to a given class. 
*Softmax Crossentropy with logits* measures the probability of error. *Softmax Crossentropy with logits* measures the probability error. 
* Metrics allow you to monitor the training and evaluation of the model. Accuracy simply represents the percentage of correct predictions out of all the predictions made.  

At this point, we can call the `printSummary()` method to get a printout of the neural network's architecture. 

```kotlin
it.printSummary()
```

```
==============================================================================
Model type: Sequential
______________________________________________________________________________
Layer (type)                           Output Shape              Param #      
==============================================================================
input_1(Input)                         [None, 28, 28, 1]         0            
______________________________________________________________________________
flatten_2(Flatten)                     [None, 784]               0            
______________________________________________________________________________
dense_3(Dense)                         [None, 300]               235500       
______________________________________________________________________________
dense_4(Dense)                         [None, 100]               30100        
______________________________________________________________________________
dense_5(Dense)                         [None, 10]                1010         
______________________________________________________________________________
==============================================================================
Total trainable params: 266610
Total frozen params: 0
Total params: 266610
==============================================================================

```
Great! We have now defined the structure of our neural network, specified the optimization algorithm that will be used during its training, and identified accuracy as the metric we will use to gauge its success.

In the [following tutorial](training_a_model.md), you'll learn how to train this model on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).
