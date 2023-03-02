In this tutorial you'll learn about the basic building blocks and concepts needed to get started with Deep Learning. 
We'll use Kotlin DL to create a neural network, train it and use the resulting model to generate predictions.
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

## Training a model

In the previous sections, we defined our neural network's structure,
specified the optimization algorithm that will be used during its training, and identified accuracy as the metric we will use to gauge its success.

In this section you'll learn how to train this model on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

Before you can use data, typically, some preprocessing is required.
In this case, it's minimal – all the images are already the same size and are grayscale.
With the built-in functionality, we can convert the [Fashion MNIST image archives](https://github.com/zalandoresearch/fashion-mnist#get-the-data) into a dataset object used for model training.

```kotlin
val (train, test) = fashionMnist()
```

You may also notice that we are splitting the data into two sets.
We have the test set, which we won't be touching until we are satisfied with the model and want to confirm its performance on unseen data.
And we have the train set which we'll use during the training process.

Note that it's a slightly simplified approach.
In practice, there is also a third subset of data.
It's called a validation set, and it's needed to monitor the model target metrics during training and tuning the model hyperparameters.
Another common technique for robust model evaluation is [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)).
Splitting a dataset for model evaluation is essential in deep learning and machine learning, but it's enough for our tutorial to use a simplified approach.

Now everything is ready to train the model. Use the `fit()` method for this:

```kotlin
model.use {
    it.compile(
        optimizer = Adam(),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )

    it.printSummary()

    // You can think of the training process as "fitting" the model to describe the given data :)
    it.fit(
        dataset = train,
        epochs = 10,
        batchSize = 100
    )

    val accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

    println("Accuracy: $accuracy")
    it.save(File("model/my_model"), writingMode = WritingMode.OVERRIDE)
}
```

Here are some important parameters that we need to pass to the `fit()` method:
* `epochs` - Number of iterations over the data you want the training process to perform. Epoch = iteration.
* `batchSize` - How many examples will be used for updating the model's parameters (aka weights and biases) at a time.

After the model has been trained, it's important to evaluate its performance on the test set, so that we can check how it generalizes to the new data.

```kotlin
val accuracy = it.evaluate(dataset = test,
    batchSize = 100).metrics[Metrics.ACCURACY]

println("Accuracy: $accuracy")
```

```
Accuracy: 0.8821001648902893
```

---
**NOTE**

The results are nondeterministic, and you may have a slightly different Accuracy value.

--- 

When we are happy with the model's evaluation metric, we can save the model for future use in the production environment.

```kotlin
it.save(File("model/my_model"), writingMode = WritingMode.OVERRIDE)
```

And just like that, we have trained, evaluated, and saved a deep learning model that we can now use to generate predictions (aka inference).

## Loading trained model for inference

In this section let's look at how to load and use a trained model to generate predictions on new, previously unseen data.

For illustration purposes, and to simplify this tutorial,
we'll use the test data subset to generate a prediction example that the model has not been trained on.

The example images in the test data have the same size and format as the ones the model has been trained on,
so we do not need to do any additional preprocessing.
However, if you train the model on your data,
use the same image preprocessing before using an image for inference as you did while training your model.
Every model expects to get the same input as it was trained on.

To load the model simply use the path to it, tell it how incoming images should be reshaped (if needed), and call the
`predict` method on them.

```kotlin
fun main() {
    val (train, test) = fashionMnist()

    TensorFlowInferenceModel.load(File("model/my_model")).use {
        it.reshape(28, 28, 1)
        val prediction = it.predict(test.getX(0))
        val actualLabel = test.getY(0)

        println("Predicted label is: $prediction. This corresponds to class ${stringLabels[prediction]}.")
        println("Actual label is: $actualLabel.")
    }
}
```

```
Predicted label is: 9. This corresponds to class Ankle boots.
Actual label is: 9.0.
```

Congratulations! You have learned to create, train, and use your first neural network model!