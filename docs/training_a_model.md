In the [previous tutorial](create_your_first_nn.md) we have defined a neural network. Now let's train it on the actual data. 

Before you can use any data, typically some preprocessing is required. In this case it's minimal - all images are 
already of the same size, and grayscale only. 
With built-in functionality we can convert the [Fashion MNIST image archives](https://github.com/zalandoresearch/fashion-mnist#get-the-data) into a Dataset object that we can use for model training.    

```kotlin
val (train, test) = Dataset.createTrainAndTestDatasets(
    trainFeaturesPath = "datasets/mnist/train-images-idx3-ubyte.gz",
    trainLabelsPath = "datasets/mnist/train-labels-idx1-ubyte.gz",
    testFeaturesPath = "datasets/mnist/t10k-images-idx3-ubyte.gz",
    testLabelsPath = "datasets/mnist/t10k-labels-idx1-ubyte.gz",
    numClasses = 10,
    ::extractImages,
    ::extractLabels
)

val (newTrain, validation) = train.split(splitRatio = 0.95)
```

You may also notice that we are splitting the data into three sets. First, we have the train and the test sets. We won't be touching 
the test set up until the very last moment when we are satisfied with the model and want to confirm its performance on unseen data.
However, we also split the train set into `newTrain` and `validation` sets. We'll be using these during the training and validation 
process.  

Now everything is ready to train the model. Use the `fit` method for this: 

```kotlin
model.use {
    it.compile(
        optimizer = Adam(),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )

    it.summary()

    // You can think of the training process as "fitting" the model to describe the given data :)
    it.fit(
        dataset = newTrain,
        epochs = 10,
        batchSize = 100
    )

    val accuracy = it.evaluate(dataset = validation, batchSize = 100).metrics[Metrics.ACCURACY]

    println("Accuracy: $accuracy")
    it.save(File("src/model/my_model"))
}

```

Here are some important parameters that we need to pass to the `fit` method:
* `epochs`: Number of iterations over the data you want the training process to perform. Epoch = iteration. 
* `batchSize`: How many examples will be used for updating model's parameters (aka weights and biases) at a time

Once the model has been trained, it's important to evaluate its performance on the validation dataset, so that we can 
check how it generalizes to the new data. 

```kotlin
val accuracy = it.evaluate(dataset = validation,
    batchSize = 100).metrics[Metrics.ACCURACY]

println("Accuracy: $accuracy")
```

```
Accuracy: 0.9806665182113647
```

---
**NOTE**

The results are nondeterministic, and you may have a slightly different Accuracy value. 

--- 

Once we are happy with the model's evaluation metric, we can save the model for future use in the production environment.  

```kotlin
it.save(File("src/model/my_model"))
```

And just like that we have trained, evaluated, and saved a deep learning model that we can now use to generate
predictions (aka inference). In the [next tutorial](loading_trained_model_for_inference.md), you'll learn how to load and use the saved model.  
