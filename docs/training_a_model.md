In the previous tutorial we have defined a neural network. Now let's train it on the actual data. 

Before you can use any data, typically some preprocessing is required. In this case it's minimal - all images are 
already of the same size, and grayscale only. 
With built-in functionality we can convert Fashion MNIST image archives into a Dataset object that we can use for model training.    

```kotlin
val (train, test) = Dataset.createTrainAndTestDatasets(
        "MNIST/train-images-idx3-ubyte.gz",
        "MNIST/train-labels-idx1-ubyte.gz",
        "MNIST/t10k-images-idx3-ubyte.gz",
        "MNIST/t10k-labels-idx1-ubyte.gz",
        10,
        ::extractImages,
        ::extractLabels
    )
    val (newTrain, validation) = train.split(0.95)
```

You may also notice that we are splitting the data into three sets. First, we have the train and the est sets. We won't be touching 
the test set up until the very last moment when we are satisfied with the model and want to confirm its performance on unseen data.
However, we also split the train set into `newTrain` and `validation` sets. We'll be using these during the training and validation 
process.  

Now everything is ready to train the model. Use `fit` method for this: 

```kotlin
model.use{
        it.compile(Adam(), Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, Metrics.ACCURACY)
        it.summary()

        // You can think of the training process as "fitting" the model to describe the given data :)
        it.fit(dataset = newTrain, epochs = 10, batchSize = 100, verbose = false)

        val accuracy = it.evaluate(dataset = validation, batchSize = 100).metrics[Metrics.ACCURACY]
        println("Accuracy: $accuracy")
        it.save(File("src/model/my_model"))
    }

```

Here are some important parameters that we need to pass to the `fit` method:
* `epochs`: Number of iterations over the data you want the training process to perform. Epoch = iteration. 
* `batchSize`: How many examples will be used for updating model's parameters (aka weights and biases) at a time
*  You can set `verbose` to `true` you want to monitor the training process for every batch, and `false` if you only want 
to see updates per epoch.

Here's what you can expect to see as the output during the training process: 
```
DEBUG o.j.kotlinx.dl.api.core.Sequential - Initialization of TensorFlow Graph variables.
INFO  o.j.kotlinx.dl.api.core.Sequential - epochs: 1 loss: 0.49412054 metric: 0.8269117
INFO  o.j.kotlinx.dl.api.core.Sequential - epochs: 2 loss: 0.37001204 metric: 0.8689473
INFO  o.j.kotlinx.dl.api.core.Sequential - epochs: 3 loss: 0.33682162 metric: 0.8792109
INFO  o.j.kotlinx.dl.api.core.Sequential - epochs: 4 loss: 0.31501168 metric: 0.8863342
INFO  o.j.kotlinx.dl.api.core.Sequential - epochs: 5 loss: 0.29811463 metric: 0.8926845
INFO  o.j.kotlinx.dl.api.core.Sequential - epochs: 6 loss: 0.28373528 metric: 0.8974922
INFO  o.j.kotlinx.dl.api.core.Sequential - epochs: 7 loss: 0.27145803 metric: 0.9017029
INFO  o.j.kotlinx.dl.api.core.Sequential - epochs: 8 loss: 0.26044646 metric: 0.9054747
INFO  o.j.kotlinx.dl.api.core.Sequential - epochs: 9 loss: 0.24978207 metric: 0.9097908
INFO  o.j.kotlinx.dl.api.core.Sequential - epochs: 10 loss: 0.24072464 metric: 0.9134394

```  

Once the model has been trained, it's important to evaluate its performance on the validation dataset, so that we can 
check how it generalizes to the new data. 

```kotlin
val accuracy = model.evaluate(dataset = validation, batchSize = 100).metrics[Metrics.ACCURACY]
println("Accuracy: $accuracy")
```

```
Accuracy: 0.8909999132156372
```

---
**NOTE**

The results are nondeterministic, and you may have slightly different Accuracy value. 

--- 

Once we are happy with the model's evaluation metric, we can save the model for future use in the production environment.  

```kotlin
it.save(File("src/model/my_model"))
```

And just like that we have trained, evaluated, and saved a deep learning model that we can now use to generate
predictions (aka inference). In the [next tutorial](loading_trained_model_for_inference.md), you'll learn how to load and use the saved model.  
