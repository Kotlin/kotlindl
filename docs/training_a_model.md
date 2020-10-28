In the previous tutorial you have learned how you can define a neural network. Now let's train it on the actual data. 

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

You may also notice that we are splitting the data into three sets. First we have train and test sets. We won't be touching 
the test set up until the very last moment when we are happy with the model and want to confirm its performance on unseen data.
However, we also split the train set into `newTrain` and `validation` sets. We'll be using these during the training and validation 
process.  

Now everything is ready to train the model. Use `fit` method for this: 

```kotlin
model.fit(dataset = newTrain, epochs = 30, batchSize = 100, verbose = false)
```
* `epochs`: Number of iterations over the data you want the training process to perform. Epoch = iteration. 
* `batchSize`: How many examples will be used for updating model's parameters at a time
*  You can set `verbose` to true you want to monitor the training process for every batch, and `false` if you only want 
to see updates per batch.

Here's what you can expect to see as the output during the training process: 
```
DEBUG api.core.Sequential - Initialization of TensorFlow Graph variables.
INFO  api.core.Sequential - epochs: 1 loss: 2.8337643 metric: 0.68498194
INFO  api.core.Sequential - epochs: 2 loss: 0.57636863 metric: 0.81217456
INFO  api.core.Sequential - epochs: 3 loss: 0.50755733 metric: 0.8307009
INFO  api.core.Sequential - epochs: 4 loss: 0.47304747 metric: 0.8395257
INFO  api.core.Sequential - epochs: 5 loss: 0.44947788 metric: 0.84642005
INFO  api.core.Sequential - epochs: 6 loss: 0.43019584 metric: 0.8521927
INFO  api.core.Sequential - epochs: 7 loss: 0.41350496 metric: 0.85778904
INFO  api.core.Sequential - epochs: 8 loss: 0.39861047 metric: 0.8623504
INFO  api.core.Sequential - epochs: 9 loss: 0.3852057 metric: 0.8665084
INFO  api.core.Sequential - epochs: 10 loss: 0.372514 metric: 0.8703684
INFO  api.core.Sequential - epochs: 11 loss: 0.36032277 metric: 0.87396497
INFO  api.core.Sequential - epochs: 12 loss: 0.3486484 metric: 0.8781054
INFO  api.core.Sequential - epochs: 13 loss: 0.33731312 metric: 0.88173723
INFO  api.core.Sequential - epochs: 14 loss: 0.32656613 metric: 0.884965
INFO  api.core.Sequential - epochs: 15 loss: 0.31635296 metric: 0.88829905
INFO  api.core.Sequential - epochs: 16 loss: 0.3069187 metric: 0.891598
INFO  api.core.Sequential - epochs: 17 loss: 0.29805952 metric: 0.8941766
INFO  api.core.Sequential - epochs: 18 loss: 0.28962037 metric: 0.8972118
INFO  api.core.Sequential - epochs: 19 loss: 0.28165802 metric: 0.8995099
INFO  api.core.Sequential - epochs: 20 loss: 0.27431554 metric: 0.9024576
INFO  api.core.Sequential - epochs: 21 loss: 0.2670171 metric: 0.9043867
INFO  api.core.Sequential - epochs: 22 loss: 0.2601832 metric: 0.90642184
INFO  api.core.Sequential - epochs: 23 loss: 0.25347552 metric: 0.90887773
INFO  api.core.Sequential - epochs: 24 loss: 0.24706276 metric: 0.9110887
INFO  api.core.Sequential - epochs: 25 loss: 0.2407111 metric: 0.91365016
INFO  api.core.Sequential - epochs: 26 loss: 0.23494755 metric: 0.91535187
INFO  api.core.Sequential - epochs: 27 loss: 0.22893474 metric: 0.91786027
INFO  api.core.Sequential - epochs: 28 loss: 0.223119 metric: 0.9197548
INFO  api.core.Sequential - epochs: 29 loss: 0.21775614 metric: 0.92207074
INFO  api.core.Sequential - epochs: 30 loss: 0.21214716 metric: 0.9247379
```  

Once the model has been trained, it's important to evaluate its performance on the validation dataset, so that you can 
check how it generalizes to the new data. 

```kotlin
val accuracy = model.evaluate(dataset = validation, batchSize = 100).metrics[Metrics.ACCURACY]
println("Accuracy: $accuracy")
```

```
Accuracy: 0.8919999599456787
```

If you are happy with the model's evaluation metric, you can save the model for future use in your production environment.  

```kotlin
model.save(File("src/model/my_model"))
model.close()
```

And just like that you have trained, evaluated, and saved your first deep learning model that you can now use to generate
predictions (aka inference). In the [next tutorial](loading_trained_model_for_inference.md), you'll learn how to load and use a saved model.  
