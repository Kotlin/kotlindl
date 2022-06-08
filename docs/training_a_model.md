In the [previous tutorial](create_your_first_nn.md) we defined a neural network. Now let's train it on the actual data. 

Before you can use data, typically some preprocessing is required. 
In this case, it's minimal – all the images are already the same size and are grayscale. 
With the built-in functionality, we can convert the [Fashion MNIST image archives](https://github.com/zalandoresearch/fashion-mnist#get-the-data) into a dataset object that we can use for model training.    

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
In the [next tutorial](loading_trained_model_for_inference.md), you'll learn how to load and use the model.
