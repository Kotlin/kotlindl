In the previous tutorials, we [created](create_your_first_nn.md), [trained, and saved](training_a_model.md) a deep learning model. 
Now let's look at how we can load and use that model to generate predictions on new, previously unseen data.

For illustration purposes, and to simplify this tutorial, 
we'll use the test data subset to generate a prediction example that the model has not been trained on.

The example images in the test data have the same size and format as the ones the model has been trained on, 
so we do not need to do any additional preprocessing. 
However, if you are going to train the model on your own data, 
make sure to use the same image preprocessing before using an image for inference as you did while training your model. 
Every model expects to get exactly the same input as it was trained on. 

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

Congratulations! You have learned how to create, train, and use your first neural network model! 
