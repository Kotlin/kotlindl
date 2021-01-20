In the previous tutorials, we have [created](create_your_first_nn.md), [trained and saved](training_a_model.md) a deep learning model. 
Now let's see how we can load and use that model to generate predictions on the new, previously unseen data. 

For the illustration purposes, and to simplify the tutorial, we'll use the test data subset to generate a prediction for 
an example that the model has not been trained on. 

The examples in the test data have the same size and format as the ones the model has been trained on, so we do not need 
to do any additional preprocessing. However, if you are going to be training models on your own data, make sure to use 
the same image preprocessing before using an image for inference as you did for training your model. Every model expects 
exactly the same input as it was trained on.  

To load the model simply use the path to it, tell it how incoming images should be reshaped (if needed), 
and call the `predict` method on them.

```kotlin
fun reshapeInput(inputData: FloatArray): Array<Array<FloatArray>> {
    val reshaped = Array(
        1
    ) { Array(28) { FloatArray(28) } }
    for (i in inputData.indices) reshaped[0][i / 28][i % 28] = inputData[i]
    return reshaped
}

fun main() {
    InferenceModel.load(File(PATH_TO_MODEL)).use {
        it.reshape(::reshapeInput)
        val prediction = it.predict(test.getX(0))
        val actualLabel = test.getLabel(0)

        println("Predicted label is: $prediction. This corresponds to class ${stringLabels[prediction]}.")
        println("Actual label is: $actualLabel.")
    }
}
```

```
Predicted label is: 7. This corresponds to class Sneaker.
Actual label is: 7.
```

Congratulations! You have learned how to create, train and use your first neural network model! 