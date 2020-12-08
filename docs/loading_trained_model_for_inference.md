In the previous tutorials, we have created, trained and saved a deep learning model. 
Now let's see how we can load and use that model to generate predictions on the new, previously unseen data. 

While loading the model and calling `predict` method on it is going to be quite straightforward, there are a few 
important steps that you need to do before that. 

First, we need to make sure the image that we'll be using is of the same size. We've trained our model on monochrome 
images 28 x 28 pixels. So we may need to crop, or resize the images, and convert them to grayscale in order to be able to 
use with this model. As a developer, you're responsible for making sure, your input matches the model's requirements. 

Here's an image of a bag we'll be using in this example, after the require preprocessing:

![](images/test-image-bag.png)

The model expects as an input a 28 x 28 x 1 array. We can read in the image as 
a FloatArray, and define a function that will help us reshape a regular FloatArray into a multi-dimensional array with 
appropriate shape:

```kotlin
val floatArray = ImageConverter.toRawFloatArray(File(PATH_TO_TEST_IMAGE))

fun reshapeInput(inputData: FloatArray): Array<Array<FloatArray>> {
    val reshaped = Array(
            1
    ) { Array(28) { FloatArray(28) } }
    for (i in inputData.indices) reshaped[0][i / 28][i % 28] = inputData[i]
    return reshaped
}

``` 

At this point we can simply load the model using the path to it, tell it how incoming images should be reshaped, 
and call the `predict` method on them. 

```kotlin
fun main() {
    InferenceModel.load(File(PATH_TO_MODEL)).use {
        it.reshape(::reshapeInput)
        val prediction = it.predict(floatArray)
        println("Predicted label is: $prediction. This corresponds to class ${stringLabels[prediction]}.")
    }
}
```

```
Predicted label is: 8. This corresponds to class Bag.
```

Congratulations! You have learned how to create, train and use your first neural network model! 