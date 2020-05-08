package examples.inference

import org.tensorflow.Tensor
import tf_api.inference.prepareModelForInference
import tf_api.keras.Input
import tf_api.keras.Output
import tf_api.keras.metric.Metrics
import util.MnistUtils
import java.util.*


private const val PATH_TO_MODEL = "src/main/resources/model1"
private const val IMAGE_PATH = "src/main/resources/datasets/test/t10k-images-idx3-ubyte"
private const val LABEL_PATH = "src/main/resources/datasets/test/t10k-labels-idx1-ubyte"

fun main() {
    val images = MnistUtils.mnistAsList(
        IMAGE_PATH,
        LABEL_PATH, Random(0), 10000
    )

    val mnistModel = prepareModelForInference {
        loadModel(PATH_TO_MODEL)
        reshape(::reshape)
        input(Input.PLACEHOLDER)
        output(Output.ARGMAX)
    }

    mnistModel.use {
        val prediction = it.predict(images[0])
        println("Predicted Label is: " + prediction[0].toInt())
        println("Correct Label is: " + images[0].label)

        val predictions = it.predictAll(images)
        println(predictions.toString())

        println("Accuracy is : ${it.evaluate(images, Metrics.ACCURACY)}")
    }
}

private fun reshape(doubles: DoubleArray): Tensor<*>? {
    val reshaped = Array(
        1
    ) { Array(28) { FloatArray(28) } }
    for (i in doubles.indices) reshaped[0][i / 28][i % 28] = doubles[i].toFloat()
    return Tensor.create(reshaped)
}