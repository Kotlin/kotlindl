package examples.inference

import api.inference.savedmodel.Input
import api.inference.savedmodel.Output
import api.inference.savedmodel.SavedModelInferenceModel
import api.keras.metric.Metrics
import org.tensorflow.Tensor
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

    SavedModelInferenceModel().use {
        it.loadModel(PATH_TO_MODEL)
        println(it)

        it.reshape(::reshape)
        it.input(Input.PLACEHOLDER)
        it.output(Output.ARGMAX)

        val prediction = it.predict(images[0] as util.MnistUtils.Image)
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