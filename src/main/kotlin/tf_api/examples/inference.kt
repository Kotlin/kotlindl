package tf_api.examples

import tf_api.TFModel
import tf_api.TFModel.Companion.loadModelFromDirectory
import tf_api.TFModel.Companion.loadModelFromFile
import tf_api.TensorFlow
import util.MnistUtils
import java.util.*

const val IMAGE_PATH = "src/main/resources/datasets/test/t10k-images-idx3-ubyte"
const val LABEL_PATH = "src/main/resources/datasets/test/t10k-labels-idx1-ubyte"
const val PATH_TO_MODEL_1 = "src/main/resources/model3/saved_model.pb"
const val PATH_TO_MODEL_2 = "src/main/resources/model1"

fun main() {
    println(TensorFlow.version())

    val images = MnistUtils.mnistAsList(IMAGE_PATH, LABEL_PATH, Random(0), 10000)

    val model1: TFModel = loadModelFromFile(PATH_TO_MODEL_1)

    println(model1.toString())

    val model2: TFModel = loadModelFromDirectory(PATH_TO_MODEL_2)

    println(model2.toString())

    val prediction = model2.predict(images[0])

    println(prediction)

    val massivePrediction = model2.predict(images)

    println(massivePrediction)
}