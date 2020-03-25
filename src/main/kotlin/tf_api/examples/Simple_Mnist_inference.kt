package tf_api.examples

import tf_api.TFModel
import tf_api.TFModel.Companion.loadModelFromDirectory


private const val PATH_TO_MODEL_2 = "src/main/resources/models/mnist1"

fun main() {
    val model2: TFModel = loadModelFromDirectory(PATH_TO_MODEL_2)

    println(model2)
}