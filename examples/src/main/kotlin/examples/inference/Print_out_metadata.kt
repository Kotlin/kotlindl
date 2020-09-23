package examples.inference

import api.inference.savedmodel.SavedModel

private const val PATH_TO_MODEL = "src/main/resources/model1"

fun main() {
    val model = SavedModel().loadModel(PATH_TO_MODEL)

    println(model.toString())
}