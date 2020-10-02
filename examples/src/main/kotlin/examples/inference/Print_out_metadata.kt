package examples.inference

import api.inference.savedmodel.SavedModel

private const val PATH_TO_MODEL = "examples/src/main/resources/savedmodel"

fun main() {
    val model = SavedModel.load(PATH_TO_MODEL)

    println(model.toString())
}