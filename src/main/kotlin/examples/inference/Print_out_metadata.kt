package examples.inference

import api.inference.SavedModelInferenceModel

private const val PATH_TO_MODEL = "src/main/resources/model1"

fun main() {
    val model = SavedModelInferenceModel().loadModel(PATH_TO_MODEL)

    println(model.toString())
}