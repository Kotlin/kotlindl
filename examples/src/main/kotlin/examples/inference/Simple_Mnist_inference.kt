package examples.inference

import api.core.metric.Metrics
import api.inference.savedmodel.Input
import api.inference.savedmodel.Output
import api.inference.savedmodel.SavedModel
import datasets.Dataset
import datasets.handlers.*
import org.tensorflow.Tensor

private const val PATH_TO_MODEL = "examples/src/main/resources/savedmodel"

fun main() {
    val (train, test) = Dataset.createTrainAndTestDatasets(
        TRAIN_IMAGES_ARCHIVE,
        TRAIN_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )

    SavedModel.load(PATH_TO_MODEL).use {
        println(it)

        it.reshape(::reshapeInput)
        it.input(Input.PLACEHOLDER)
        it.output(Output.ARGMAX)

        val prediction = it.predict(train.getX(0))

        println("Predicted Label is: $prediction")
        println("Correct Label is: " + train.getLabel(0))

        val predictions = it.predictAll(test)
        println(predictions.toString())

        println("Accuracy is : ${it.evaluate(test, Metrics.ACCURACY)}")
    }
}

fun reshapeInput(inputData: FloatArray): Tensor<*> {
    val reshaped = Array(
        1
    ) { Array(28) { FloatArray(28) } }
    for (i in inputData.indices) reshaped[0][i / 28][i % 28] = inputData[i]
    return Tensor.create(reshaped)
}