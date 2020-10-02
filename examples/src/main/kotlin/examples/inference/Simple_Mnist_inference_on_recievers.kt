package examples.inference

import datasets.Dataset
import datasets.handlers.*

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

    /*val mnistModel = prepareModelForInference {
        loadModel(PATH_TO_MODEL)
        reshape(::reshapeInput)
        input(Input.PLACEHOLDER)
        output(Output.ARGMAX)
    }

    mnistModel.use {
        val prediction = it.predict(train.getX(0))

        println("Predicted Label is: $prediction")
        println("Correct Label is: " + train.getLabel(0))

        val predictions = it.predictAll(test)
        println(predictions.toString())

        println("Accuracy is : ${it.evaluate(test, Metrics.ACCURACY)}")
    }*/
}

