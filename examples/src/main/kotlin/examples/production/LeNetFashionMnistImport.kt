package examples.production

import api.inference.InferenceModel
import datasets.Dataset
import datasets.handlers.*
import java.io.File

private const val PATH_TO_MODEL = "savedmodels/fashionLenet"

fun main() {
    val (train, test) = Dataset.createTrainAndTestDatasets(
        FASHION_TRAIN_IMAGES_ARCHIVE,
        FASHION_TRAIN_LABELS_ARCHIVE,
        FASHION_TEST_IMAGES_ARCHIVE,
        FASHION_TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )

    val inferenceModel = InferenceModel.load(File(PATH_TO_MODEL), loadOptimizerState = true)

    inferenceModel.use {
        it.reshape(::mnistReshape)

        var accuracy = 0.0
        val amountOfTestSet = 10000
        for (imageId in 0..amountOfTestSet) {
            val prediction = it.predict(train.getX(imageId))

            if (prediction == getLabel(train, imageId))
                accuracy += (1.0 / amountOfTestSet)
        }
        println("Accuracy: $accuracy")

        val amountOfOps = 1000
        val start = System.currentTimeMillis()
        for (i in 0..amountOfOps) {
            it.predict(train.getX(i % 50000))
        }
        println("Time, s: ${(System.currentTimeMillis() - start) / 1000f}")
        println("Throughput, op/s: ${amountOfOps / ((System.currentTimeMillis() - start) / 1000f)}")
    }
}
