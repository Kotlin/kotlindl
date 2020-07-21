package examples.production

import api.inference.InferenceModel
import api.keras.dataset.ImageDataset
import examples.keras.fashionmnist.util.FASHION_TEST_IMAGES_ARCHIVE
import examples.keras.fashionmnist.util.FASHION_TEST_LABELS_ARCHIVE
import examples.keras.fashionmnist.util.FASHION_TRAIN_IMAGES_ARCHIVE
import examples.keras.fashionmnist.util.FASHION_TRAIN_LABELS_ARCHIVE
import examples.keras.mnist.util.AMOUNT_OF_CLASSES
import examples.keras.mnist.util.extractImages
import examples.keras.mnist.util.extractLabels

private const val PATH_TO_MODEL = "savedmodels/fashionLenet"

fun main() {
    val (train, test) = ImageDataset.createTrainAndTestDatasets(
        FASHION_TRAIN_IMAGES_ARCHIVE,
        FASHION_TRAIN_LABELS_ARCHIVE,
        FASHION_TEST_IMAGES_ARCHIVE,
        FASHION_TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )

    InferenceModel<Float>().use {
        it.load(PATH_TO_MODEL)

        var accuracy = 0.0
        val amountOfTestSet = 10000
        for (imageId in 0..amountOfTestSet) {
            val prediction = it.predict(train.getImage(imageId))

            if (prediction == getLabel(train, imageId))
                accuracy += (1.0 / amountOfTestSet)

            //println("Prediction: $prediction Ground Truth: ${getLabel(train, imageId)}")
        }
        println("Accuracy: $accuracy")

        val amountOfOps = 1000
        val start = System.currentTimeMillis()
        for (i in 0..amountOfOps) {
            it.predict(train.getImage(i % 50000))
        }
        println("Time, s: ${(System.currentTimeMillis() - start) / 1000f}")
        println("Throughput, op/s: ${amountOfOps / ((System.currentTimeMillis() - start) / 1000f)}")
    }
}
