package examples.production

import api.inference.InferenceModel
import api.keras.dataset.ImageDataset
import examples.keras.mnist.util.*

private const val PATH_TO_MODEL = "fashionLenet"

fun main() {
    val (train, test) = ImageDataset.createTrainAndTestDatasets(
        TRAIN_IMAGES_ARCHIVE,
        TRAIN_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
        NUM_LABELS,
        ::extractImages,
        ::extractLabels
    )

    val imageId1 = 0
    val imageId2 = 1
    val imageId3 = 2

    InferenceModel().use {
        it.load(PATH_TO_MODEL)

        val prediction = it.predict(train.getImage(imageId1))

        println("Prediction: $prediction Ground Truth: ${getLabel(train, imageId1)}")

        val prediction2 = it.predict(train.getImage(imageId2))

        println("Prediction: $prediction2 Ground Truth: ${getLabel(train, imageId2)}")

        val prediction3 = it.predict(train.getImage(imageId3))

        println("Prediction: $prediction3 Ground Truth: ${getLabel(train, imageId3)}")

        val amountOfOps = 1000
        val start = System.currentTimeMillis()
        for (i in 0..amountOfOps) {
            it.predict(train.getImage(i % 50000))
        }
        println("Time, s: ${(System.currentTimeMillis() - start) / 1000f}")
        println("Throughput, op/s: ${amountOfOps / ((System.currentTimeMillis() - start) / 1000f)}")
    }
}
