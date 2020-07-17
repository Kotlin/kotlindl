package examples.production

import api.inference.InferenceModel
import api.keras.dataset.ImageDataset
import examples.keras.mnist.util.*

private const val PATH_TO_MODEL = "savedmodels/lenet5"

fun main() {
    val (train, test) = ImageDataset.createTrainAndTestDatasets(
        TRAIN_IMAGES_ARCHIVE,
        TRAIN_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )

    val imageId1 = 0
    val imageId2 = 1
    val imageId3 = 2

    InferenceModel<Float>().use {
        it.load(PATH_TO_MODEL)

        val prediction = it.predict(train.getImage(imageId1))

        println("Prediction: $prediction Ground Truth: ${getLabel(train, imageId1)}")

        val prediction2 = it.predict(train.getImage(imageId2))

        println("Prediction: $prediction2 Ground Truth: ${getLabel(train, imageId2)}")

        val prediction3 = it.predict(train.getImage(imageId3))

        println("Prediction: $prediction3 Ground Truth: ${getLabel(train, imageId3)}")
    }
}
