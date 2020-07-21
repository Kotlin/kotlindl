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
    }
}
