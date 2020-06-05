package examples.production

import api.keras.dataset.ImageDataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.SGD
import examples.keras.mnist.util.*

private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000

private val learningSchedule = mapOf(
    1 to 0.1f,
    2 to 0.05f,
    3 to 0.025f,
    4 to 0.01f,
    5 to 0.005f,
    6 to 0.0025f,
    7 to 0.001f,
    8 to 0.001f,
    9 to 0.001f,
    10 to 0.0005f
)

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

    val imageId = 0
    lenet5.use {
        it.compile(optimizer = SGD(learningSchedule), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

        val accuracy = it.evaluate(dataset = test, metric = Metrics.ACCURACY, batchSize = TEST_BATCH_SIZE)

        println("Accuracy $accuracy")

        val prediction = it.predict(train.getImage(imageId))

        println("Prediction: $prediction")

        val trainImageLabel = train.getImageLabel(imageId)

        val maxIdx = trainImageLabel.indexOf(trainImageLabel.max()!!)

        println("Ground Truth: $maxIdx")
    }
}
