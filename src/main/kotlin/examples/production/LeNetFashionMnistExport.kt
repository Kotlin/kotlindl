package examples.production

import api.keras.dataset.ImageDataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.SGD
import examples.keras.fashionmnist.util.*
import examples.keras.mnist.util.NUM_LABELS

private const val EPOCHS = 2
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000

private val learningSchedule = mapOf(
    1 to 0.1f,
    2 to 0.05f,
    3 to 0.025f,
    4 to 0.01f,
    5 to 0.005f,
    6 to 0.0025f,
    7 to 0.001f
)

private val fashionMnistLabelEncoding = mapOf(
    0 to "T-shirt/top",
    1 to "Trouser",
    2 to "Pullover",
    3 to "Dress",
    4 to "Coat",
    5 to "Sandal",
    6 to "Shirt",
    7 to "Sneaker",
    8 to "Bag",
    9 to "Ankle boot"
)

fun main() {
    val (train, test) = ImageDataset.createTrainAndTestDatasets(
        FASHION_TRAIN_IMAGES_ARCHIVE,
        FASHION_TRAIN_LABELS_ARCHIVE,
        FASHION_TEST_IMAGES_ARCHIVE,
        FASHION_TEST_LABELS_ARCHIVE,
        NUM_LABELS,
        ::extractFashionImages,
        ::extractFashionLabels
    )

    val imageId = 0

    lenet5.use {
        it.compile(optimizer = SGD(learningSchedule), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

        val accuracy = it.evaluate(dataset = test, metric = Metrics.ACCURACY, batchSize = TEST_BATCH_SIZE)

        println("Accuracy $accuracy")

        println("Ground Truth: ${fashionMnistLabelEncoding[getLabel(train, imageId)]}")

        val prediction = it.predict(train.getImage(imageId))

        println("Prediction: ${fashionMnistLabelEncoding[prediction]}")

        it.save("savedmodels/fashionLenet")
    }
}