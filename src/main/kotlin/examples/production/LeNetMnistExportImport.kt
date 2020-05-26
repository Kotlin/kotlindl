package examples.production

import examples.keras.mnist.util.*
import tf_api.inference.InferenceModel
import tf_api.keras.dataset.ImageDataset
import tf_api.keras.loss.LossFunctions
import tf_api.keras.metric.Metrics
import tf_api.keras.optimizers.SGD

private const val PATH_TO_MODEL = "lenet5"
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000

private val LEARNING_SCHEDULE = mapOf(
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

    val imageId1 = 0
    val imageId2 = 1
    val imageId3 = 2

    lenet5.use {
        it.compile(optimizer = SGD(LEARNING_SCHEDULE), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

        it.save(PATH_TO_MODEL)

        val prediction = it.predict(train.getImage(imageId1))

        println("Prediction: $prediction Ground Truth: ${getLabel(train, imageId1)}")

        val prediction2 = it.predict(train.getImage(imageId2))

        println("Prediction: $prediction2 Ground Truth: ${getLabel(train, imageId2)}")

        val prediction3 = it.predict(train.getImage(imageId3))

        println("Prediction: $prediction3 Ground Truth: ${getLabel(train, imageId3)}")

        val accuracy = it.evaluate(dataset = test, metric = Metrics.ACCURACY, batchSize = TEST_BATCH_SIZE)
        println("Accuracy $accuracy")
    }

    InferenceModel().use {
        it.load(PATH_TO_MODEL)

        val prediction = it.predict(train.getImage(imageId1))

        println("Prediction: $prediction Ground Truth: ${getLabel(train, imageId1)}")

        val prediction2 = it.predict(train.getImage(imageId2))

        println("Prediction: $prediction2 Ground Truth: ${getLabel(train, imageId2)}")

        val prediction3 = it.predict(train.getImage(imageId3))

        println("Prediction: $prediction3 Ground Truth: ${getLabel(train, imageId3)}")
    }
}