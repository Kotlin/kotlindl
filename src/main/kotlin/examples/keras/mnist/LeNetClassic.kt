package examples.keras.mnist

import examples.keras.mnist.util.*
import tf_api.keras.Sequential
import tf_api.keras.activations.Activations
import tf_api.keras.dataset.ImageDataset
import tf_api.keras.initializers.Constant
import tf_api.keras.initializers.TruncatedNormal
import tf_api.keras.initializers.Zeros
import tf_api.keras.layers.Dense
import tf_api.keras.layers.Flatten
import tf_api.keras.layers.Input
import tf_api.keras.layers.twodim.AvgPool2D
import tf_api.keras.layers.twodim.Conv2D
import tf_api.keras.layers.twodim.ConvPadding
import tf_api.keras.loss.LossFunctions
import tf_api.keras.metric.Metrics
import tf_api.keras.optimizers.SGD

private const val LEARNING_RATE = 0.1f
private const val EPOCHS = 20
private const val TRAINING_BATCH_SIZE = 1000
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L
private const val TEST_BATCH_SIZE = 1000

private val lenet5Classic = Sequential.of<Float>(
    Input(
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS
    ),
    Conv2D(
        filters = 6,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Tanh,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME,
        name = "1"
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Conv2D(
        filters = 16,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Tanh,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME,
        name = "2"
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Flatten(), // 3136
    Dense(
        outputSize = 120,
        activation = Activations.Tanh,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Constant(0.1f),
        name = "3"
    ),
    Dense(
        outputSize = 84,
        activation = Activations.Tanh,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Constant(0.1f),
        name = "4"
    ),
    Dense(
        outputSize = NUM_LABELS,
        activation = Activations.Linear,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Constant(0.1f),
        name = "5"
    )
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

    lenet5Classic.compile(optimizer = SGD(LEARNING_RATE), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

    lenet5Classic.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

    val accuracy = lenet5Classic.evaluate(dataset = test, metric = Metrics.ACCURACY, batchSize = TEST_BATCH_SIZE)

    lenet5Classic.close()

    println("Accuracy: $accuracy")
}
