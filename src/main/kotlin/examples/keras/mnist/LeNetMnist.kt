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

private const val LEARNING_RATE = 0.05f
private const val EPOCHS = 4
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L

/**
 * Kotlin implementation of LeNet on Keras.
 * Architecture could be copied here: https://github.com/TaavishThaman/LeNet-5-with-Keras/blob/master/lenet_5.py
 */
private val model = Sequential.of<Float>(
    Input(
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS
    ),
    Conv2D(
        filters = 32,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Conv2D(
        filters = 64,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Flatten(), // 3136
    Dense(
        outputSize = 512,
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Constant(0.1f)
    ),
    Dense(
        outputSize = NUM_LABELS,
        activation = Activations.Linear, // TODO: https://stats.stackexchange.com/questions/348036/difference-between-mathematical-and-tensorflow-implementation-of-softmax-crossen
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Constant(0.1f)
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

    model.use {
        it.compile(optimizer = SGD(LEARNING_RATE), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

        val accuracy = it.evaluate(dataset = test, metric = Metrics.MAE, batchSize = TEST_BATCH_SIZE)

        println("Accuracy: $accuracy")
    }
}
