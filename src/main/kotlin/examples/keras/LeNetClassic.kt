package examples.keras

import tensorflow.training.util.ImageDataset
import tf_api.keras.Metric
import tf_api.keras.Sequential
import tf_api.keras.activations.Activations
import tf_api.keras.initializers.Constant
import tf_api.keras.initializers.TruncatedNormal
import tf_api.keras.initializers.Zeros
import tf_api.keras.layers.Dense
import tf_api.keras.layers.Flatten
import tf_api.keras.layers.Input
import tf_api.keras.layers.twodim.Conv2D
import tf_api.keras.layers.twodim.ConvPadding
import tf_api.keras.layers.twodim.MaxPool2D
import tf_api.keras.loss.LossFunctions
import tf_api.keras.optimizers.SGD


private const val LEARNING_RATE = 0.2f
private const val EPOCHS = 20
private const val TRAINING_BATCH_SIZE = 500

private const val NUM_LABELS = 10
private const val PIXEL_DEPTH = 255f
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val VALIDATION_SIZE = 0

private const val SEED = 12L
private const val PADDING_TYPE = "SAME"
private const val INPUT_NAME = "input"
private const val OUTPUT_NAME = "output"
private const val TRAINING_LOSS = "training_loss"

/**
 * Kotlin implementation of LeNet on Keras.
 * https://github.com/TaavishThaman/LeNet-5-with-Keras/blob/master/lenet_5.py
 */
private val model = Sequential.of<Float>(
    Input(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
    Conv2D(
        filters = 32,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
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
    MaxPool2D(
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


private val modelOld = Sequential.of<Float>(
    Input(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
    Conv2D(
        filters = 32,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Conv2D(
        filters = 64,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Flatten(), // 3136
    Dense(
        outputSize = 120,
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Constant(0.1f)
    ),
    Dense(
        outputSize = 84,
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
    val dataset = ImageDataset.create(VALIDATION_SIZE)
    val (train, test) = dataset.split(0.75)
    model.compile(optimizer = SGD(LEARNING_RATE), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

    model.fit(trainDataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

    val accuracy = model.evaluate(testDataset = test, metric = Metric.ACCURACY)

    model.close()

    println("Accuracy: $accuracy")
}
