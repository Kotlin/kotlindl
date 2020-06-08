package examples.keras.cifar10

import api.keras.Sequential
import api.keras.activations.Activations
import api.keras.dataset.ImageDataset
import api.keras.initializers.YetAnotherXavier
import api.keras.initializers.Zeros
import api.keras.layers.Dense
import api.keras.layers.Flatten
import api.keras.layers.Input
import api.keras.layers.twodim.Conv2D
import api.keras.layers.twodim.ConvPadding
import api.keras.layers.twodim.MaxPool2D
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.SGD
import examples.keras.cifar10.util.IMAGES_ARCHIVE
import examples.keras.cifar10.util.LABELS_ARCHIVE
import examples.keras.cifar10.util.extractCifar10Images
import examples.keras.cifar10.util.extractCifar10Labels

private const val LEARNING_RATE = 0.1f
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000
private const val NUM_LABELS = 10
private const val NUM_CHANNELS = 3L
private const val IMAGE_SIZE = 32L
private const val VALIDATION_RATE = 0.75
private const val SEED = 12L

private val vgg11 = Sequential.of<Float>(
    Input(
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS
    ),
    Conv2D(
        filters = 32,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Elu,
        kernelInitializer = YetAnotherXavier(SEED),
        biasInitializer = YetAnotherXavier(SEED),
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
        activation = Activations.Elu,
        kernelInitializer = YetAnotherXavier(SEED),
        biasInitializer = YetAnotherXavier(SEED),
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Conv2D(
        filters = 128,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Elu,
        kernelInitializer = YetAnotherXavier(SEED),
        biasInitializer = YetAnotherXavier(SEED),
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 128,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Elu,
        kernelInitializer = YetAnotherXavier(SEED),
        biasInitializer = YetAnotherXavier(SEED),
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Conv2D(
        filters = 256,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Elu,
        kernelInitializer = YetAnotherXavier(SEED),
        biasInitializer = YetAnotherXavier(SEED),
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 256,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Elu,
        kernelInitializer = YetAnotherXavier(12L),
        biasInitializer = YetAnotherXavier(SEED),
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Conv2D(
        filters = 128,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Elu,
        kernelInitializer = YetAnotherXavier(SEED),
        biasInitializer = YetAnotherXavier(SEED),
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 128,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Elu,
        kernelInitializer = YetAnotherXavier(SEED),
        biasInitializer = YetAnotherXavier(SEED),
        padding = ConvPadding.SAME
    ),
    /*MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),*/
    Flatten(),
    /*Dense(
        outputSize = 4096,
        activation = Activations.Relu,
        kernelInitializer = Zeros(),
        biasInitializer = Zeros()
    ),*/
    /*Dense(
         outputSize = 2048,
         activation = Activations.Relu,
         kernelInitializer = Xavier(12L),
         biasInitializer = Zeros()
     ),*/
    Dense(
        outputSize = 1000,
        activation = Activations.Elu,
        kernelInitializer = YetAnotherXavier(12L),
        biasInitializer = Zeros()
    ),
    Dense(
        outputSize = NUM_LABELS, // changed due to 10 classes instead of
        activation = Activations.Linear,
        kernelInitializer = Zeros(),
        biasInitializer = Zeros()
    )
)

fun main() {
    val dataset = ImageDataset.create(
        IMAGES_ARCHIVE,
        LABELS_ARCHIVE,
        NUM_LABELS,
        ::extractCifar10Images,
        ::extractCifar10Labels
    )

    val (train, test) = dataset.split(VALIDATION_RATE)

    vgg11.use {
        it.compile(optimizer = SGD(LEARNING_RATE), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

        it.save("vgg11")

        val accuracy = it.evaluate(dataset = test, metric = Metrics.ACCURACY, batchSize = TEST_BATCH_SIZE)

        println("Accuracy: $accuracy")
    }
}
