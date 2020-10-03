package examples.keras.cifar10

import api.core.Sequential
import api.core.activation.Activations
import api.core.initializer.HeNormal
import api.core.initializer.Zeros
import api.core.layer.Dense
import api.core.layer.Flatten
import api.core.layer.Input
import api.core.layer.twodim.Conv2D
import api.core.layer.twodim.ConvPadding
import api.core.layer.twodim.MaxPool2D
import api.core.loss.Losses
import api.core.metric.Metrics
import api.core.optimizer.SGD
import datasets.Dataset
import java.io.File

private const val PATH_TO_MODEL = "savedmodels/vgg11"
private const val LEARNING_RATE = 0.1f
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000
private const val NUM_LABELS = 10
private const val NUM_CHANNELS = 3L
private const val IMAGE_SIZE = 32L
private const val VALIDATION_RATE = 0.75
private const val SEED = 12L

private val vgg11 = Sequential.of(
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
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeNormal(SEED),
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
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeNormal(SEED),
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
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeNormal(SEED),
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 128,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Elu,
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeNormal(SEED),
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
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeNormal(SEED),
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 256,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Elu,
        kernelInitializer = HeNormal(12L),
        biasInitializer = HeNormal(SEED),
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
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeNormal(SEED),
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 128,
        kernelSize = longArrayOf(3, 3),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Elu,
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeNormal(SEED),
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
        kernelInitializer = HeNormal(SEED),
        biasInitializer = Zeros()
    ),
    Dense(
        outputSize = NUM_LABELS, // changed due to 10 classes instead of
        activation = Activations.Linear,
        kernelInitializer = HeNormal(SEED),
        biasInitializer = Zeros()
    )
)

fun main() {
    val dataset = Dataset.create(
        IMAGES_ARCHIVE,
        LABELS_ARCHIVE,
        NUM_LABELS,
        ::extractCifar10Images,
        ::extractCifar10Labels
    )

    val (train, test) = dataset.split(VALIDATION_RATE)

    vgg11.use {
        it.compile(
            optimizer = SGD(LEARNING_RATE),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

        it.save(File(PATH_TO_MODEL))

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")
    }
}
