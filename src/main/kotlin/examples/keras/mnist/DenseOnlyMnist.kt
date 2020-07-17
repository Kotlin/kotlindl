package examples.keras.mnist

import api.keras.Sequential
import api.keras.activations.Activations
import api.keras.dataset.ImageDataset
import api.keras.initializers.HeNormal
import api.keras.initializers.Zeros
import api.keras.layers.Dense
import api.keras.layers.Input
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import examples.keras.mnist.util.*

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 1000
private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 500

private val model = Sequential.of<Float>(
    Input(784),
    Dense(1024, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(1024, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(1024, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(128, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(10, Activations.Softmax, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros())
)

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

    model.compile(optimizer = Adam(), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

    model.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

    val accuracy = model.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

    model.close()

    println("Accuracy: $accuracy")
}
