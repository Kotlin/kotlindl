package examples.keras.mnist

import api.core.Sequential
import api.core.activation.Activations
import api.core.initializer.HeNormal
import api.core.initializer.Zeros
import api.core.layer.Dense
import api.core.layer.Input
import api.core.loss.LossFunctions
import api.core.metric.Metrics
import api.core.optimizer.Adam
import datasets.Dataset
import datasets.handlers.*

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 1000
private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 500

private val model = Sequential.of(
    Input(784),
    Dense(1024, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(1024, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(1024, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(128, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(10, Activations.Softmax, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros())
)

fun main() {
    val (train, test) = Dataset.createTrainAndTestDatasets(
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
