package examples.keras.mnist

import api.keras.Sequential
import api.keras.activations.Activations
import api.keras.dataset.ImageDataset
import api.keras.initializers.TruncatedNormal
import api.keras.initializers.Zeros
import api.keras.layers.Dense
import api.keras.layers.Input
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.SGD
import examples.keras.mnist.util.*

private val model = Sequential.of<Float>(
    Input(784),
    Dense(1024, Activations.Relu, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros()),
    Dense(1024, Activations.Relu, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros()),
    Dense(1024, Activations.Relu, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros()),
    Dense(128, Activations.Relu, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros()),
    Dense(10, Activations.Softmax, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros())
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

    model.compile(optimizer = SGD(0.1f), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

    model.fit(dataset = train, epochs = 10, batchSize = 100, verbose = true)

    val accuracy = model.evaluate(dataset = test, metric = Metrics.ACCURACY, batchSize = -1)

    model.close()

    println("Accuracy: $accuracy")
}
