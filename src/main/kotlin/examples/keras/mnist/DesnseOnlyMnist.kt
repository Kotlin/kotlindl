package examples.keras.mnist

import examples.keras.mnist.util.*
import tf_api.keras.Sequential
import tf_api.keras.activations.Activations
import tf_api.keras.dataset.ImageDataset
import tf_api.keras.initializers.TruncatedNormal
import tf_api.keras.initializers.Zeros
import tf_api.keras.layers.Dense
import tf_api.keras.layers.Input
import tf_api.keras.loss.LossFunctions
import tf_api.keras.metric.Metrics
import tf_api.keras.optimizers.SGD

private const val VALIDATION_SIZE = 0

private val model = Sequential.of<Float>(
    Input(784),
    Dense(1024, Activations.Relu, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros()),
    Dense(1024, Activations.Relu, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros()),
    Dense(1024, Activations.Relu, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros()),
    Dense(128, Activations.Relu, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros()),
    Dense(10, Activations.Softmax, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros())
)

fun main() {
    val dataset = ImageDataset.create(
        TRAIN_IMAGES_ARCHIVE,
        TRAIN_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
        NUM_LABELS,
        examples.keras.mnist.util.VALIDATION_SIZE,
        ::extractImages,
        ::extractLabels
    )

    val (train, test) = dataset.split(0.75)

    model.compile(optimizer = SGD(0.1f), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

    model.fit(trainDataset = train, epochs = 10, batchSize = 100, isDebugMode = false)

    val accuracy = model.evaluate(testDataset = test, metric = Metrics.ACCURACY, batchSize = -1)

    model.close()

    println("Accuracy: $accuracy")
}
