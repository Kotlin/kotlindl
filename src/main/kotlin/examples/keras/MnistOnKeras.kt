package examples.keras

import tensorflow.training.util.ImageDataset
import tf_api.keras.Sequential
import tf_api.keras.activations.Activations
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
    Dense(128, Activations.Sigmoid, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros()),
    Dense(10, Activations.Softmax, kernelInitializer = TruncatedNormal(123L), biasInitializer = Zeros())
)

fun main() {
    val dataset = ImageDataset.create(VALIDATION_SIZE)
    val (train, test) = dataset.split(0.75)

    model.compile(optimizer = SGD(0.2f), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

    model.fit(trainDataset = train, epochs = 1, batchSize = 100)

    val accuracy = model.evaluate(testDataset = test, metric = Metrics.ACCURACY)

    model.close()

    println("Accuracy: $accuracy")
}
