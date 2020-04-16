package examples.keras

import org.tensorflow.Graph
import org.tensorflow.op.Ops
import tensorflow.training.util.ImageDataset
import tf_api.keras.Metric
import tf_api.keras.Sequential
import tf_api.keras.activations.Activations
import tf_api.keras.initializers.Ones
import tf_api.keras.initializers.TruncatedNormal
import tf_api.keras.initializers.Zeros
import tf_api.keras.layers.*
import tf_api.keras.loss.LossFunctions
import tf_api.keras.optimizers.Optimizers


private const val LEARNING_RATE = 0.2f
private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 500

private const val NUM_LABELS = 10L
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
    Source(28, 28),
    Conv2D(
        filterShape = longArrayOf(5, 5, 1, 6),
        strides = longArrayOf(1, 1),
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(123L),
        biasInitializer = Ones()
    ),
    AvgPool(poolSize = intArrayOf(2, 2), strides = intArrayOf(2, 2)),
    Conv2D(
        filterShape = longArrayOf(5, 5, 6, 16),
        strides = longArrayOf(1, 1),
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(123L),
        biasInitializer = Ones()
    ),
    AvgPool(poolSize = intArrayOf(2, 2), strides = intArrayOf(2, 2)),
    Flatten(), // 400
    Dense(
        outputSize = 120,
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(123L),
        biasInitializer = Zeros()
    ),
    Dense(
        outputSize = 84,
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(123L),
        biasInitializer = Zeros()
    ),
    Dense(
        outputSize = 10,
        activation = Activations.Softmax,
        kernelInitializer = TruncatedNormal(123L),
        biasInitializer = Zeros()
    )
)

fun main() {
    val dataset = ImageDataset.create(VALIDATION_SIZE)
    val (train, test) = dataset.split(0.75)

    Graph().use { graph ->
        val tf = Ops.create(graph)

        model.compile(tf, optimizer = Optimizers.SGD, loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        model.fit(graph, tf, trainDataset = train, epochs = 100, batchSize = 1000)

        val accuracy = model.evaluate(testDataset = test, metric = Metric.ACCURACY)

        println("Accuracy: $accuracy")
    }
}
