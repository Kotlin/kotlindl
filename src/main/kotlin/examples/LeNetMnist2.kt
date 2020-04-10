package examples

import org.tensorflow.Graph
import org.tensorflow.op.Ops
import tensorflow.training.util.ImageDataset
import tf_api.blocks.*
import tf_api.blocks.layers.*


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


/*
var classifier = Sequential {
    Conv2D<Float>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Flatten<Float>()
    Dense<Float>(inputSize: 400, outputSize: 120, activation: relu)
    Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    Dense<Float>(inputSize: 84, outputSize: 10)
}
 */

/*val model = Sequential.of(
    Flatten(),
    Dense(
        128,
        activation = Activation.Relu,
        kernelInitializer = Initializer.TRUNCATED_NORMAL,
        biasInitializer = Initializer.ONES
    ),
    Dense(
        10,
        activation = Activation.Softmax,
        kernelInitializer = Initializer.TRUNCATED_NORMAL,
        biasInitializer = Initializer.ONES
    )
)*/


val model = Sequential.of(
    Source(28, 28),
    Conv2D(filterShape = intArrayOf(5, 5, 1, 6), padding = intArrayOf(1, 1), activation = Activation.Relu),
    AvgPool(poolSize = intArrayOf(2, 2), strides = intArrayOf(2, 2)),
    Conv2D(filterShape = intArrayOf(5, 5, 6, 16), padding = intArrayOf(1, 1), activation = Activation.Relu),
    AvgPool(poolSize = intArrayOf(2, 2), strides = intArrayOf(2, 2)),
    Flatten(),
    Dense(outputSize = 120, activation = Activation.Relu),
    Dense(outputSize = 84, activation = Activation.Relu),
    Dense(outputSize = 10)
)


fun main() {
    val dataset = ImageDataset.create(VALIDATION_SIZE)
    val (train, test) = dataset.split(0.75)

    Graph().use { graph ->
        val tf = Ops.create(graph)

        model.compile(optimizer = Optimizer.SGD, loss = LossFunction.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        model.fit(tf, trainDataset = train, epochs = 100, batchSize = 1000)

        val accuracy = model.evaluate(testDataset = test, metric = Metric.ACCURACY)

        println("Accuracy: $accuracy")
    }
}
