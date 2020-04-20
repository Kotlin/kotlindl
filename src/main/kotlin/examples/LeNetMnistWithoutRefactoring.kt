package examples

import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Slice
import org.tensorflow.op.core.Variable
import org.tensorflow.op.random.TruncatedNormal
import tensorflow.training.util.ImageBatch
import tensorflow.training.util.ImageDataset
import tf_api.keras.TensorShape

private const val LEARNING_RATE = 0.2f
private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 500

private const val NUM_LABELS = 10L
private const val PIXEL_DEPTH = 255f
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val VALIDATION_SIZE = 0

private const val SEED = 12L
const val PADDING_TYPE = "SAME"
private const val INPUT_NAME = "input"
private const val OUTPUT_NAME = "output"
private const val TRAINING_LOSS = "training_loss"

fun main() {
    val dataset =
        ImageDataset.create(VALIDATION_SIZE)

    Graph().use { graph ->
        val tf = Ops.create(graph)

        // Define placeholders
        val images = tf.withName(INPUT_NAME).placeholder(
            Float::class.javaObjectType,
            Placeholder.shape(
                Shape.make(
                    -1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    NUM_CHANNELS
                )
            )
        )

        val labels = tf.placeholder(Float::class.javaObjectType)

        // First conv layer

        // Generate random data to fill the weight matrix
        val truncatedNormal = tf.random.truncatedNormal(
            tf.constant(longArrayOf(5, 5, NUM_CHANNELS, 32)),
            Float::class.javaObjectType,
            TruncatedNormal.seed(SEED)
        )

        val conv1Weights: Variable<Float> =
            tf.variable(Shape.make(5L, 5L, NUM_CHANNELS, 32), Float::class.javaObjectType)

        println("Conv1W" + TensorShape(conv1Weights.ref().shape()).dims().contentToString())

        val conv1WeightsInit = tf.assign(conv1Weights, tf.math.mul(truncatedNormal, tf.constant(0.1f)))

        val conv1 = tf.nn.conv2d(
            images, conv1Weights, mutableListOf(1L, 1L, 1L, 1L),
            PADDING_TYPE
        );

        println("Conv1" + TensorShape(conv1.output().shape()).dims().contentToString())

        val conv1Biases: Variable<Float> = tf.variable(Shape.make(32), Float::class.javaObjectType)

        println("conv1Biases" + TensorShape(conv1Biases.ref().shape()).dims().contentToString())

        val conv1BiasesInit = tf.assign(
            conv1Biases, tf.zeros(
                constArray(
                    tf,
                    32
                ), Float::class.javaObjectType
            )
        )

        val relu1 = tf.nn.relu(tf.nn.biasAdd(conv1, conv1Biases))

        println("relu1" + TensorShape(relu1.asOutput().shape()).dims().contentToString())

        // First pooling layer
        val pool1 = tf.nn.maxPool(
            relu1,
            tf.constant(intArrayOf(1, 2, 2, 1)),
            tf.constant(intArrayOf(1, 2, 2, 1)),
            PADDING_TYPE
        )

        println("pool1" + TensorShape(pool1.output().shape()).dims().contentToString())

        // Second conv layer
        val truncatedNormal2 = tf.random.truncatedNormal(
            tf.constant(longArrayOf(5, 5, 32, 64)),
            Float::class.javaObjectType,
            TruncatedNormal.seed(SEED)
        )

        val conv2Weights: Variable<Float> =
            tf.variable(Shape.make(5, 5, 32, 64), Float::class.javaObjectType)

        println("Conv2W" + TensorShape(conv2Weights.ref().shape()).dims().contentToString())

        val conv2WeightsInit = tf.assign(conv2Weights, tf.math.mul(truncatedNormal2, tf.constant(0.1f)))

        val conv2 = tf.nn.conv2d(
            pool1, conv2Weights, mutableListOf(1L, 1L, 1L, 1L),
            PADDING_TYPE
        );

        println("conv2" + TensorShape(conv2.output().shape()).dims().contentToString())

        val conv2Biases: Variable<Float> = tf.variable(Shape.make(64), Float::class.javaObjectType)

        val conv2BiasesInit = tf.assign(
            conv2Biases, tf.zeros(
                constArray(
                    tf,
                    64
                ), Float::class.javaObjectType
            )
        )

        val relu2 = tf.nn.relu(tf.nn.biasAdd(conv2, conv2Biases))

        // Second pooling layer
        val pool2 = tf.nn.maxPool(
            relu2,
            tf.constant(intArrayOf(1, 2, 2, 1)),
            tf.constant(intArrayOf(1, 2, 2, 1)),
            PADDING_TYPE
        )

        println("pool2" + TensorShape(pool2.output().shape()).dims().contentToString())

        // Flatten inputs
        val slice: Slice<Int> = tf.slice(
            tf.shape(pool2),
            tf.constant(intArrayOf(0)),
            tf.constant(intArrayOf(1))
        )

        val mutableListOf: MutableList<Operand<Int>> = mutableListOf(slice, tf.constant(intArrayOf(-1)))

        val flatten = tf.reshape(
            pool2,
            tf.concat(
                mutableListOf,
                tf.constant(0)
            )
        )

        println("flatten" + TensorShape(flatten.output().shape()).dims().contentToString())

        // Fully connected layer
        val truncatedNormal3 = tf.random.truncatedNormal(
            tf.constant(longArrayOf(IMAGE_SIZE * IMAGE_SIZE * 4, 512)),
            Float::class.javaObjectType,
            TruncatedNormal.seed(SEED)
        )

        val fc1Weights: Variable<Float> =
            tf.variable(Shape.make(IMAGE_SIZE * IMAGE_SIZE * 4, 512), Float::class.javaObjectType)

        val fc1WeightsInit = tf.assign(fc1Weights, tf.math.mul(truncatedNormal3, tf.constant(0.1f)))

        val fc1Biases: Variable<Float> = tf.variable(Shape.make(512), Float::class.javaObjectType)

        val fc1BiasesInit = tf.assign(fc1Biases, tf.fill(tf.constant(intArrayOf(512)), tf.constant(0.1f)))

        val relu3 = tf.nn.relu(tf.math.add(tf.linalg.matMul(flatten, fc1Weights), fc1Biases))

        println("fc1WeightsInit" + TensorShape(fc1WeightsInit.outputRef().shape()).dims().contentToString())

        // Softmax layer
        val truncatedNormal4 = tf.random.truncatedNormal(
            tf.constant(longArrayOf(512, NUM_LABELS)),
            Float::class.javaObjectType,
            TruncatedNormal.seed(SEED)
        )

        val fc2Weights: Variable<Float> =
            tf.variable(Shape.make(512, NUM_LABELS), Float::class.javaObjectType)

        val fc2WeightsInit = tf.assign(fc2Weights, tf.math.mul(truncatedNormal4, tf.constant(0.1f)))

        println("fc2WeightsInit" + TensorShape(fc2WeightsInit.outputRef().shape()).dims().contentToString())


        val fc2Biases: Variable<Float> = tf.variable(Shape.make(NUM_LABELS), Float::class.javaObjectType)

        val fc2BiasesInit =
            tf.assign(fc2Biases, tf.fill(tf.constant(intArrayOf(NUM_LABELS.toInt())), tf.constant(0.1f)))

        val logits = tf.math.add(tf.linalg.matMul(relu3, fc2Weights), fc2Biases)

        // Predicted outputs
        val prediction = tf.withName(OUTPUT_NAME).nn.softmax(logits)

        // labels are one-hot due to preprocessing
        val batchLoss = tf.nn.softmaxCrossEntropyWithLogits(logits, labels)

        val loss = tf.withName(TRAINING_LOSS).math.mean(batchLoss.loss(), tf.constant(0))

        // Define gradients
        val learningRate = tf.constant(LEARNING_RATE)
        val variables =
            listOf(conv1Weights, conv1Biases, conv2Weights, conv2Biases, fc1Weights, fc1Biases, fc2Weights, fc2Biases)

        val gradients = tf.gradients(loss, variables)

        val conv1WeightsGD = tf.train.applyGradientDescent(variables[0], learningRate, gradients.dy<Float>(0))
        val conv1BiasesGD = tf.train.applyGradientDescent(variables[1], learningRate, gradients.dy<Float>(1))
        val conv2WeightsGD = tf.train.applyGradientDescent(variables[2], learningRate, gradients.dy<Float>(2))
        val conv2BiasesGD = tf.train.applyGradientDescent(variables[3], learningRate, gradients.dy<Float>(3))
        val fc1WeightsGD = tf.train.applyGradientDescent(variables[4], learningRate, gradients.dy<Float>(4))
        val fc1BiasesGD = tf.train.applyGradientDescent(variables[5], learningRate, gradients.dy<Float>(5))
        val fc2WeightsGD = tf.train.applyGradientDescent(variables[6], learningRate, gradients.dy<Float>(6))
        val fc2BiasesGD = tf.train.applyGradientDescent(variables[7], learningRate, gradients.dy<Float>(7))

        //printTFGraph(graph)

        Session(graph).use { session ->
            // Initialize graph variables
            session.runner()
                .addTarget(conv1WeightsInit)
                .addTarget(conv1BiasesInit)
                .addTarget(conv2WeightsInit)
                .addTarget(conv2BiasesInit)
                .addTarget(fc1WeightsInit)
                .addTarget(fc1BiasesInit)
                .addTarget(fc2WeightsInit)
                .addTarget(fc2BiasesInit)
                .run()
            for (i in 1..EPOCHS) {
                // Train the graph
                val batchIter: ImageDataset.ImageBatchIterator = dataset.trainingBatchIterator(
                    TRAINING_BATCH_SIZE
                )

                while (batchIter.hasNext()) {
                    val batch: ImageBatch = batchIter.next()
                    Tensor.create(
                        longArrayOf(
                            batch.size().toLong(),
                            IMAGE_SIZE,
                            IMAGE_SIZE,
                            NUM_CHANNELS
                        ),
                        batch.images()
                    ).use { batchImages ->
                        Tensor.create(longArrayOf(batch.size().toLong(), 10), batch.labels()).use { batchLabels ->
                            val lossValue = session.runner()
                                .addTarget(conv1WeightsGD)
                                .addTarget(conv1BiasesGD)
                                .addTarget(conv2WeightsGD)
                                .addTarget(conv2BiasesGD)
                                .addTarget(fc1WeightsGD)
                                .addTarget(fc1BiasesGD)
                                .addTarget(fc2WeightsGD)
                                .addTarget(fc2BiasesGD)
                                .feed(images.asOutput(), batchImages)
                                .feed(labels.asOutput(), batchLabels)
                                .fetch(TRAINING_LOSS)
                                .run()[0].floatValue()
                            println("epochs: $i lossValue: $lossValue")
                        }
                    }
                }
            }


            val predicted: Operand<Long> = tf.math.argMax(prediction, tf.constant(1))
            val expected: Operand<Long> = tf.math.argMax(labels, tf.constant(1))

            // Define multi-classification metric
            val accuracy = tf.math.mean(
                tf.dtypes.cast(
                    tf.math.equal(predicted, expected),
                    Float::class.javaObjectType
                ), constArray(tf, 0)
            )

            val testBatch: ImageBatch = dataset.testBatch()
            Tensor.create(
                longArrayOf(
                    testBatch.size().toLong(),
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    NUM_CHANNELS
                ),
                testBatch.images()
            ).use { testImages ->
                Tensor.create(testBatch.shape(10), testBatch.labels()).use { testLabels ->
                    session.runner()
                        .fetch(accuracy)
                        .feed(images.asOutput(), testImages)
                        .feed(labels.asOutput(), testLabels)
                        .run()[0].use { value -> println("Accuracy: " + value.floatValue()) }
                }
            }
        }
    }
}

fun constArray(tf: Ops, vararg i: Int): Operand<Int> {
    return tf.constant(i)
}