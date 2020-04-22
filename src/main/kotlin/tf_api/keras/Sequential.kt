package tf_api.keras

import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Variable
import tensorflow.training.util.ImageBatch
import tensorflow.training.util.ImageDataset
import tf_api.TFModel
import tf_api.keras.layers.Input
import tf_api.keras.layers.Layer
import tf_api.keras.loss.LossFunctions
import tf_api.keras.loss.SoftmaxCrossEntropyWithLogits
import tf_api.keras.metric.Metrics
import tf_api.keras.optimizers.Optimizer
import tf_api.keras.optimizers.SGD


private const val SEED = 12L
private const val PADDING_TYPE = "SAME"
private const val INPUT_NAME = "input"
private const val OUTPUT_NAME = "output"
private const val TRAINING_LOSS = "training_loss"
private const val NUM_LABELS = 10L
private const val PIXEL_DEPTH = 255f
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L

class Sequential<T : Number>(input: Input<T>, vararg layers: Layer<T>) : TFModel<T>() {


    private val firstLayer: Input<T> = input

    private val layers: List<Layer<T>> = listOf(*layers)

    private var trainableVars: MutableList<Variable<T>> = mutableListOf()

    private var initializers: MutableList<Operand<T>> = mutableListOf()

    private var optimizer: Optimizer<T> = SGD(0.2f)

    private var loss: LossFunctions = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS

    private var metrics: List<Metrics> = listOf(Metrics.ACCURACY)

    // Required for evaluation
    private lateinit var yPred: Operand<T>

    private lateinit var xOp: Operand<T>

    private lateinit var yOp: Operand<T>

    private var tf: Ops

    init {
        val graph = Graph()
        tf = Ops.create(graph)
        session = Session(graph)
    }

    override fun compile(optimizer: Optimizer<T>, loss: LossFunctions, metric: Metrics) {
        this.loss = loss
        this.metrics = listOf(metric) // handle multiple metrics
        this.optimizer = optimizer

        firstLayer.defineVariables(tf)
        var inputShape: Shape = firstLayer.computeOutputShape()


        layers.forEach {
            it.defineVariables(tf, inputShape = inputShape)

            trainableVars.addAll(it.variables.values)
            initializers.addAll(it.initializers.values)

            println(it.toString() + " " + TensorShape(inputShape).dims().contentToString())

            inputShape = it.computeOutputShape(inputShape)

            println(it.toString() + " " + TensorShape(inputShape).dims().contentToString())
        }
    }

    override fun fit(dataset: ImageDataset, epochs: Int, batchSize: Int) {
        xOp = tf.withName(INPUT_NAME).placeholder(
            Float::class.javaObjectType,
            Placeholder.shape(
                Shape.make(
                    -1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    NUM_CHANNELS
                )
            )
        ) as Operand<T>
        yOp = tf.placeholder(Float::class.javaObjectType) as Operand<T>

        // Compute Output / Loss / Accuracy
        yPred = transformInputWithNNModel(xOp)

        val imageShape = longArrayOf(
            batchSize.toLong(),
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS
        )

        val labelShape = longArrayOf(
            batchSize.toLong(),
            10
        )

        val loss = when (loss) {
            LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS -> SoftmaxCrossEntropyWithLogits<T>().getTFOperand(
                tf,
                yPred,
                yOp
            )
            else -> TODO("Implement it")
        }

        val targets = optimizer.prepareTargets(tf, loss, trainableVars)

            // Initialize graph variables
            val runner = session.runner()
            initializers.forEach {
                runner.addTarget(it)
            }
            runner.run()

            for (i in 1..epochs) {
                // Train the graph
                val batchIter: ImageDataset.ImageBatchIterator = dataset.trainingBatchIterator(
                    batchSize
                )
                while (batchIter.hasNext()) {
                    val batch: ImageBatch = batchIter.next()

                    Tensor.create(
                        imageShape,
                        batch.images()
                    ).use { batchImages ->
                        Tensor.create(labelShape, batch.labels()).use { batchLabels ->
                            trainOnEpoch(targets, batchImages, batchLabels, i, xOp, yOp)
                        }
                    }

                }
        }
    }

    override fun evaluate(testDataset: ImageDataset, metric: Metrics): Double {
        val imageShape = longArrayOf(
            10000,
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS
        )

        val prediction = tf.withName(examples.OUTPUT_NAME).nn.softmax(yPred)
        val yTrue: Operand<T> = yOp

        // Define multi-classification metric
        val metricOp = Metrics.convert<T>(metric).apply(tf, prediction, yTrue, getDType())

        val testBatch: ImageBatch = testDataset.testBatch()

        Tensor.create(
            imageShape,
            testBatch.images()
        ).use { testImages ->
            Tensor.create(testBatch.shape(NUM_LABELS.toInt()), testBatch.labels()).use { testLabels ->
                val metricValue = session.runner()
                    .fetch(metricOp)
                    .feed(xOp.asOutput(), testImages)
                    .feed(yOp.asOutput(), testLabels)
                    .run()[0]

                return metricValue.floatValue().toDouble()
            }
        }
    }

    private fun transformInputWithNNModel(input: Operand<T>): Operand<T> {
        var out: Operand<T> = input
        for (layer in layers) {
            out = layer.transformInput(tf, out)
        }
        return out
    }

    private fun trainOnEpoch(
        targets: List<Operand<T>>,
        batchImages: Tensor<Float>,
        batchLabels: Tensor<Float>,
        i: Int,
        xOp: Operand<T>,
        yOp: Operand<T>
    ) {
        val runner = session.runner()

        targets.forEach {
            runner.addTarget(it)
        }

        runner
            .feed(xOp.asOutput(), batchImages)
            .feed(yOp.asOutput(), batchLabels)

        runner
            .fetch(TRAINING_LOSS)

        val lossValue = runner.run()[0].floatValue()
        println("epochs: $i lossValue: $lossValue")

    }

    companion object {
        fun <T : Number> of(input: Input<T>, vararg layers: Layer<T>): TFModel<T> {
            return Sequential(input, *layers)
        }
    }

    override fun close() {
        session.close()
    }

    fun getDType(): Class<T> {
        return Float::class.javaObjectType as Class<T>
    }
}