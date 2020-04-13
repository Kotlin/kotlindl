package tf_api.blocks

import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Variable
import tensorflow.training.util.ImageBatch
import tensorflow.training.util.ImageDataset
import tf_api.TFModel
import tf_api.blocks.layers.Layer
import tf_api.blocks.loss.LossFunctions
import tf_api.blocks.loss.SoftmaxCrossEntropyWithLogits
import tf_api.blocks.optimizers.GradientDescentOptimizer
import tf_api.blocks.optimizers.Optimizers


private const val SEED = 12L
private const val PADDING_TYPE = "SAME"
private const val INPUT_NAME = "input"
private const val OUTPUT_NAME = "output"
private const val TRAINING_LOSS = "training_loss"
private const val NUM_LABELS = 10L
private const val PIXEL_DEPTH = 255f
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L

class Sequential<T : Number>(vararg layers: Layer<T>) : TFModel<T>() {
    private val layers: List<Layer<T>> = listOf(*layers)

    private var trainableVars: MutableList<Variable<T>> = mutableListOf()

    private var initializers: MutableList<Operand<T>> = mutableListOf()

    private var optimizer: Optimizers = Optimizers.SGD

    private var loss: LossFunctions = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS

    private var metrics: List<Metric> = listOf(Metric.ACCURACY)

    override fun compile(optimizer: Optimizers, loss: LossFunctions, metric: Metric) {
        this.loss = loss
        this.metrics = listOf(metric) // handle multiple metrics
        this.optimizer = optimizer

        layers.forEach {
            trainableVars.addAll(it.variables.values)
            initializers.addAll(it.initializers.values)
        }
    }

    override fun fit(graph: Graph, dataset: ImageDataset, epochs: Int, batchSize: Int) {
        val tf = Ops.create(graph)

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


        val imageShape = longArrayOf(
            //batch.size().toLong(),
            //IMAGE_SIZE,
            //IMAGE_SIZE,
            //NUM_CHANNELS
        )

        val labelShape = longArrayOf(
            //batch.size().toLong(),
            10
        )

        val loss = when (loss) {
            LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS -> SoftmaxCrossEntropyWithLogits<Float>().getTFOperand(
                tf,
                logits,
                labels
            )
            else -> TODO("Implement it")
        }

        val targets = when (optimizer) {
            Optimizers.SGD -> GradientDescentOptimizer<T>(0.2f).prepareTargets(tf, loss, trainableVars)
            else -> TODO("Implement it")
        }

        Session(graph).use { session ->
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
                            trainOnEpoch(session, targets, tf, batchImages, batchLabels, i)
                        }
                    }

                }
            }
        }
    }

    private fun trainOnEpoch(
        session: Session,
        targets: List<Operand<T>>,
        tf: Ops,
        batchImages: Tensor<Float>?,
        batchLabels: Tensor<Float>?,
        i: Int
    ) {
        val runner = session.runner()

        targets.forEach {
            runner.addTarget(it)
        }

        feedRunner(tf, runner, batchImages, batchLabels)

        runner
            .fetch(TRAINING_LOSS)

        val lossValue = runner.run()[0].floatValue()
        println("epochs: $i lossValue: $lossValue")
    }

    private fun feedRunner(
        tf: Ops,
        runner: Session.Runner,
        batchImages: Tensor<Float>?,
        batchLabels: Tensor<Float>?
    ) {

        // TODO: remove after correct input() layer
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

        runner
            .feed(images.asOutput(), batchImages)
            .feed(labels.asOutput(), batchLabels)
    }

    companion object {
        fun <T : Number> of(vararg layers: Layer<T>): TFModel<T> {
            return Sequential()
        }
    }
}