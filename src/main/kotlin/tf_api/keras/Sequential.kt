package tf_api.keras

import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import examples.exportimport.ReluGraphics
import examples.exportimport.ReluGraphics2
import mu.KotlinLogging
import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import tf_api.KGraph
import tf_api.TrainableTFModel
import tf_api.TrainingHistory
import tf_api.keras.dataset.ImageBatch
import tf_api.keras.dataset.ImageDataset
import tf_api.keras.layers.Dense
import tf_api.keras.layers.Input
import tf_api.keras.layers.Layer
import tf_api.keras.loss.LossFunctions
import tf_api.keras.metric.Metrics
import tf_api.keras.optimizers.Optimizer
import tf_api.keras.optimizers.SGD
import tf_api.keras.shape.TensorShape
import tf_api.keras.shape.tail
import java.io.BufferedWriter
import java.io.File
import javax.swing.JFrame

private const val TRAINING_LOSS = "training_loss"

private const val OUTPUT_NAME = "output"

private val logger = KotlinLogging.logger {}

/**
 * Sequential groups a linear stack of layers into a TFModel.
 * Also, it provides training and inference features on this model.
 *
 * @param T the type of data elements in Tensors.
 * @property [input] the input layer with initial shapes.
 * @property [layers] the layers to describe the model design.
 * @constructor Creates a Sequential group with [input] and [layers].
 */

class Sequential<T : Number>(input: Input<T>, vararg layers: Layer<T>) : TrainableTFModel<T>() {
    /** Input layer. */
    private val firstLayer: Input<T> = input

    /** The bunch of layers. */
    private val layers: List<Layer<T>> = listOf(*layers)

    /** A list of variables to train. */
    private var trainableVars: MutableList<Variable<T>> = mutableListOf()

    /** A list of initializer to initialize the trainableVariables. */
    private var initializers: MutableList<Operand<T>> = mutableListOf()

    /** Optimizer. Approach how aggressively to update the weights. */
    private var optimizer: Optimizer<T> = SGD(0.2f)

    /** Loss function. */
    private var loss: LossFunctions = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS

    /** List of metrics for evaluation phase. */
    private var metrics: List<Metrics> = listOf(Metrics.ACCURACY)

    /** TensorFlow operand for prediction phase. */
    private lateinit var yPred: Operand<T>

    /** TensorFlow operand for X data. */
    private lateinit var xOp: Operand<T>

    /** TensorFlow operand for Y data. */
    private lateinit var yOp: Operand<T>

    /** Amount of classes for classification tasks. -1 is a default value for regression tasks. */
    private var amountOfClasses: Long = -1

    private var mu.KLogger.level
        get() = (logger.underlyingLogger as Logger).level
        set(value) {
            (underlyingLogger as Logger).level = value
        }

    init {
        logger.level = Level.DEBUG

        kGraph = KGraph(Graph().toGraphDef())
        tf = Ops.create(kGraph.tfGraph)
        session = Session(kGraph.tfGraph)

        // TODO: think about different logic for different architectures and regression and unsupervised tasks
        if (layers.last() is Dense) {
            amountOfClasses = (layers.last() as Dense).outputSize.toLong()
        }
    }

    companion object {
        /**
         * Creates the [Sequential] model.
         *
         * @param [T] The type of data elements in Tensors.
         * @property [input] The input layer with initial shapes.
         * @property [layers] The layers to describe the model design.
         * @return the [Sequential] model.
         */
        fun <T : Number> of(input: Input<T>, vararg layers: Layer<T>): TrainableTFModel<T> {
            return Sequential(input, *layers)
        }
    }

    /**
     * Configures the model for training.
     *
     * @param [optimizer] Optimizer instance.
     * @param [loss] Loss function.
     * @param [metric] Metric to evaluate during training.
     */
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

            logger.debug { it.toString() + " " + TensorShape(inputShape).dims().contentToString() }

            inputShape = it.computeOutputShape(inputShape)

            logger.debug { it.toString() + " " + TensorShape(inputShape).dims().contentToString() }
        }
    }

    /**
     * Trains the model for a fixed number of [epochs] (iterations on a dataset).
     *
     * @param [dataset] The train dataset that combines input data (X) and target data (Y).
     * @param [epochs] Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
     * @param [batchSize] Number of samples per gradient update.
     * @param [verbose] Verbosity mode. False = silent, True = one line per batch and epoch.
     *
     * @return A [TrainingHistory] object. Its History.history attribute is a record of training loss values and metrics values per each batch and epoch.
     */
    override fun fit(
        dataset: ImageDataset,
        epochs: Int,
        batchSize: Int,
        verbose: Boolean
    ): TrainingHistory {
        val trainingHistory = TrainingHistory()

        File("logs.txt").bufferedWriter().use { file ->

            this.isDebugMode = verbose
            if (!isDebugMode) {
                logger.level = Level.INFO
            }

            xOp = firstLayer.input
            yOp = tf.placeholder(getDType()) as Operand<T>

            yPred = transformInputWithNNModel(xOp)

            val (xBatchShape, yBatchShape) = calculateXYShapes(batchSize)

            val lossOp = LossFunctions.convert<T>(loss).apply(tf, yPred, yOp, getDType())

            val prediction = tf.withName(OUTPUT_NAME).nn.softmax(yPred)

            val metricOp = Metrics.convert<T>(Metrics.ACCURACY).apply(tf, prediction, yOp, getDType())

            file.write("Initialization")
            file.newLine()
            logger.debug { "Initialization of TensorFlow Graph variables" }

            initializeGraphVariables()

            for (i in 1..epochs) {

                val targets = optimizer.prepareTargets(tf, lossOp, trainableVars, i)

                if (verbose) {
                    debugSequentialTraining(file, i)
                }

                val batchIter: ImageDataset.ImageBatchIterator = dataset.batchIterator(
                    batchSize
                )

                var batchCounter = 0

                while (batchIter.hasNext()) {
                    val batch: ImageBatch = batchIter.next()

                    Tensor.create(
                        xBatchShape,
                        batch.images()
                    ).use { batchImages ->
                        Tensor.create(yBatchShape, batch.labels()).use { batchLabels ->
                            val (lossValue, metricValue) = trainOnEpoch(targets, batchImages, batchLabels, metricOp)

                            trainingHistory.append(i, batchCounter, lossValue.toDouble(), metricValue.toDouble())

                            logger.debug { "epochs: $i lossValue: $lossValue metricValue: $metricValue" }
                        }
                    }
                    batchCounter++
                }
            }
        }
        return trainingHistory
    }

    private fun initializeGraphVariables() {
        val runner = session.runner()

        initializers.forEach {
            runner.addTarget(it)
        }

        runner.run()
    }

    private fun trainOnEpoch(
        targets: List<Operand<T>>,
        batchImages: Tensor<Float>,
        batchLabels: Tensor<Float>,
        metricOp: Operand<T>
    ): Pair<Float, Float> {
        val runner = session.runner()

        targets.forEach {
            runner.addTarget(it)
        }

        runner
            .feed(xOp.asOutput(), batchImages)
            .feed(yOp.asOutput(), batchLabels)

        runner
            .fetch(TRAINING_LOSS)
            .fetch(metricOp)

        val tensorList = runner.run()
        val lossValue = tensorList[0].floatValue()
        val metricValue = tensorList[1].floatValue()

        return Pair(lossValue, metricValue)
    }

    /**
     * Returns the loss value & metrics values for the model in test (evaluation) mode.
     *
     * @param [dataset] The train dataset that combines input data (X) and target data (Y).
     * @param [batchSize] Number of samples per batch of computation.
     * @param [metric] Metric to evaluate during test phase.
     *
     * @return Value of calculated metric.
     */
    override fun evaluate(
        dataset: ImageDataset,
        metric: Metrics,
        batchSize: Int
    ): Double {
        val prediction = tf.withName(OUTPUT_NAME).nn.softmax(yPred)

        val metricOp = Metrics.convert<T>(metric).apply(tf, prediction, yOp, getDType())

        val (imageShape, labelShape) = calculateXYShapes(batchSize)

        val batchIter: ImageDataset.ImageBatchIterator = dataset.batchIterator(
            batchSize
        )

        var averageAccuracyAccum = 0.0f
        var amountOfBatches = 0

        while (batchIter.hasNext()) {
            val batch: ImageBatch = batchIter.next()
            amountOfBatches++

            Tensor.create(
                imageShape,
                batch.images()
            ).use { testImages ->
                Tensor.create(labelShape, batch.labels()).use { testLabels ->
                    val metricValue = session.runner()
                        .fetch(metricOp)
                        .feed(xOp.asOutput(), testImages)
                        .feed(yOp.asOutput(), testLabels)
                        .run()[0]

                    logger.info { "test batch acc: ${metricValue.floatValue()}" }

                    averageAccuracyAccum += metricValue.floatValue()
                }
            }
        }

        return (averageAccuracyAccum / amountOfBatches).toDouble()
    }


    /**
     * Generates output predictions for the input samples.
     * Computation is done in batches.
     */
    override fun predict(dataset: ImageDataset, batchSize: Int): IntArray {
        assert(dataset.imagesSize() % batchSize == 0)

        val prediction = tf.withName(OUTPUT_NAME).nn.softmax(yPred)

        val imageShape = calculateXShape(batchSize)

        val predictions = IntArray(dataset.imagesSize()) { Int.MIN_VALUE }

        val batchIter: ImageDataset.ImageBatchIterator = dataset.batchIterator(
            batchSize
        )

        var amountOfBatches = 0

        while (batchIter.hasNext()) {
            val batch: ImageBatch = batchIter.next()
            amountOfBatches++

            Tensor.create(
                imageShape,
                batch.images()
            ).use { testImages ->
                val predictionsTensor = session.runner()
                    .fetch(prediction)
                    .feed(xOp.asOutput(), testImages)
                    .run()[0]

                val dst = Array(imageShape[0].toInt()) { FloatArray(amountOfClasses.toInt()) { 0.0f } }

                predictionsTensor.copyTo(dst)

                val argMaxBatchPrediction = IntArray(imageShape[0].toInt()) { 0 }

                dst.forEachIndexed { index, element ->
                    argMaxBatchPrediction[index] = element.indexOf(element.max()!!)
                }

                argMaxBatchPrediction.copyInto(predictions, batchSize * (amountOfBatches - 1))
            }
        }
        return predictions
    }

    /**
     * Predicts the probability distribution for all classes for the given image.
     */
    override fun predictSoftly(image: FloatArray): FloatArray {
        val predictionData: Array<FloatArray> = arrayOf(image)

        val prediction = tf.withName(OUTPUT_NAME).nn.softmax(yPred)

        val imageShape = calculateXShape(1)

        Tensor.create(
            imageShape,
            ImageDataset.serializeToBuffer(predictionData, 0, 1)
        ).use { testImages ->

            val tensorList = session.runner()
                .fetch(prediction)
                .fetch("Relu")
                .fetch("Relu_1")
                .fetch("Conv2d")
                .fetch("Conv2d_1")
                .feed(xOp.asOutput(), testImages)
                .run()
            val predictionsTensor = tensorList[0]
            val reluTensor = tensorList[1]
            val relu1Tensor = tensorList[2]
            val conv2dTensor = tensorList[3]
            val conv2d1Tensor = tensorList[4]

            //[1, 28, 28, 32]
            //[1, 14, 14, 64]

            val dstData = Array(1) { Array(28) { Array(28) { FloatArray(32) } } }
            reluTensor.copyTo(dstData)

            val frame = JFrame("Visualise the matrix weights on Relu")
            frame.contentPane.add(ReluGraphics(dstData))
            frame.setSize(1500, 1500)
            frame.isVisible = true
            frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
            frame.isResizable = false


            val dstData2 = Array(1) { Array(14) { Array(14) { FloatArray(64) } } }
            relu1Tensor.copyTo(dstData2)

            val frame2 = JFrame("Visualise the matrix weights on Relu_1")
            frame2.contentPane.add(ReluGraphics2(dstData2))
            frame2.setSize(1500, 1500)
            frame2.isVisible = true
            frame2.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
            frame2.isResizable = false

            val dst = Array(1) { FloatArray(amountOfClasses.toInt()) { 0.0f } }

            predictionsTensor.copyTo(dst)

            return dst[0]
        }
    }


    /**
     * Predicts the unknown class for the given image.
     */
    override fun predict(image: FloatArray): Int {
        val softPrediction = predictSoftly(image)
        return softPrediction.indexOf(softPrediction.max()!!)
    }

    private fun calculateXYShapes(batchSize: Int): Pair<LongArray, LongArray> {
        val xBatchShape = calculateXShape(batchSize)

        val yBatchShape = longArrayOf(
            batchSize.toLong(),
            amountOfClasses
        )
        return Pair(xBatchShape, yBatchShape)
    }

    private fun transformInputWithNNModel(input: Operand<T>): Operand<T> {
        var out: Operand<T> = input
        for (layer in layers) {
            out = layer.transformInput(tf, out)
        }
        return out
    }

    private fun calculateXShape(batchSize: Int): LongArray {
        val imageShape = calculateXShape(batchSize.toLong())
        return imageShape
    }

    private fun calculateXShape(amountOfImages: Long): LongArray {
        val xTensorShape = firstLayer.input.asOutput().shape()

        val imageShape = longArrayOf(
            amountOfImages,
            *tail(xTensorShape)
        )
        return imageShape
    }

    override fun close() {
        session.close()
    }

    fun getGraph(): KGraph {
        return kGraph
    }

    private fun debugSequentialTraining(file: BufferedWriter, i: Int) {
        file.write("Epoch # $i started")
        file.newLine()

        val modelWeightsExtractorRunner = session.runner()

        trainableVars.forEach {
            modelWeightsExtractorRunner.fetch(it)
        }

        val modelWeights = modelWeightsExtractorRunner.run()

        for (modelWeight in modelWeights.withIndex()) {
            val variableName = trainableVars[modelWeight.index].asOutput().op().name()

            val tensorForCopying = modelWeight.value

            when (modelWeight.value.shape().size) {
                1 -> {
                    val dst = FloatArray(modelWeight.value.shape()[0].toInt()) { 0.0f }
                    tensorForCopying.copyTo(dst)
                    file.write("Variable $variableName")
                    file.newLine()
                    file.write(dst.contentToString())
                    file.newLine()
                }
                2 -> {
                    val dst =
                        Array(modelWeight.value.shape()[0].toInt()) { FloatArray(modelWeight.value.shape()[1].toInt()) }
                    tensorForCopying.copyTo(dst)

                    file.write("Variable $variableName")
                    file.newLine()
                    file.write(dst.contentDeepToString())
                    file.newLine()
                }
                3 -> {
                    val dst = Array(modelWeight.value.shape()[0].toInt()) {
                        Array(modelWeight.value.shape()[1].toInt()) {
                            FloatArray(modelWeight.value.shape()[2].toInt())
                        }
                    }
                    tensorForCopying.copyTo(dst)
                    file.write("Variable $variableName")
                    file.newLine()
                    file.write(dst.contentDeepToString())
                    file.newLine()

                }
                4 -> {
                    val dst = Array(modelWeight.value.shape()[0].toInt()) {
                        Array(modelWeight.value.shape()[1].toInt()) {
                            Array(modelWeight.value.shape()[2].toInt()) {
                                FloatArray(modelWeight.value.shape()[3].toInt())
                            }
                        }
                    }
                    tensorForCopying.copyTo(dst)
                    file.write("Variable $variableName")
                    file.newLine()
                    file.write(dst.contentDeepToString())
                    file.newLine()

                    /*if(dst.size == 5) {
                        val frame = JFrame("Visualise the matrix weights on $i epochs")
                        frame.contentPane.add(JFrameGraphics(dst))
                        frame.setSize(1000, 1000)
                        frame.isVisible = true
                        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
                        frame.isResizable = false
                    }*/
                }
            }
        }
        file.flush()
    }

    override fun save(pathToModelDirectory: String) {
        val directory = File(pathToModelDirectory);
        if (!directory.exists()) {
            directory.mkdir()
        }
        File("$pathToModelDirectory/graph.pb").writeBytes(kGraph.tfGraph.toGraphDef())

        val modelWeightsExtractorRunner = session.runner()

        trainableVars.forEach {
            modelWeightsExtractorRunner.fetch(it)
        }

        val modelWeights = modelWeightsExtractorRunner.run()

        File("$pathToModelDirectory/variableNames.txt").bufferedWriter().use { variableNamesFile ->


            for (modelWeight in modelWeights.withIndex()) {
                val variableName = trainableVars[modelWeight.index].asOutput().op().name()
                variableNamesFile.write(variableName)
                variableNamesFile.newLine()


                File("$pathToModelDirectory/$variableName.txt").bufferedWriter().use { file ->
                    val tensorForCopying = modelWeight.value

                    when (modelWeight.value.shape().size) {
                        1 -> {
                            val dst = FloatArray(modelWeight.value.shape()[0].toInt()) { 0.0f }
                            tensorForCopying.copyTo(dst)
                            // file.write("Variable $variableName")
                            // file.newLine()
                            file.write(dst.contentToString())
                            //  file.newLine()
                        }
                        2 -> {
                            val dst =
                                Array(modelWeight.value.shape()[0].toInt()) { FloatArray(modelWeight.value.shape()[1].toInt()) }
                            tensorForCopying.copyTo(dst)

                            // file.write("Variable $variableName")
                            // file.newLine()
                            file.write(dst.contentDeepToString())
                            //file.newLine()
                        }
                        3 -> {
                            val dst = Array(modelWeight.value.shape()[0].toInt()) {
                                Array(modelWeight.value.shape()[1].toInt()) {
                                    FloatArray(modelWeight.value.shape()[2].toInt())
                                }
                            }
                            tensorForCopying.copyTo(dst)
                            // file.write("Variable $variableName")
                            // file.newLine()
                            file.write(dst.contentDeepToString())
                            // file.newLine()

                        }
                        4 -> {
                            val dst = Array(modelWeight.value.shape()[0].toInt()) {
                                Array(modelWeight.value.shape()[1].toInt()) {
                                    Array(modelWeight.value.shape()[2].toInt()) {
                                        FloatArray(modelWeight.value.shape()[3].toInt())
                                    }
                                }
                            }
                            tensorForCopying.copyTo(dst)
                            // file.write("Variable $variableName")
                            file.newLine()
                            // file.write(dst.contentDeepToString())
                            // file.newLine()

                            /*if(dst.size == 5) {
                                val frame = JFrame("Visualise the matrix weights on $i epochs")
                                frame.contentPane.add(JFrameGraphics(dst))
                                frame.setSize(1000, 1000)
                                frame.isVisible = true
                                frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
                                frame.isResizable = false
                            }*/
                        }
                    }

                    file.flush()
                }
                variableNamesFile.flush()
            }

        }
    }
}