package api.keras

import api.*
import api.keras.activations.Activations
import api.keras.dataset.ImageBatch
import api.keras.dataset.ImageDataset
import api.keras.exceptions.RepeatableLayerNameException
import api.keras.layers.Dense
import api.keras.layers.Input
import api.keras.layers.Layer
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Optimizer
import api.keras.shape.TensorShape
import api.keras.shape.tail
import api.tensor.convertTensorToFlattenFloatArray
import api.tensor.convertTensorToMultiDimArray
import ch.qos.logback.classic.Level
import mu.KotlinLogging
import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.nn.Softmax
import java.io.File

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
    private lateinit var lossOp: Operand<T>

    /** Input layer. */
    val firstLayer: Input<T> = input

    /** The bunch of layers. */
    val layers: List<Layer<T>> = listOf(*layers)

    /** The bunch of layers. */
    private var layersByName: Map<String, Layer<T>> = mapOf()

    private var isModelCompiled: Boolean = false

    private val logger = KotlinLogging.logger {}

    init {
        for (layer in layers) {
            if (layersByName.containsKey(layer.name)) {
                throw RepeatableLayerNameException(layer.name)
            } else {
                layersByName = layersByName + (layer.name to layer)
            }
        }

        kGraph = KGraph(Graph().toGraphDef())
        tf = Ops.create(kGraph.tfGraph)
        session = Session(kGraph.tfGraph)
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
        fun <T : Number> of(input: Input<T>, vararg layers: Layer<T>): Sequential<T> {
            preProcessLayerNames(layers)
            val seqModel = Sequential(input, *layers)
            postProcessLayerNames(layers, seqModel)
            return seqModel
        }

        /**
         * Creates the [Sequential] model.
         *
         * @param [T] The type of data elements in Tensors.
         * @property [input] The input layer with initial shapes.
         * @property [layers] The layers to describe the model design.
         * @return the [Sequential] model.
         */
        fun <T : Number> of(input: Input<T>, layers: List<Layer<T>>): Sequential<T> {
            preProcessLayerNames(layers.toTypedArray())
            val seqModel = Sequential(input, *layers.toTypedArray())
            postProcessLayerNames(layers.toTypedArray(), seqModel)
            return seqModel
        }

        private fun <T : Number> preProcessLayerNames(layers: Array<out Layer<T>>) {
            var cnt = 1
            for (layer in layers) {
                if (layer.name.isEmpty()) {
                    layer.name = defaultLayerName(cnt)
                    cnt++
                }
            }
        }

        private fun <T : Number> postProcessLayerNames(
            layers: Array<out Layer<T>>,
            seqModel: Sequential<T>
        ) {
            for (layer in layers) {
                layer.parentModel = seqModel
            }
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
        if (isModelCompiled) logger.info { "Model was recompiled." }

        validateModelArchitecture()
        amountOfClasses = (layers.last() as Dense).outputSize.toLong()

        this.loss = loss
        this.metric = metric
        this.metrics = listOf(metric) // handle multiple metrics
        this.optimizer = optimizer

        firstLayer.defineVariables(tf)
        var inputShape: Shape = firstLayer.computeOutputShape()

        layers.forEach {
            it.defineVariables(tf, kGraph, inputShape)

            inputShape = it.computeOutputShape(inputShape)
            val dims = TensorShape(inputShape).dims()
            it.outputShape = dims

            logger.debug { it.toString() + " " + dims.contentToString() }
        }

        xOp = firstLayer.input
        yOp = tf.placeholder(getDType()) as Operand<T>

        yPred = transformInputWithNNModel(xOp)
        lossOp = LossFunctions.convert<T>(loss).apply(tf, yPred, yOp, getDType())

        isModelCompiled = true
    }

    private fun validateModelArchitecture(): Unit {
        require(layers.last() is Dense) { "DL architectures are not finished with Dense layer are not supported yet!" }
        require(!layers.last().hasActivation()) { "Last layer must have an activation function." }
        require((layers.last() as Dense).activation != Activations.Sigmoid) { "The last dense layer should have Sigmoid activation, alternative activations are not supported yet!" }
    }

    override fun fit(
        trainingDataset: ImageDataset,
        validationDataset: ImageDataset,
        epochs: Int,
        trainBatchSize: Int,
        validationBatchSize: Int,
        verbose: Boolean,
        isWeightsInitRequired: Boolean
    ): TrainingHistory {
        return internalFit(
            verbose,
            trainBatchSize,
            epochs,
            trainingDataset,
            true,
            validationDataset,
            validationBatchSize,
            isWeightsInitRequired
        )
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
        verbose: Boolean,
        isWeightsInitRequired: Boolean
    ): TrainingHistory {
        return internalFit(
            verbose,
            batchSize,
            epochs,
            dataset,
            false,
            null,
            null,
            isWeightsInitRequired
        )
    }

    private fun internalFit(
        verbose: Boolean,
        trainBatchSize: Int,
        epochs: Int,
        trainingDataset: ImageDataset,
        validationIsEnabled: Boolean,
        validationDataset: ImageDataset?,
        validationBatchSize: Int?,
        isWeightsInitRequired: Boolean = true
    ): TrainingHistory {
        check(isModelCompiled) { "The model is not compile yet. Call 'compile' method to compile the model." }

        if (isWeightsInitRequired) {
            logger.debug { "Initialization of TensorFlow Graph variables" }
            kGraph.initializeGraphVariables(session)
        }

        val trainingHistory = TrainingHistory()

        this.isDebugMode = verbose
        if (!isDebugMode) {
            logger.level = Level.INFO
        }

        val (xBatchShape, yBatchShape) = calculateXYShapes(trainBatchSize)

        val prediction = tf.withName(OUTPUT_NAME).nn.softmax(yPred)

        val metricOp = Metrics.convert<T>(metric).apply(tf, prediction, yOp, getDType())

        val targets = optimizer.prepareTargets(kGraph, tf, lossOp)

        kGraph.initializeOptimizerVariables(session)

        for (i in 1..epochs) {
            val batchIter: ImageDataset.ImageBatchIterator = trainingDataset.batchIterator(
                trainBatchSize
            )

            var batchCounter = 0
            var averageTrainingMetricAccum = 0.0f
            var averageTrainingLossAccum = 0.0f

            while (batchIter.hasNext()) {
                val batch: ImageBatch = batchIter.next()

                Tensor.create(
                    xBatchShape,
                    batch.images()
                ).use { batchImages ->
                    Tensor.create(yBatchShape, batch.labels()).use { batchLabels ->
                        val (lossValue, metricValue) = trainOnBatch(targets, batchImages, batchLabels, metricOp)

                        averageTrainingLossAccum += lossValue
                        averageTrainingMetricAccum += metricValue
                        trainingHistory.append(i, batchCounter, lossValue.toDouble(), metricValue.toDouble())

                        logger.debug { "Batch stat: { lossValue: $lossValue metricValue: $metricValue }" }
                    }
                }
                batchCounter++
            }

            val avgTrainingMetricValue = (averageTrainingMetricAccum / batchCounter)
            val avgLossValue = (averageTrainingLossAccum / batchCounter)

            if (validationIsEnabled) {
                val evaluationResult = evaluate(validationDataset!!, validationBatchSize!!)
                val validationMetricValue = evaluationResult.metrics[metric]
                val validationLossValue = evaluationResult.lossValue

                logger.info { "epochs: $i loss: $avgLossValue metric: $avgTrainingMetricValue val loss: $validationLossValue val metric: $validationMetricValue" }
            } else {
                logger.info { "epochs: $i loss: $avgLossValue metric: $avgTrainingMetricValue" }
            }
        }
        return trainingHistory
    }

    /**
     * Returns the loss value and metric value on train batch.
     *
     */
    private fun trainOnBatch(
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

        try {
            val tensorList = runner.run()
            val lossValue = tensorList[0].floatValue()
            val metricValue = tensorList[1].floatValue()

            return Pair(lossValue, metricValue)
        } catch (e: TensorFlowException) {
            e.printStackTrace()
            throw RuntimeException(e.message)
        }

    }

    override fun evaluate(dataset: ImageDataset, batchSize: Int): EvaluationResult {
        val prediction = tf.withName(OUTPUT_NAME).nn.softmax(yPred)

        val metricOp = Metrics.convert<T>(metric).apply(tf, prediction, yOp, getDType())

        val (imageShape, labelShape) = calculateXYShapes(batchSize)

        val batchIter: ImageDataset.ImageBatchIterator = dataset.batchIterator(
            batchSize
        )

        var averageMetricAccum = 0.0f
        var averageLossAccum = 0.0f
        var amountOfBatches = 0

        while (batchIter.hasNext()) {
            val batch: ImageBatch = batchIter.next()
            amountOfBatches++

            Tensor.create(
                imageShape,
                batch.images()
            ).use { testImages ->
                Tensor.create(labelShape, batch.labels()).use { testLabels ->
                    val lossAndMetrics = session.runner()
                        .fetch(metricOp)
                        .fetch(TRAINING_LOSS)
                        .feed(xOp.asOutput(), testImages)
                        .feed(yOp.asOutput(), testLabels)
                        .run()

                    val metricValue = lossAndMetrics[0]
                    val lossValue = lossAndMetrics[1]

                    averageMetricAccum += metricValue.floatValue()
                    averageLossAccum += lossValue.floatValue()
                }
            }
        }

        val avgMetricValue = (averageMetricAccum / amountOfBatches).toDouble()
        val avgLossValue = (averageLossAccum / amountOfBatches).toDouble()

        return EvaluationResult(avgLossValue, mapOf(metric to avgMetricValue))
    }


    /**
     * Generates output predictions for the input samples.
     * Computation is done in batches.
     */
    override fun predictAll(dataset: ImageDataset, batchSize: Int): IntArray {
        require(dataset.imagesSize() % batchSize == 0) { "The amount of images must be a multiple of batch size." }

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
     * Predicts the unknown class for the given image.
     */
    override fun predict(image: FloatArray): Int {
        val softPrediction = predictSoftly(image)
        return softPrediction.indexOf(softPrediction.max()!!)
    }

    override fun predictAndGetActivations(image: FloatArray): Pair<Int, List<*>> {
        val (softPrediction, activations) = predictSoftlyAndGetActivations(image, true)
        return Pair(softPrediction.indexOf(softPrediction.max()!!), activations)
    }

    override fun predictSoftly(image: FloatArray): FloatArray {
        val (softPrediction, _) = predictSoftlyAndGetActivations(image, false)
        return softPrediction
    }

    /**
     * Predicts the probability distribution for all classes for the given image.
     */
    override fun predictSoftlyAndGetActivations(
        image: FloatArray,
        formActivationData: Boolean
    ): Pair<FloatArray, List<*>> {
        val predictionData: Array<FloatArray> = arrayOf(image)

        val prediction = tf.withName(OUTPUT_NAME).nn.softmax(yPred)

        val imageShape = calculateXShape(1)

        Tensor.create(
            imageShape,
            ImageDataset.serializeToBuffer(predictionData, 0, 1)
        ).use { testImages ->
            val tensors =
                formPredictionAndActivationsTensors(prediction, testImages, formActivationData)

            val predictionsTensor = tensors[0]

            val dst = Array(1) { FloatArray(amountOfClasses.toInt()) { 0.0f } }

            predictionsTensor.copyTo(dst)

            val activations = mutableListOf<Any>()
            if (formActivationData && tensors.size > 1) {
                for (i in 1 until tensors.size) {
                    activations.add(tensors[i].convertTensorToMultiDimArray())
                }
            }
            return Pair(dst[0], activations.toList())
        }
    }

    private fun formPredictionAndActivationsTensors(
        prediction: Softmax<T>,
        testImages: Tensor<Float>,
        visualizationIsEnabled: Boolean
    ): List<Tensor<*>> {
        val runner = session
            .runner()
            .fetch(prediction)
            .feed(xOp.asOutput(), testImages)

        if (visualizationIsEnabled) {
            for (layer in layers) {
                if (layer.hasActivation()) runner.fetch(defaultActivationName(layer))
            }
        }

        return runner.run()
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
        return calculateXShape(batchSize.toLong())
    }

    private fun calculateXShape(amountOfImages: Long): LongArray {
        val xTensorShape = firstLayer.input.asOutput().shape()

        return longArrayOf(
            amountOfImages,
            *tail(xTensorShape)
        )
    }

    override fun close() {
        session.close()
    }

    fun getGraph(): KGraph<T> {
        return kGraph
    }

    override fun save(pathToModelDirectory: String) {
        val directory = File(pathToModelDirectory)
        if (!directory.exists()) {
            directory.mkdir()
        }
        File("$pathToModelDirectory/graph.pb").writeBytes(kGraph.tfGraph.toGraphDef())

        val modelWeightsExtractorRunner = session.runner()

        val variables = kGraph.variables()

        variables.forEach {
            modelWeightsExtractorRunner.fetch(it)
        }

        val modelWeights = modelWeightsExtractorRunner.run()

        File("$pathToModelDirectory/variableNames.txt").bufferedWriter().use { variableNamesFile ->
            for (modelWeight in modelWeights.withIndex()) {
                val variableName = variables[modelWeight.index].asOutput().op().name()
                variableNamesFile.write(variableName)
                variableNamesFile.newLine()

                File("$pathToModelDirectory/$variableName.txt").bufferedWriter().use { file ->
                    val tensorForCopying = modelWeight.value

                    tensorForCopying.use {
                        val reshaped = tensorForCopying.convertTensorToFlattenFloatArray()

                        for (i in 0..reshaped.size - 2) {
                            file.write(reshaped[i].toString() + " ")
                        }

                        file.write(reshaped[reshaped.size - 1].toString())
                        file.flush()
                    }
                }
                variableNamesFile.flush()
            }
        }
    }

    infix fun getLayer(layerName: String): Layer<T> {
        return layersByName[layerName]!!
    }

    fun summary(stringLayerNameTypeSize: Int = 30, stringOutputShapeSize: Int = 26) {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }

        logger.info("=================================================================")
        logger.info("Model: Sequential")
        logger.info("_________________________________________________________________")
        logger.info("Layer (type)                 Output Shape              Param #   ")
        logger.info("=================================================================")

        var totalParams = 0
        for (l in layers) {
            totalParams += l.getParams()
            logger.info(createLayerDescription(l, stringLayerNameTypeSize, stringOutputShapeSize))
            logger.info("_________________________________________________________________")
        }

        logger.info("=================================================================")
        logger.info("Total params: $totalParams")
        logger.info("=================================================================")
    }

    private fun createLayerDescription(
        l: Layer<T>,
        stringLayerNameTypeSize: Int,
        stringOutputShapeSize: Int
    ): String {
        val firstPart = "${l.name}(${l::class.simpleName})"

        val stringBuilder = StringBuilder(firstPart)
        for (i in 1 until stringLayerNameTypeSize - firstPart.length) {
            stringBuilder.append(" ")
        }

        val secondPart = l.outputShape.contentToString()

        stringBuilder.append(secondPart)

        for (i in 1 until stringOutputShapeSize - secondPart.length) {
            stringBuilder.append(" ")
        }

        stringBuilder.append(l.getParams())

        return stringBuilder.toString()
    }
}