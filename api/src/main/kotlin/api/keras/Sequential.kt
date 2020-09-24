package api.keras

import api.core.KGraph
import api.inference.keras.loadKerasModel
import api.inference.keras.saveConfig
import api.keras.callbacks.Callback
import api.keras.exceptions.RepeatableLayerNameException
import api.keras.history.*
import api.keras.layers.Dense
import api.keras.layers.Input
import api.keras.layers.Layer
import api.keras.loss.LossFunctions
import api.keras.metric.EvaluationResult
import api.keras.metric.Metrics
import api.keras.optimizers.Optimizer
import api.keras.shape.TensorShape
import api.keras.shape.tail
import api.keras.util.OUTPUT_NAME
import api.keras.util.TRAINING_LOSS
import api.keras.util.defaultActivationName
import api.tensor.convertTensorToFlattenFloatArray
import api.tensor.convertTensorToMultiDimArray
import ch.qos.logback.classic.Level
import datasets.DataBatch
import datasets.Dataset
import mu.KotlinLogging
import org.tensorflow.*
import org.tensorflow.op.Ops
import java.io.File

/**
 * Sequential model groups a linear stack of layers into a TensorFlow Model.
 * Also, it provides training and inference features on this model.
 *
 * @property [input] the input layer with initial shapes.
 * @property [layers] the layers to describe the model design.
 * @constructor Creates a Sequential group with [input] and [layers].
 */
class Sequential(input: Input, vararg layers: Layer) : TrainableModel() {
    /** Logger for Sequential model. */
    val logger = KotlinLogging.logger {}

    /** Input layer. */
    val firstLayer: Input = input

    /** The bunch of layers. */
    val layers: List<Layer> = listOf(*layers)

    /** Layers indexed by name. */
    private var layersByName: Map<String, Layer> = mapOf()

    /** Main loss operand. */
    private lateinit var lossOp: Operand<Float>

    /** A list of targets to be optimized. */
    private lateinit var targets: List<Operand<Float>>

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
         * @property [input] The input layer with initial shapes.
         * @property [layers] The layers to describe the model design.
         * @return the [Sequential] model.
         */
        fun of(input: Input, vararg layers: Layer): Sequential {
            preProcessLayerNames(layers)
            val seqModel = Sequential(input, *layers)
            postProcessLayerNames(layers, seqModel)
            return seqModel
        }

        /**
         * Creates the [Sequential] model.
         * @property [layers] The layers to describe the model design.
         * NOTE: First layer should be input layer.
         * @return the [Sequential] model.
         */
        fun of(layers: List<Layer>): Sequential {
            require(layers.isNotEmpty()) { "Model should contain layers!" }
            val input = layers[0]
            require(input is Input) { "Model should start from the Input layer" }

            val otherLayers = layers.subList(1, layers.size)
            preProcessLayerNames(otherLayers.toTypedArray())
            val seqModel = Sequential(input, *otherLayers.toTypedArray())
            postProcessLayerNames(otherLayers.toTypedArray(), seqModel)
            return seqModel
        }

        /**
         * Creates the [Sequential] model.
         *
         * @property [input] The input layer with initial shapes.
         * @property [layers] The layers to describe the model design.
         * @return the [Sequential] model.
         */
        fun of(input: Input, layers: List<Layer>): Sequential {
            preProcessLayerNames(layers.toTypedArray())
            val seqModel = Sequential(input, *layers.toTypedArray())
            postProcessLayerNames(layers.toTypedArray(), seqModel)
            return seqModel
        }

        private fun preProcessLayerNames(layers: Array<out Layer>) {
            var cnt = 1
            for (layer in layers) {
                if (layer.name.isEmpty()) {
                    val generatedLayerName = layer::class.simpleName!!.toLowerCase() + "_" + cnt
                    layer.name = generatedLayerName
                    cnt++
                }
            }
        }

        private fun postProcessLayerNames(
            layers: Array<out Layer>,
            seqModel: Sequential
        ) {
            for (layer in layers) {
                layer.parentModel = seqModel
            }
        }

        // TODO: change signature - File with config instead of path to directory
        /**
         * Loads a Sequential model from json file with model configuration.
         *
         * @param [pathToModelDirectory] Directory, containing file 'modelConfig.json'.
         * @return Non-compiled and non-trained Sequential model.
         */
        fun load(pathToModelDirectory: String): Sequential {
            val jsonConfig = File("$pathToModelDirectory/modelConfig.json")
            return loadKerasModel(jsonConfig)
        }
    }

    override fun compile(optimizer: Optimizer, loss: LossFunctions, metric: Metrics, callback: Callback) {
        if (isModelCompiled) logger.info { "Model was recompiled." }

        validateModelArchitecture()
        amountOfClasses = (layers.last() as Dense).outputSize.toLong()

        this.loss = loss
        this.metric = metric
        this.metrics = listOf(metric) // handle multiple metrics
        this.optimizer = optimizer
        this.callback = callback
        this.callback.model = this // TODO: cyclic reference

        firstLayer.defineVariables(tf)
        var inputShape: Shape = firstLayer.computeOutputShape()

        layers.forEach {
            it.defineVariables(tf, kGraph, inputShape)

            inputShape = it.computeOutputShape(inputShape)
            val dims = TensorShape(inputShape).dims()
            it.outputShape = dims

            logger.debug { it.toString() + "; outputShape: " + dims.contentToString() }
        }

        xOp = firstLayer.input
        yOp = tf.placeholder(getDType()) as Operand<Float>

        yPred = transformInputWithNNModel(xOp)
        lossOp = LossFunctions.convert(loss).apply(tf, yPred, yOp)

        targets = optimizer.prepareTargets(kGraph, tf, lossOp)

        isModelCompiled = true
    }

    private fun validateModelArchitecture() {
        require(layers.last() is Dense) { "DL architectures are not finished with Dense layer are not supported yet!" }
        require(layers.last().hasActivation()) { "Last layer must have an activation function." }
//        require((layers.last() as Dense).activation != Activations.Sigmoid) { "The last dense layer should have Linear activation, alternative activations are not supported yet!" }
    }

    override fun fit(
        trainingDataset: Dataset,
        validationDataset: Dataset,
        epochs: Int,
        trainBatchSize: Int,
        validationBatchSize: Int,
        verbose: Boolean,
        isWeightsInitRequired: Boolean,
        isOptimizerInitRequired: Boolean
    ): TrainingHistory {
        return internalFit(
            verbose,
            trainBatchSize,
            epochs,
            trainingDataset,
            true,
            validationDataset,
            validationBatchSize,
            isWeightsInitRequired,
            isOptimizerInitRequired
        )
    }

    override fun fit(
        dataset: Dataset,
        epochs: Int,
        batchSize: Int,
        verbose: Boolean,
        isWeightsInitRequired: Boolean,
        isOptimizerInitRequired: Boolean
    ): TrainingHistory {
        return internalFit(
            verbose,
            batchSize,
            epochs,
            dataset,
            false,
            null,
            null,
            isWeightsInitRequired,
            isOptimizerInitRequired
        )
    }

    /**
     * Initializes kGraph variables.
     *
     * NOTE: Model becomes initialized after this method call. (Flag [isModelInitialized] = true)
     */
    fun init() {
        require(!isModelInitialized) { "Model is initialized already!" }
        logger.debug { "Initialization of TensorFlow Graph variables" }
        kGraph.initializeGraphVariables(session)
        isModelInitialized = true
    }

    private fun internalFit(
        verbose: Boolean,
        trainBatchSize: Int,
        epochs: Int,
        trainingDataset: Dataset,
        validationIsEnabled: Boolean,
        validationDataset: Dataset?,
        validationBatchSize: Int?,
        isWeightsInitRequired: Boolean = true,
        isOptimizerInitRequired: Boolean = true
    ): TrainingHistory {
        check(isModelCompiled) { "The model is not compile yet. Call 'compile' method to compile the model." }

        if (isWeightsInitRequired) {
            logger.debug { "Initialization of TensorFlow Graph variables" }
            kGraph.initializeGraphVariables(session)
            isModelInitialized = true
        }

        val trainingHistory = TrainingHistory()

        this.isDebugMode = verbose
        if (!isDebugMode) {
            logger.level = Level.INFO
        }

        val prediction = when (loss) {
            LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS -> tf.withName(OUTPUT_NAME).nn.softmax(yPred)
            else -> tf.withName(OUTPUT_NAME).identity(yPred)
        }

        val metricOp = Metrics.convert(metric).apply(tf, prediction, yOp)

        if (isOptimizerInitRequired) kGraph.initializeOptimizerVariables(session)

        callback.onTrainBegin()

        for (i in 1..epochs) {
            if (!stopTraining) {
                callback.onEpochBegin(i, trainingHistory)
                val batchIter: Dataset.BatchIterator = trainingDataset.batchIterator(
                    trainBatchSize
                )

                var batchCounter = 0
                var averageTrainingMetricAccum = 0.0f
                var averageTrainingLossAccum = 0.0f

                while (batchIter.hasNext() && !stopTraining) { // TODO: analyze before release <==== could be stopped via callback
                    callback.onTrainBatchBegin(batchCounter, trainBatchSize, trainingHistory)
                    val batch: DataBatch = batchIter.next()

                    val (xBatchShape, yBatchShape) = calculateXYShapes(batch)

                    Tensor.create(
                        xBatchShape,
                        batch.x
                    ).use { batchImages ->
                        Tensor.create(yBatchShape, batch.y).use { batchLabels ->
                            val (lossValue, metricValue) = trainOnBatch(targets, batchImages, batchLabels, metricOp)

                            averageTrainingLossAccum += lossValue
                            averageTrainingMetricAccum += metricValue
                            val batchTrainingEvent =
                                BatchTrainingEvent(i, batchCounter, lossValue.toDouble(), metricValue.toDouble())
                            trainingHistory.appendBatch(batchTrainingEvent)

                            logger.debug { "Batch stat: { lossValue: $lossValue metricValue: $metricValue }" }

                            callback.onTrainBatchEnd(batchCounter, trainBatchSize, batchTrainingEvent, trainingHistory)
                        }
                    }
                    batchCounter++
                }

                val avgTrainingMetricValue = (averageTrainingMetricAccum / batchCounter)
                val avgLossValue = (averageTrainingLossAccum / batchCounter)

                val epochTrainingEvent = EpochTrainingEvent(
                    i,
                    avgLossValue.toDouble(), avgTrainingMetricValue.toDouble(), Double.NaN, Double.NaN
                )

                if (validationIsEnabled) {
                    val evaluationResult = evaluate(validationDataset!!, validationBatchSize!!)
                    val validationMetricValue = evaluationResult.metrics[metric]
                    val validationLossValue = evaluationResult.lossValue
                    epochTrainingEvent.valLossValue = validationLossValue
                    epochTrainingEvent.valMetricValue = validationMetricValue!!
                    logger.info { "epochs: $i loss: $avgLossValue metric: $avgTrainingMetricValue val loss: $validationLossValue val metric: $validationMetricValue" }
                } else {
                    logger.info { "epochs: $i loss: $avgLossValue metric: $avgTrainingMetricValue" }

                }
                trainingHistory.appendEpoch(epochTrainingEvent)
                callback.onEpochEnd(i, epochTrainingEvent, trainingHistory)
            }
        }
        callback.onTrainEnd(trainingHistory)
        return trainingHistory
    }

    private fun batchValidation(
        batch: DataBatch,
        xBatchShape: LongArray,
        yBatchShape: LongArray
    ) {
        check(TensorShape(xBatchShape).numElements().toInt() == batch.x.capacity())
        {
            "The calculated [from the Sequential model] data batch shape ${xBatchShape.contentToString()} doesn't match actual data buffer size ${
                batch.x.capacity()
            }. Please, check input data."
        }
        check(TensorShape(yBatchShape).numElements().toInt() == batch.y.capacity())
        {
            "The calculated [from the Sequential model] label batch shape ${yBatchShape.contentToString()} doesn't match actual data buffer size ${
                batch.y.capacity()
            }. " +
                    "\nPlease, check the input label data or correct amount of classes [amount of neurons] in last Dense layer, if you have a classification problem." +
                    "\nHighly likely, you have different amount of classes presented in data and described in model as desired output."
        }
    }

    /**
     * Returns the loss value and metric value on train batch.
     */
    private fun trainOnBatch(
        targets: List<Operand<Float>>,
        batchImages: Tensor<Float>,
        batchLabels: Tensor<Float>,
        metricOp: Operand<Float>
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

    override fun evaluate(dataset: Dataset, batchSize: Int): EvaluationResult {
        val evaluationHistory = History()

        callback.onTestBegin()

        val prediction = when (loss) {
            LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS -> tf.withName(OUTPUT_NAME).nn.softmax(yPred)
            else -> tf.withName(OUTPUT_NAME).identity(yPred)
        }

        val metricOp = Metrics.convert(metric).apply(tf, prediction, yOp)

        val batchIter: Dataset.BatchIterator = dataset.batchIterator(
            batchSize
        )

        var averageMetricAccum = 0.0f
        var averageLossAccum = 0.0f
        var batchCounter = 0

        while (batchIter.hasNext()) {
            callback.onTestBatchBegin(batchCounter, batchSize, evaluationHistory)
            val batch: DataBatch = batchIter.next()
            val (imageShape, labelShape) = calculateXYShapes(batch)

            Tensor.create(
                imageShape,
                batch.x
            ).use { testImages ->
                Tensor.create(labelShape, batch.y).use { testLabels ->
                    val lossAndMetrics = session.runner()
                        .fetch(metricOp)
                        .fetch(TRAINING_LOSS)
                        .feed(xOp.asOutput(), testImages)
                        .feed(yOp.asOutput(), testLabels)
                        .run()

                    val metricValue = lossAndMetrics[0].floatValue()
                    val lossValue = lossAndMetrics[1].floatValue()

                    averageMetricAccum += metricValue
                    averageLossAccum += lossValue

                    val batchEvent = BatchEvent(batchCounter, lossValue.toDouble(), metricValue.toDouble())
                    evaluationHistory.appendBatch(batchEvent)

                    callback.onTestBatchEnd(batchCounter, batchSize, batchEvent, evaluationHistory)
                }
            }

            batchCounter++
        }

        val avgMetricValue = (averageMetricAccum / batchCounter).toDouble()
        val avgLossValue = (averageLossAccum / batchCounter).toDouble()

        callback.onTestEnd(evaluationHistory)
        return EvaluationResult(avgLossValue, mapOf(metric to avgMetricValue))
    }


    override fun predictAll(dataset: Dataset, batchSize: Int): IntArray {
        require(dataset.xSize() % batchSize == 0) { "The amount of images must be a multiple of batch size." }
        callback.onPredictBegin()

        val prediction = when (loss) {
            LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS -> tf.withName(OUTPUT_NAME).nn.softmax(yPred)
            else -> tf.withName(OUTPUT_NAME).identity(yPred)
        }

        val imageShape = calculateXShape(batchSize)

        val predictions = IntArray(dataset.xSize()) { Int.MIN_VALUE }

        val batchIter: Dataset.BatchIterator = dataset.batchIterator(
            batchSize
        )

        var batchCounter = 0

        while (batchIter.hasNext()) {
            callback.onPredictBatchBegin(batchCounter, batchSize)

            val batch: DataBatch = batchIter.next()

            Tensor.create(
                imageShape,
                batch.x
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

                callback.onPredictBatchEnd(batchCounter, batchSize)
                batchCounter++
                argMaxBatchPrediction.copyInto(predictions, batchSize * (batchCounter - 1))
            }
        }
        callback.onPredictEnd()
        return predictions
    }

    /**
     * Predicts the unknown class for the given image.
     */
    override fun predict(inputData: FloatArray): Int {
        val softPrediction = predictSoftly(inputData)
        return softPrediction.indexOf(softPrediction.max()!!)
    }

    override fun predict(inputData: FloatArray, predictionTensorName: String): Int {
        val softPrediction = predictSoftly(inputData, predictionTensorName)
        return softPrediction.indexOf(softPrediction.max()!!)
    }

    override fun predictAndGetActivations(inputData: FloatArray, predictionTensorName: String): Pair<Int, List<*>> {
        val (softPrediction, activations) = predictSoftlyAndGetActivations(inputData, true, predictionTensorName)
        return Pair(softPrediction.indexOf(softPrediction.max()!!), activations)
    }

    override fun predictSoftly(inputData: FloatArray, predictionTensorName: String): FloatArray {
        val (softPrediction, _) = predictSoftlyAndGetActivations(inputData, false, predictionTensorName)
        return softPrediction
    }

    /**
     * Predicts the probability distribution for all classes for the given image.
     */
    override fun predictSoftlyAndGetActivations(
        inputData: FloatArray,
        visualizationIsEnabled: Boolean,
        predictionTensorName: String
    ): Pair<FloatArray, List<*>> {
        val predictionData: Array<FloatArray> = arrayOf(inputData)

        val prediction = when (loss) {
            LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS -> tf.withName(OUTPUT_NAME).nn.softmax(yPred)
            else -> tf.withName(OUTPUT_NAME).identity(yPred)
        }

        val imageShape = calculateXShape(1)

        Tensor.create(
            imageShape,
            Dataset.serializeToBuffer(predictionData, 0, 1)
        ).use { testImages ->
            val tensors =
                formPredictionAndActivationsTensors(prediction, testImages, visualizationIsEnabled)

            val predictionsTensor = tensors[0]

            val dst = Array(1) { FloatArray(amountOfClasses.toInt()) { 0.0f } }

            predictionsTensor.copyTo(dst)

            val activations = mutableListOf<Any>()
            if (visualizationIsEnabled && tensors.size > 1) {
                for (i in 1 until tensors.size) {
                    activations.add(tensors[i].convertTensorToMultiDimArray())
                }
            }
            return Pair(dst[0], activations.toList())
        }
    }

    private fun formPredictionAndActivationsTensors(
        prediction: Operand<Float>,
        testImages: Tensor<Float>,
        visualizationIsEnabled: Boolean
    ): List<Tensor<*>> {
        val runner = session
            .runner()
            .fetch(prediction)
            .feed(xOp.asOutput(), testImages)

        if (visualizationIsEnabled) {
            for (layer in layers) {
                if (layer.hasActivation() && layer != layers.last()) runner.fetch(defaultActivationName(layer))
            }
        }

        return runner.run()
    }

    private fun calculateXYShapes(batch: DataBatch): Pair<LongArray, LongArray> {
        val batchSize = batch.size

        val xBatchShape = calculateXShape(batchSize)

        val yBatchShape = longArrayOf(
            batchSize.toLong(),
            amountOfClasses
        )

        batchValidation(batch, xBatchShape, yBatchShape)

        return Pair(xBatchShape, yBatchShape)
    }

    private fun transformInputWithNNModel(input: Operand<Float>): Operand<Float> {
        var out: Operand<Float> = input
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

    /**
     * Returns KGraph.
     *
     * NOTE: Be careful, this is a direct access to the model graph, not a copy.
     */
    fun kGraph(): KGraph {
        return kGraph
    }

    override fun save(
        pathToModelDirectory: String,
        modelFormat: ModelFormat,
        saveOptimizerState: Boolean,
        modelWritingMode: ModelWritingMode
    ) {
        val directory = File(pathToModelDirectory)

        when (modelWritingMode) {
            ModelWritingMode.FAIL_IF_EXISTS -> {
                check(!directory.exists()) { "The directory exists on path $pathToModelDirectory, please be careful it could contain valuable model! Change this mode to OVERRIDE if you want to override this directory." }
                directory.mkdir()
            }
            ModelWritingMode.OVERRIDE -> {
                if (directory.exists()) {
                    directory.deleteRecursively()
                }
                directory.mkdir()
            }
            ModelWritingMode.APPEND -> {
                if (!directory.exists()) {
                    directory.mkdir()
                }
            }
        }

        when (modelFormat) {
            ModelFormat.TF_GRAPH_CUSTOM_VARIABLES -> saveInSimpleFormat(pathToModelDirectory, saveOptimizerState)
            ModelFormat.TF_GRAPH -> saveInSavedModelFormat(pathToModelDirectory)
            ModelFormat.KERAS_CONFIG_CUSTOM_VARIABLES -> saveInKerasFormat(pathToModelDirectory, saveOptimizerState)
        }
    }

    private fun saveInKerasFormat(pathToModelDirectory: String, saveOptimizerState: Boolean) {
        saveModel(pathToModelDirectory)
        saveVariables(pathToModelDirectory, saveOptimizerState)
    }

    private fun saveModel(pathToModelDirectory: String) {
        val jsonConfig = File("$pathToModelDirectory/modelConfig.json")
        this.saveConfig(jsonConfig)
    }

    private fun saveInSavedModelFormat(pathToModelDirectory: String) {
        saveGraphDef(pathToModelDirectory)
    }

    private fun saveInSimpleFormat(pathToModelDirectory: String, saveOptimizerState: Boolean) {
        saveGraphDef(pathToModelDirectory)
        saveVariables(pathToModelDirectory, saveOptimizerState)
    }

    private fun saveGraphDef(pathToModelDirectory: String) {
        File("$pathToModelDirectory/graph.pb").writeBytes(kGraph.tfGraph.toGraphDef())
    }

    private fun saveVariables(pathToModelDirectory: String, saveOptimizerState: Boolean) {
        val modelWeightsExtractorRunner = session.runner()

        var variables = kGraph.layerVariables()

        if (saveOptimizerState) {
            variables = variables + kGraph.optimizerVariables()
        }

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

    override fun loadVariablesFromTxtFiles(pathToModelDirectory: String, loadOptimizerState: Boolean) {
        // Load variables names
        val variableNames = File("$pathToModelDirectory/variableNames.txt").readLines()
        if (variableNames.isNotEmpty()) {
            for (variableName in variableNames) {
                if (!loadOptimizerState && variableName.startsWith("optimizer")) // skip loading optimizers' variables
                    continue
                else if (loadOptimizerState && isOptimizerNameAndRelatedToFrozenLayer(variableName)) // skip loading optimizers' variables for frozen layers
                    continue
                else loadVariable(variableName, pathToModelDirectory)
            }
        }
    }

    private fun isOptimizerNameAndRelatedToFrozenLayer(variableName: String): Boolean {
        return variableName.startsWith("optimizer") && kGraph().frozenLayerVariables()
            .map { it.ref().op().name() } // extract names
            .any { variableName.contains(it) }
    }

    /**
     * Return layer by [layerName].
     *
     * @param [layerName] Should be existing layer name. Throws an error otherwise.
     */
    infix fun getLayer(layerName: String): Layer {
        return layersByName[layerName] ?: error("No such layer $layerName in the model.")
    }

    /**
     * Formats and builds the model description.
     *
     * @return list of layer descriptions.
     */
    fun summary(stringLayerNameTypeSize: Int = 30, stringOutputShapeSize: Int = 26): List<String> {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }

        logger.info("=================================================================")
        logger.info("Model: Sequential")
        logger.info("_________________________________________________________________")
        logger.info("Layer (type)                 Output Shape              Param #   ")
        logger.info("=================================================================")

        var totalTrainableParams = 0
        var totalFrozenParams = 0

        val layerDescriptions = mutableListOf<String>()

        for (l in layers) {
            if (l.isTrainable) totalTrainableParams += l.getParams() else totalFrozenParams += l.getParams()
            val layerDescription = createLayerDescription(l, stringLayerNameTypeSize, stringOutputShapeSize)
            layerDescriptions.add(layerDescription)
            logger.info(layerDescription)
            logger.info("_________________________________________________________________")
        }

        logger.info("=================================================================")
        logger.info("Total trainable params: $totalTrainableParams")
        logger.info("Total frozen params: $totalFrozenParams")
        logger.info("Total params: ${totalTrainableParams + totalFrozenParams}")
        logger.info("=================================================================")

        return layerDescriptions
    }

    private fun createLayerDescription(
        l: Layer,
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

        for (i in 0 until stringOutputShapeSize - secondPart.length) {
            stringBuilder.append(" ")
        }

        stringBuilder.append(l.getParams())

        return stringBuilder.toString()
    }
}