/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import mu.KLogger
import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.exception.RepeatableLayerNameException
import org.jetbrains.kotlinx.dl.api.core.history.*
import org.jetbrains.kotlinx.dl.api.core.layer.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.Input
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.loss.LossFunction
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.loss.SoftmaxCrossEntropyWithLogits
import org.jetbrains.kotlinx.dl.api.core.metric.EvaluationResult
import org.jetbrains.kotlinx.dl.api.core.metric.Metric
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Optimizer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.shape.tail
import org.jetbrains.kotlinx.dl.api.core.util.OUTPUT_NAME
import org.jetbrains.kotlinx.dl.api.core.util.TRAINING_LOSS
import org.jetbrains.kotlinx.dl.api.core.util.defaultActivationName
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToFlattenFloatArray
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToMultiDimArray
import org.jetbrains.kotlinx.dl.api.inference.keras.loadModelLayers
import org.jetbrains.kotlinx.dl.api.inference.keras.saveModelConfiguration
import org.jetbrains.kotlinx.dl.datasets.DataBatch
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import java.io.File
import java.io.FileNotFoundException
import java.nio.FloatBuffer

/**
 * Sequential model groups a linear stack of layers into a TensorFlow Model.
 * Also, it provides training and inference features on this model.
 *
 * @property [inputLayer] the input layer with initial shapes.
 * @property [layers] the layers to describe the model design.
 * @constructor Creates a Sequential group with [inputLayer] and [layers].
 */
public class Sequential(input: Input, vararg layers: Layer) : TrainableModel() {
    /** Logger for Sequential model. */
    public val logger: KLogger = KotlinLogging.logger {}

    /** Input layer. */
    public val inputLayer: Input = input

    /** The bunch of layers. */
    public val layers: List<Layer> = listOf(*layers)

    /** Layers indexed by name. */
    private var layersByName: Map<String, Layer> = mapOf()

    /** TensorFlow operand for prediction phase. */
    private lateinit var yPredOp: Operand<Float>

    /** TensorFlow loss operand. */
    private lateinit var lossOp: Operand<Float>

    /** TensorFlow prediction operand. */
    private lateinit var predictionOp: Operand<Float>

    /** TensorFlow prediction operand. */
    private lateinit var metricOp: Operand<Float>

    /** A list of targets to be optimized. */
    private lateinit var targets: List<Operand<Float>>

    /** TensorFlow operand for X data. */
    private lateinit var xOp: Operand<Float>

    /** TensorFlow operand for Y data. */
    private lateinit var yTrueOp: Operand<Float>

    /** TensorFlow operand for batch size data. */
    private lateinit var numberOfLossesOp: Operand<Float>

    /** TensorFlow operand for batch size data. */
    private lateinit var training: Operand<Float>

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

    public companion object {
        /**
         * Creates the [Sequential] model.
         *
         * @property [input] The input layer with initial shapes.
         * @property [layers] The layers to describe the model design.
         * @return the [Sequential] model.
         */
        @JvmStatic
        public fun of(input: Input, vararg layers: Layer): Sequential {
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
        @JvmStatic
        public fun of(layers: List<Layer>): Sequential {
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
        @JvmStatic
        public fun of(input: Input, layers: List<Layer>): Sequential {
            preProcessLayerNames(layers.toTypedArray())
            val seqModel = Sequential(input, *layers.toTypedArray())
            postProcessLayerNames(layers.toTypedArray(), seqModel)
            return seqModel
        }

        private fun preProcessLayerNames(layers: Array<out Layer>) {
            var cnt = 1
            for (layer in layers) {
                if (layer.name.isEmpty()) {
                    val generatedLayerName = (layer::class.simpleName ?: return).toLowerCase() + "_" + cnt
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

        /**
         * Loads a [Sequential] model from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [Sequential] model.
         * @return Non-compiled and non-trained Sequential model.
         */
        @JvmStatic
        public fun loadModelConfiguration(configuration: File): Sequential {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            return org.jetbrains.kotlinx.dl.api.inference.keras.loadModelConfiguration(configuration)
        }

        /**
         * Loads a [Sequential] model layers from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [Sequential] model.
         * @return Pair of <input layer; list of layers>.
         */
        @JvmStatic
        public fun loadModelLayersFromConfiguration(configuration: File): Pair<Input, MutableList<Layer>> {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            return loadModelLayers(configuration)
        }

        /**
         * Loads a [Sequential] model from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return Non-compiled and non-trained Sequential model.
         */
        @JvmStatic
        public fun loadDefaultModelConfiguration(modelDirectory: File): Sequential {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a 'modelConfig.json' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/modelConfig.json")

            if (!configuration.exists()) throw FileNotFoundException(
                "File 'modelConfig.json' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            return org.jetbrains.kotlinx.dl.api.inference.keras.loadModelConfiguration(configuration)
        }

        /**
         * Loads a [Sequential] model layers from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return Pair of <input layer; list of layers>.
         */
        @JvmStatic
        public fun loadModelLayersFromDefaultConfiguration(modelDirectory: File): Pair<Input, MutableList<Layer>> {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a 'modelConfig.json' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/modelConfig.json")

            if (!configuration.exists()) throw FileNotFoundException(
                "File 'modelConfig.json' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            return loadModelLayers(configuration)
        }
    }

    override fun compile(optimizer: Optimizer, loss: Losses, metric: Metrics, callback: Callback) {
        compile(optimizer, Losses.convert(loss), Metrics.convert(metric), callback)
    }

    override fun compile(optimizer: Optimizer, loss: LossFunction, metric: Metric, callback: Callback) {
        check(!isModelCompiled) { "The model is compiled already. Graph is created. Create new model and compile it." }

        validateModelArchitecture()
        amountOfClasses = if (layers.last() is Dense) (layers.last() as Dense).outputSize.toLong() else 1

        this.loss = loss
        this.metric = metric
        this.metrics = listOf(metric) // handle multiple metrics
        this.optimizer = optimizer
        this.callback = callback
        this.callback.model = this // TODO: cyclic reference

        inputLayer.defineVariables(tf)
        var inputShape: Shape = inputLayer.computeOutputShape()

        layers.forEach {
            it.defineVariables(tf, kGraph, inputShape)

            inputShape = it.computeOutputShape(inputShape)
            val tensorShape = TensorShape(inputShape)
            val dims = tensorShape.dims()

            check(tensorShape.tail().all { elem -> elem > 0 })
            {
                "The last dimensions (except first = -1) of shape of layer ${it.name} contains zero or negative dimension values: ${dims.contentToString()}.\n" +
                        "Analyze your model architecture and layer output shapes carefully to discover a problem."
            }

            it.outputShape = dims

            logger.debug { "$it; outputShape: $tensorShape" }
        }

        xOp = inputLayer.input
        yTrueOp = tf.placeholder(getDType()) as Operand<Float>
        numberOfLossesOp = tf.withName("numberOfLosses").placeholder(
            getDType(),
            Placeholder.shape(Shape.scalar())
        )

        training = tf.withName("training").placeholder(
            getDType(),
            Placeholder.shape(Shape.scalar())
        )



        yPredOp = forward(xOp)
        lossOp = loss.apply(tf, yPredOp, yTrueOp, numberOfLossesOp)
        targets = optimizer.prepareTargets(kGraph, tf, lossOp)

        predictionOp = when (loss) {
            is SoftmaxCrossEntropyWithLogits -> tf.withName(OUTPUT_NAME).nn.softmax(yPredOp)
            else -> tf.withName(OUTPUT_NAME).identity(yPredOp)
        }

        metricOp = metric.apply(tf, predictionOp, yTrueOp, numberOfLossesOp)

        isModelCompiled = true
    }

    override fun compile(optimizer: Optimizer, loss: Losses, metric: Metric, callback: Callback) {
        compile(optimizer, Losses.convert(loss), metric, callback)
    }

    override fun compile(optimizer: Optimizer, loss: LossFunction, metric: Metrics, callback: Callback) {
        compile(optimizer, loss, Metrics.convert(metric), callback)
    }

    private fun validateModelArchitecture() {
        //  require(layers.last() is Dense) { "DL architectures are not finished with Dense layer are not supported yet!" }
        //   require(layers.last().hasActivation()) { "Last layer must have an activation function." }
//        require((layers.last() as Dense).activation != Activations.Sigmoid) { "The last dense layer should have Linear activation, alternative activations are not supported yet!" }
    }

    override fun fit(
        trainingDataset: Dataset,
        validationDataset: Dataset,
        epochs: Int,
        trainBatchSize: Int,
        validationBatchSize: Int
    ): TrainingHistory {
        return internalFit(
            trainBatchSize,
            epochs,
            trainingDataset,
            true,
            validationDataset,
            validationBatchSize
        )
    }

    override fun fit(
        dataset: Dataset,
        epochs: Int,
        batchSize: Int
    ): TrainingHistory {
        return internalFit(
            batchSize,
            epochs,
            dataset,
            false,
            null,
            null
        )
    }

    /**
     * Initializes kGraph variables.
     *
     * NOTE: Model becomes initialized after this method call. (Flags [isModelInitialized] and [isOptimizerVariableInitialized] are set up to true)
     */
    public fun init() {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(!isModelInitialized) { "Model is initialized already!" }
        check(!isOptimizerVariableInitialized) { "Optimizer variables are initialized already!" }

        logger.debug { "Initialization of TensorFlow Graph variables." }
        kGraph.initializeGraphVariables(session)
        isModelInitialized = true
    }

    private fun internalFit(
        trainBatchSize: Int,
        epochs: Int,
        trainingDataset: Dataset,
        validationIsEnabled: Boolean,
        validationDataset: Dataset?,
        validationBatchSize: Int?
    ): TrainingHistory {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }

        if (!isModelInitialized) {
            logger.debug { "Initialization of TensorFlow Graph variables." }
            kGraph.initializeGraphVariables(session)
            isModelInitialized = true
        }

        val trainingHistory = TrainingHistory()

        if (!isOptimizerVariableInitialized) {
            logger.debug { "Initialization of optimizer variables." }
            kGraph.initializeOptimizerVariables(session)
            isOptimizerVariableInitialized = true
        }

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
                    ).use { batchImagesTensor ->
                        Tensor.create(yBatchShape, batch.y).use { batchLabelsTensor ->
                            Tensor.create(TensorShape(yBatchShape).numElements().toFloat())
                                .use { numberOfLossesTensor ->
                                    Tensor.create(1.0f).use { isTraining ->
                                        val (lossValue, metricValue) = trainOnBatch(
                                            targets,
                                            batchImagesTensor,
                                            batchLabelsTensor,
                                            numberOfLossesTensor as Tensor<Float>,
                                            isTraining as Tensor<Float>,
                                            metricOp
                                        )
                                        if (lossValue.isNaN() || lossValue == Float.POSITIVE_INFINITY || lossValue == Float.NEGATIVE_INFINITY) {
                                            logger.debug { "Loss function value is NaN. You could use TerminateOnNaN callback to stop it earlier." }
                                        }

                                        averageTrainingLossAccum += lossValue
                                        averageTrainingMetricAccum += metricValue
                                        val batchTrainingEvent =
                                            BatchTrainingEvent(
                                                i,
                                                batchCounter,
                                                lossValue.toDouble(),
                                                metricValue.toDouble()
                                            )
                                        trainingHistory.appendBatch(batchTrainingEvent)

                                        logger.debug { "Batch stat: { lossValue: $lossValue metricValue: $metricValue }" }

                                        callback.onTrainBatchEnd(
                                            batchCounter,
                                            trainBatchSize,
                                            batchTrainingEvent,
                                            trainingHistory
                                        )
                                    }
                                }
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
                    val validationMetricValue = evaluationResult.metrics[Metrics.convertBack(metric)]
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
        numberOfLosses: Tensor<Float>,
        isTraining: Tensor<Float>,
        metricOp: Operand<Float>
    ): Pair<Float, Float> {
        val runner = session.runner()

        targets.forEach {
            runner.addTarget(it)
        }

        runner
            .feed(xOp.asOutput(), batchImages)
            .feed(yTrueOp.asOutput(), batchLabels)
            .feed(numberOfLossesOp.asOutput(), numberOfLosses)
            .feed(training.asOutput(), isTraining)

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
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method." }

        val evaluationHistory = History()

        callback.onTestBegin()

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
            ).use { testImagesTensor ->
                Tensor.create(labelShape, batch.y).use { testLabelsTensor ->
                    Tensor.create(TensorShape(labelShape).numElements().toFloat()).use { numberOfLossesTensor ->
                        Tensor.create(0.0f).use { isTraining ->
                            val lossAndMetricsTensors = session.runner()
                                .fetch(metricOp)
                                .fetch(TRAINING_LOSS)
                                .feed(xOp.asOutput(), testImagesTensor)
                                .feed(yTrueOp.asOutput(), testLabelsTensor)
                                .feed(training.asOutput(), isTraining)
                                .feed(
                                    numberOfLossesOp.asOutput(),
                                    numberOfLossesTensor
                                ) // TODO: change to number of loss pieces
                                .run()

                            val metricValue = lossAndMetricsTensors[0].floatValue()
                            val lossValue = lossAndMetricsTensors[1].floatValue()

                            averageMetricAccum += metricValue
                            averageLossAccum += lossValue

                            val batchEvent = BatchEvent(batchCounter, lossValue.toDouble(), metricValue.toDouble())
                            evaluationHistory.appendBatch(batchEvent)

                            callback.onTestBatchEnd(batchCounter, batchSize, batchEvent, evaluationHistory)
                        }
                    }

                }
            }

            batchCounter++
        }

        val avgMetricValue = (averageMetricAccum / batchCounter).toDouble()
        val avgLossValue = (averageLossAccum / batchCounter).toDouble()

        callback.onTestEnd(evaluationHistory)
        return EvaluationResult(avgLossValue, mapOf(Metrics.convertBack(metric) to avgMetricValue))
    }

    override fun predict(dataset: Dataset, batchSize: Int): IntArray {
        require(dataset.xSize() % batchSize == 0) { "The amount of images must be a multiple of batch size." }
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method." }

        callback.onPredictBegin()

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
                Tensor.create(0.0f).use { isTraining ->
                    val predictionsTensor = session.runner()
                        .fetch(predictionOp)
                        .feed(xOp.asOutput(), testImages)
                        .feed(training.asOutput(), isTraining)
                        .run()[0]

                    val dst = Array(imageShape[0].toInt()) { FloatArray(amountOfClasses.toInt()) { 0.0f } }

                    predictionsTensor.copyTo(dst)

                    val argMaxBatchPrediction = IntArray(imageShape[0].toInt()) { 0 }

                    dst.forEachIndexed { index, element ->
                        argMaxBatchPrediction[index] = element.indexOfFirst { it == element.maxOrNull()!! }
                    }

                    callback.onPredictBatchEnd(batchCounter, batchSize)
                    batchCounter++
                    argMaxBatchPrediction.copyInto(predictions, batchSize * (batchCounter - 1))
                }
            }
        }
        callback.onPredictEnd()
        return predictions
    }

    override fun predict(inputData: FloatArray): Int {
        val softPrediction = predictSoftly(inputData)
        return softPrediction.indexOfFirst { it == softPrediction.maxOrNull()!! }
    }

    override fun predict(inputData: FloatArray, predictionTensorName: String): Int {
        val softPrediction = predictSoftly(inputData, predictionTensorName)
        return softPrediction.indexOfFirst { it == softPrediction.maxOrNull()!! }
    }

    override fun predictAndGetActivations(inputData: FloatArray, predictionTensorName: String): Pair<Int, List<*>> {
        val (softPrediction, activations) = internalPredict(inputData, true, predictionTensorName)
        return Pair(softPrediction.indexOfFirst { it == softPrediction.maxOrNull()!! }, activations)
    }

    override fun predictSoftly(dataset: Dataset, batchSize: Int): Array<FloatArray> {
        require(dataset.xSize() % batchSize == 0) { "The amount of images must be a multiple of batch size." }
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method." }

        callback.onPredictBegin()

        val imageShape = calculateXShape(batchSize)

        val predictions = Array(dataset.xSize()) { FloatArray(amountOfClasses.toInt()) { 0.0f } }

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
                    .fetch(predictionOp)
                    .feed(xOp.asOutput(), testImages)
                    .run()[0]

                val dst = Array(imageShape[0].toInt()) { FloatArray(amountOfClasses.toInt()) { 0.0f } }

                predictionsTensor.copyTo(dst)

                callback.onPredictBatchEnd(batchCounter, batchSize)
                batchCounter++
                dst.copyInto(predictions, batchSize * (batchCounter - 1))
            }
        }
        callback.onPredictEnd()
        return predictions
    }

    override fun predictSoftly(inputData: FloatArray, predictionTensorName: String): FloatArray {
        val (softPrediction, _) = internalPredict(inputData, false, predictionTensorName)
        return softPrediction
    }

    override fun predictSoftlyAndGetActivations(
        inputData: FloatArray,
        predictionTensorName: String
    ): Pair<FloatArray, List<*>> {
        return internalPredict(inputData, true, predictionTensorName)
    }

    private fun internalPredict(
        inputData: FloatArray,
        visualizationIsEnabled: Boolean,
        predictionTensorName: String
    ): Pair<FloatArray, List<*>> {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method." }

        val imageShape = calculateXShape(1)

        Tensor.create(
            imageShape,
            FloatBuffer.wrap(inputData)
        ).use { testImages ->
            val tensors =
                formPredictionAndActivationsTensors(predictionTensorName, testImages, visualizationIsEnabled)

            val predictionsTensor = tensors[0]

            val dst = Array(1) { FloatArray(amountOfClasses.toInt()) { 0.0f } }

            predictionsTensor.copyTo(dst)

            val activations = mutableListOf<Any>()
            if (visualizationIsEnabled && tensors.size > 1) {
                for (i in 1 until tensors.size) {
                    activations.add(tensors[i].convertTensorToMultiDimArray())
                }
            }

            tensors.forEach { it.close() }
            return Pair(dst[0], activations.toList())
        }
    }

    private fun formPredictionAndActivationsTensors(
        predictionTensorName: String,
        testImages: Tensor<Float>,
        visualizationIsEnabled: Boolean
    ): List<Tensor<*>> {
        val runner = session
            .runner()

        if (predictionTensorName.isEmpty()) {
            runner
                .fetch(predictionOp)
                .feed(xOp.asOutput(), testImages)

        } else {
            require(kGraph().tfGraph.operation(predictionTensorName) != null) { "No such tensor output named [$predictionTensorName] in the TensorFlow graph!" }

            runner
                .fetch(predictionTensorName)
                .feed(xOp.asOutput(), testImages)
        }

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

    private fun forward(input: Operand<Float>): Operand<Float> {
        var out: Operand<Float> = input
        for (layer in layers) {
            out = layer.forward(tf, out, training, numberOfLossesOp)
        }
        return out
    }

    private fun calculateXShape(batchSize: Int): LongArray {
        return calculateXShape(batchSize.toLong())
    }

    private fun calculateXShape(amountOfImages: Long): LongArray {
        val xTensorShape = inputLayer.input.asOutput().shape()

        return longArrayOf(
            amountOfImages,
            *tail(xTensorShape)
        )
    }

    /**
     * Returns KGraph.
     *
     * NOTE: Be careful, this is a direct access to the model graph, not a copy.
     */
    public fun kGraph(): KGraph {
        return kGraph
    }

    override fun save(
        modelDirectory: File,
        savingFormat: SavingFormat,
        saveOptimizerState: Boolean,
        writingMode: WritingMode
    ) {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method." }
        check(isOptimizerVariableInitialized) { "The optimizer variables are not initialized yet. Initialize the optimizer variables with init() method or load optimizer weights to use this method." }

        val pathToModelDirectory = modelDirectory.absolutePath
        when (writingMode) {
            WritingMode.FAIL_IF_EXISTS -> {
                check(!modelDirectory.exists()) { "The directory exists on path $pathToModelDirectory, please be careful it could contain valuable model! Change this mode to OVERRIDE if you want to override this directory." }
                modelDirectory.mkdir()
            }
            WritingMode.OVERRIDE -> {
                if (modelDirectory.exists()) {
                    modelDirectory.deleteRecursively()
                }
                modelDirectory.mkdir()
            }
            WritingMode.APPEND -> {
                if (!modelDirectory.exists()) {
                    modelDirectory.mkdir()
                }
            }
        }

        when (savingFormat) {
            SavingFormat.TF_GRAPH_CUSTOM_VARIABLES -> saveInSimpleFormat(pathToModelDirectory, saveOptimizerState)
            SavingFormat.TF_GRAPH -> saveInSavedModelFormat(pathToModelDirectory)
            SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES -> saveInKerasFormat(pathToModelDirectory, saveOptimizerState)
        }
    }

    private fun saveInKerasFormat(pathToModelDirectory: String, saveOptimizerState: Boolean) {
        saveModel(pathToModelDirectory)
        saveVariables(pathToModelDirectory, saveOptimizerState)
    }

    private fun saveModel(pathToModelDirectory: String) {
        val jsonConfig = File("$pathToModelDirectory/modelConfig.json")
        this.saveModelConfiguration(jsonConfig)
    }

    private fun saveInSavedModelFormat(pathToModelDirectory: String) {
        saveGraphDef(pathToModelDirectory)
    }

    private fun saveInSimpleFormat(pathToModelDirectory: String, saveOptimizerState: Boolean) {
        saveGraphDef(pathToModelDirectory)
        saveVariables(pathToModelDirectory, saveOptimizerState)
    }

    private fun saveGraphDef(pathToModelDirectory: String) {
        val file = File("$pathToModelDirectory/graph.pb")
        file.writeBytes(kGraph.tfGraph.toGraphDef())
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

        val file = File("$pathToModelDirectory/variableNames.txt")

        file.bufferedWriter().use { variableNamesFile ->
            for ((index, tensorForCopying) in modelWeights.withIndex()) {
                val variableName = variables[index].asOutput().op().name()
                variableNamesFile.write(variableName)
                variableNamesFile.newLine()

                val variableNameFile = File("$pathToModelDirectory/$variableName.txt")

                variableNameFile.bufferedWriter().use { file ->

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

    override fun loadWeights(modelDirectory: File, loadOptimizerState: Boolean) {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(!isModelInitialized) { "The model is initialized already." }

        // Load variables names
        val file = File("${modelDirectory.absolutePath}/variableNames.txt")

        if (!file.exists()) throw FileNotFoundException(
            "File 'variableNames.txt' is not found. This file must be in the model directory. " +
                    "It is generated during Sequential model saving with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
        )

        val variableNames = file.readLines()

        if (variableNames.isNotEmpty()) {
            for (variableName in variableNames) {
                if (!loadOptimizerState && variableName.startsWith("optimizer")) // skip loading optimizers' variables
                    continue
                else if (loadOptimizerState && isOptimizerNameAndRelatedToFrozenLayer(variableName)) // skip loading optimizers' variables for frozen layers
                    continue
                else loadVariable(variableName, modelDirectory.absolutePath)
            }
        }

        isModelInitialized = true
        if (loadOptimizerState) isOptimizerVariableInitialized = true
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
    public infix fun getLayer(layerName: String): Layer {
        return layersByName[layerName] ?: error("No such layer $layerName in the model.")
    }

    /**
     * Formats and builds the model description.
     *
     * @return list of layer descriptions.
     */
    public fun summary(stringLayerNameTypeSize: Int = 30, stringOutputShapeSize: Int = 26): List<String> {
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

        val secondPart = TensorShape(l.outputShape).toString()

        stringBuilder.append(secondPart)

        for (i in 0 until stringOutputShapeSize - secondPart.length) {
            stringBuilder.append(" ")
        }

        stringBuilder.append(l.getParams())

        return stringBuilder.toString()
    }
}
