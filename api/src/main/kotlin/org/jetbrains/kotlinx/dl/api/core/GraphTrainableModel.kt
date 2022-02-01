/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import mu.KLogger
import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.exception.RepeatableLayerNameException
import org.jetbrains.kotlinx.dl.api.core.history.*
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.NoGradients
import org.jetbrains.kotlinx.dl.api.core.layer.core.ActivationLayer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.LossFunction
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.loss.SoftmaxCrossEntropyWithLogits
import org.jetbrains.kotlinx.dl.api.core.metric.EvaluationResult
import org.jetbrains.kotlinx.dl.api.core.metric.Metric
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Optimizer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.shape.tail
import org.jetbrains.kotlinx.dl.api.core.summary.LayerSummary
import org.jetbrains.kotlinx.dl.api.core.summary.ModelSummary
import org.jetbrains.kotlinx.dl.api.core.util.*
import org.jetbrains.kotlinx.dl.api.extension.argmax
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToFlattenFloatArray
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToMultiDimArray
import org.jetbrains.kotlinx.dl.api.inference.keras.saveModelConfiguration
import org.jetbrains.kotlinx.dl.dataset.DataBatch
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import java.io.File
import java.io.FileNotFoundException
import java.nio.FloatBuffer
import java.nio.file.Files
import java.nio.file.Paths
import java.util.*

/**
 * [GraphTrainableModel] model groups a linear stack of layers into a graph-based TensorFlow Model.
 * Also, it provides training and inference features on this model.
 *
 * @constructor Creates a [GraphTrainableModel] model via sequence of [layers].
 */
public abstract class GraphTrainableModel(vararg layers: Layer) : TrainableModel() {
    /** Logger for the model. */
    public val logger: KLogger = KotlinLogging.logger {}

    /** The layers to describe the model design. Main part of the internal state of the model. */
    public var layers: List<Layer> = listOf(*layers)

    /** First layer that is responsible for the input shape of the Neural Network. */
    public val inputLayer: Input
        get() = layers[0] as Input

    /** Returns input dimensions in order HWC (height, width, channels) */
    public override val inputDimensions: LongArray
        get() {
            return (layers[0] as Input).packedDims
        }

    /** Layers indexed by name. */
    protected var layersByName: Map<String, Layer> = mapOf()

    /** TensorFlow operand for prediction phase. */
    private lateinit var yPredOp: Operand<Float>

    /** TensorFlow loss operand. */
    protected lateinit var lossOp: Operand<Float>

    /** TensorFlow prediction operand. */
    private lateinit var predictionOp: Operand<Float>

    /** TensorFlow prediction operand. */
    private lateinit var metricOps: MutableList<Operand<Float>>

    /** A list of targets to be optimized. */
    protected lateinit var targets: List<Operand<Float>>

    /** TensorFlow operand for X data. */
    private lateinit var xOp: Operand<Float>

    /** TensorFlow operand for Y data. */
    private lateinit var yTrueOp: Operand<Float>

    /** TensorFlow operand for batch size data. */
    protected lateinit var numberOfLossesOp: Operand<Float>

    /** TensorFlow operand for switching between training and inference modes. */
    protected lateinit var training: Operand<Boolean>

    init {
        for (layer in layers) {
            if (layersByName.containsKey(layer.name)) {
                throw RepeatableLayerNameException(layer.name)
            } else {
                layersByName = layersByName + (layer.name to layer)
            }

            if (layer.parentModel != null) logger.warn { "Layer ${layer.name} is a part of model ${layer.parentModel}" }

            layer.parentModel = this
        }


        kGraph = KGraph(Graph().toGraphDef())
        tf = Ops.create(kGraph.tfGraph)
        session = Session(kGraph.tfGraph)
    }

    /** Helper method for preprocessing layer names and layer validation. */
    internal companion object {
        internal fun preProcessLayerNames(layers: Array<out Layer>) {
            var cnt = 1
            for (layer in layers) {
                if (layer.name.isEmpty()) {
                    val generatedLayerName =
                        (layer::class.simpleName ?: return).lowercase(Locale.getDefault()) + "_" + cnt
                    layer.name = generatedLayerName
                    cnt++
                }
            }
        }

        internal fun layerValidation(layers: List<Layer>) {
            require(layers.isNotEmpty()) { "Model should contain layers!" }
            val input = layers[0]
            require(input is Input) { "Model should start from the Input layer" }
        }
    }

    override fun compile(optimizer: Optimizer, loss: Losses, metric: Metrics, callback: Callback) {
        compile(optimizer, Losses.convert(loss), Metrics.convert(metric), callback)
    }

    override fun compile(optimizer: Optimizer, loss: LossFunction, metric: Metric, callback: Callback) {
        compile(optimizer, loss, listOf(metric), callback)
    }

    override fun compile(optimizer: Optimizer, loss: LossFunction, metrics: List<Metric>, callback: Callback) {
        check(!isModelCompiled) { "The model is compiled already. Graph is created. Create new model and compile it." }

        validateModelArchitecture()

        this.loss = loss
        this.metrics = metrics
        this.optimizer = optimizer

        // callback binding
        this.callback = callback
        this.callback.model = this

        buildLayers()

        // should be after outputShape calculation
        numberOfClasses = when (val lastLayer = layers.last()) {
            is Dense -> lastLayer.outputSize.toLong()
            is ActivationLayer -> lastLayer.outputShape.tail().last()  // valid for mobileNet/DenseNet
            else -> 1
        }

        xOp = inputLayer.input
        yTrueOp = tf.placeholder(getDType()) as Operand<Float>
        numberOfLossesOp = tf.withName("numberOfLosses").placeholder(
            getDType(),
            Placeholder.shape(Shape.scalar())
        )

        training = tf.withName("training").placeholder(
            Boolean::class.javaObjectType,
            Placeholder.shape(Shape.scalar())
        )

        yPredOp = forward(xOp, inputLayer)
        lossOp = buildLossFunction(loss)
        targets = optimizer.prepareTargets(kGraph, tf, lossOp)

        predictionOp = when (loss) {
            is SoftmaxCrossEntropyWithLogits -> tf.withName(OUTPUT_NAME).nn.softmax(yPredOp)
            else -> tf.withName(OUTPUT_NAME).identity(yPredOp)
        }
        metricOps = mutableListOf()
        metrics.forEach {
            metricOps.add(it.apply(tf, predictionOp, yTrueOp, numberOfLossesOp))
        }

        isModelCompiled = true
    }

    private fun buildLossFunction(loss: LossFunction): Operand<Float> {
        val basicLoss = loss.apply(tf, yPredOp, yTrueOp, numberOfLossesOp)
        var totalLoss = basicLoss
        // TODO: probably regularization output should be divided on numberOfLossesOp and changed together with loss before averaging
        kGraph.variableRegularizers.forEach { (variable, regularizer) ->
            run {
                totalLoss = tf.math.add(totalLoss, regularizer.apply(tf, variable))
            }
        }
        return tf.withName(TRAINING_LOSS).identity(totalLoss)
    }


    override fun compile(optimizer: Optimizer, loss: Losses, metric: Metric, callback: Callback) {
        compile(optimizer, Losses.convert(loss), metric, callback)
    }

    override fun compile(optimizer: Optimizer, loss: LossFunction, metric: Metrics, callback: Callback) {
        compile(optimizer, loss, Metrics.convert(metric), callback)
    }

    /** Validates architecture. */
    private fun validateModelArchitecture() {
        require(layers.none { it is NoGradients && it.isTrainable })
        {
            "All layers that implements NoGradient interface should be frozen (status isTrainable==false). " +
                    "But the following layers violates this rule: ${
                        layers.filter { it is NoGradients && it.isTrainable }.map { it.name }.toTypedArray()
                            .contentDeepToString()
                    }"
        }
        //  require(layers.last() is Dense) { "DL architectures are not finished with Dense layer are not supported yet!" }
        //  require(layers.last().hasActivation()) { "Last layer must have an activation function." }
        //  require((layers.last() as Dense).activation != Activations.Sigmoid) { "The last dense layer should have Linear activation, alternative activations are not supported yet!" }
    }

    /** Common method for building the initial part of the model static graph layer by layer via calling build() method on each layer in correct order. */
    protected abstract fun buildLayers()

    /** Forms forward path as a part of the model static graph layer by layer via calling forward() method on each layer in correct order. */
    protected abstract fun forward(input: Operand<Float>, inputLayer: Input): Operand<Float>

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

        for (layer in layers) {
            check(!(layer is NoGradients && layer.isTrainable)) {
                "Layer $layer has no gradients implementations in TensorFlow and should be non-trainable only. " +
                        "Set 'isTrainable' to 'false'."
            }
        }

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
                var averageTrainingLossAccum = 0.0f
                val averageTrainingMetricAccum = FloatArray(metrics.size) { 0.0f }

                while (batchIter.hasNext() && !stopTraining) {
                    callback.onTrainBatchBegin(batchCounter, trainBatchSize, trainingHistory)
                    val batch: DataBatch = batchIter.next()

                    val (xBatchShape, yBatchShape) = calculateXYShapes(batch)

                    Tensor.create(
                        xBatchShape,
                        serializeToBuffer(batch.x)
                    ).use { batchImagesTensor ->
                        Tensor.create(yBatchShape, serializeLabelsToBuffer(batch.y, numberOfClasses))
                            .use { batchLabelsTensor ->
                                Tensor.create(TensorShape(yBatchShape).numElements().toFloat())
                                    .use { numberOfLossesTensor ->
                                        Tensor.create(true).use { isTraining ->
                                            val (lossValue, metricValues) = trainOnBatch(
                                                targets,
                                                batchImagesTensor,
                                                batchLabelsTensor,
                                                numberOfLossesTensor as Tensor<Float>,
                                                isTraining as Tensor<Float>,
                                                metricOps
                                            )
                                            if (lossValue.isNaN() || lossValue == Float.POSITIVE_INFINITY || lossValue == Float.NEGATIVE_INFINITY) {
                                                logger.debug { "Loss function value is NaN. You could use TerminateOnNaN callback to stop it earlier." }
                                            }

                                            averageTrainingLossAccum += lossValue
                                            metrics.forEachIndexed { i, _ ->
                                                averageTrainingMetricAccum[i] += metricValues[i]
                                            }

                                            val batchTrainingEvent =
                                                BatchTrainingEvent(
                                                    i,
                                                    batchCounter,
                                                    lossValue.toDouble(),
                                                    averageTrainingMetricAccum.map { it.toDouble() }
                                                )
                                            trainingHistory.appendBatch(batchTrainingEvent)

                                            // TODO: create map (metric name and metric value)
                                            logger.debug { "Batch stat: { lossValue: $lossValue metricValues: $metricValues }" }

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

                val avgTrainingMetricValue = FloatArray(metrics.size) { 0.0f }
                averageTrainingMetricAccum.forEachIndexed { index, metricValue ->  avgTrainingMetricValue[index] = metricValue / batchCounter}

                val avgLossValue = (averageTrainingLossAccum / batchCounter)

                val nanList = mutableListOf<Double>()
                for(j in 1 .. metrics.size) {
                    nanList.add(Double.NaN)
                }

                val epochTrainingEvent = EpochTrainingEvent(
                    i,
                    avgLossValue.toDouble(), avgTrainingMetricValue.map { it.toDouble() }.toMutableList(), Double.NaN, nanList
                )

                if (validationIsEnabled) {
                    val evaluationResult = evaluate(validationDataset!!, validationBatchSize!!)
                    val validationMetricValues = metrics.map { evaluationResult.metrics[Metrics.convertBack(it)] }.toList()// TODO: probably I should it by name, not by type
                    val validationLossValue = evaluationResult.lossValue
                    epochTrainingEvent.valLossValue = validationLossValue
                    epochTrainingEvent.valMetricValues = validationMetricValues!!
                    logger.info { "epochs: $i loss: $avgLossValue metric: ${avgTrainingMetricValue.contentToString()} val loss: $validationLossValue val metrics: $validationMetricValues" } // TODO: check printing for validation
                } else {
                    logger.info { "epochs: $i loss: $avgLossValue metric: ${avgTrainingMetricValue.contentToString()}" }
                }
                trainingHistory.appendEpoch(epochTrainingEvent)
                callback.onEpochEnd(i, epochTrainingEvent, trainingHistory)
            }
        }
        callback.onTrainEnd(trainingHistory)
        return trainingHistory
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
        metricOps: MutableList<Operand<Float>>
    ): Pair<Float, List<Float>> {
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

        metricOps.forEach {
            runner.fetch(it)
        }

        try {
            val tensorList = runner.run()
            val lossValue = tensorList[0].floatValue()
            val metricValues = mutableListOf<Float>()

            check(tensorList.size == metricOps.size + 1) { "${metricOps.size} metrics are monitored, but ${tensorList.size - 1} metrics are returned!" }
            for (i in 1 .. metricOps.size) {
                metricValues.add(tensorList[i].floatValue())
            }

            return Pair(lossValue, metricValues)
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

        val averageMetricAccum = FloatArray(metrics.size) { 0.0f }
        var averageLossAccum = 0.0f
        var batchCounter = 0

        while (batchIter.hasNext()) {
            callback.onTestBatchBegin(batchCounter, batchSize, evaluationHistory)
            val batch: DataBatch = batchIter.next()
            val (imageShape, labelShape) = calculateXYShapes(batch)

            Tensor.create(
                imageShape,
                serializeToBuffer(batch.x)
            ).use { testImagesTensor ->
                Tensor.create(labelShape, serializeLabelsToBuffer(batch.y, numberOfClasses)).use { testLabelsTensor ->
                    Tensor.create(TensorShape(labelShape).numElements().toFloat()).use { numberOfLossesTensor ->
                        Tensor.create(false).use { isTraining ->
                            val runner = session.runner()
                                .fetch(TRAINING_LOSS)

                            metricOps.forEach {
                                runner.fetch(it)
                            }

                            val lossAndMetricsTensors = runner
                                .feed(xOp.asOutput(), testImagesTensor)
                                .feed(yTrueOp.asOutput(), testLabelsTensor)
                                .feed(training.asOutput(), isTraining)
                                .feed(
                                    numberOfLossesOp.asOutput(),
                                    numberOfLossesTensor
                                )
                                .run()

                            val lossValue = lossAndMetricsTensors[0].floatValue()
                            val metricValues = mutableListOf<Float>()

                            check(lossAndMetricsTensors.size == metricOps.size + 1) { "${metricOps.size} metrics are monitored, but ${lossAndMetricsTensors.size - 1} metrics are returned!" }
                            for (i in 1 .. metricOps.size) {
                                metricValues.add(lossAndMetricsTensors[i].floatValue())
                            }

                            averageLossAccum += lossValue
                            metrics.forEachIndexed { i, _ ->
                                averageMetricAccum[i] += metricValues[i]
                            }

                            val batchEvent = BatchEvent(batchCounter, lossValue.toDouble(), averageMetricAccum.map { it.toDouble() })
                            evaluationHistory.appendBatch(batchEvent)

                            callback.onTestBatchEnd(batchCounter, batchSize, batchEvent, evaluationHistory)
                        }
                    }

                }
            }

            batchCounter++
        }

        val avgMetricValue = FloatArray(metrics.size) { 0.0f }
        averageMetricAccum.forEachIndexed { index, metricValue ->  avgMetricValue[index] = metricValue / batchCounter}

        val avgLossValue = (averageLossAccum / batchCounter).toDouble()

        callback.onTestEnd(evaluationHistory)
        val metricValues = mutableMapOf<Metrics, Double>() // TODO: Metrics -> Metric class
        metrics.forEachIndexed { index, metric ->
            metricValues[Metrics.convertBack(metric)] = avgMetricValue[index].toDouble()
        }

        return EvaluationResult(avgLossValue, metricValues)
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
                serializeToBuffer(batch.x)
            ).use { testImages ->
                Tensor.create(false).use { isTraining ->
                    val predictionsTensor = session.runner()
                        .fetch(predictionOp)
                        .feed(xOp.asOutput(), testImages)
                        .feed(training.asOutput(), isTraining)
                        .run()[0]

                    val dst = Array(imageShape[0].toInt()) { FloatArray(numberOfClasses.toInt()) { 0.0f } }

                    predictionsTensor.copyTo(dst)

                    val argMaxBatchPrediction = IntArray(imageShape[0].toInt()) { 0 }

                    dst.forEachIndexed { index, element ->
                        argMaxBatchPrediction[index] = element.argmax()
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
        return softPrediction.argmax()
    }

    override fun predict(inputData: FloatArray, predictionTensorName: String): Int {
        val softPrediction = predictSoftly(inputData, predictionTensorName)
        return softPrediction.argmax()
    }

    override fun predictAndGetActivations(inputData: FloatArray, predictionTensorName: String): Pair<Int, List<*>> {
        val (softPrediction, activations) = internalPredict(inputData, true, predictionTensorName)
        return Pair(softPrediction.argmax(), activations)
    }

    override fun predictSoftly(dataset: Dataset, batchSize: Int): Array<FloatArray> {
        require(dataset.xSize() % batchSize == 0) { "The amount of images must be a multiple of batch size." }
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method." }

        callback.onPredictBegin()

        val imageShape = calculateXShape(batchSize)

        val predictions = Array(dataset.xSize()) { FloatArray(numberOfClasses.toInt()) { 0.0f } }

        val batchIter: Dataset.BatchIterator = dataset.batchIterator(
            batchSize
        )

        var batchCounter = 0

        while (batchIter.hasNext()) {
            callback.onPredictBatchBegin(batchCounter, batchSize)

            val batch: DataBatch = batchIter.next()

            Tensor.create(
                imageShape,
                serializeToBuffer(batch.x)
            ).use { testImages ->
                val predictionsTensor = session.runner()
                    .fetch(predictionOp)
                    .feed(xOp.asOutput(), testImages)
                    .run()[0]

                val dst = Array(imageShape[0].toInt()) { FloatArray(numberOfClasses.toInt()) { 0.0f } }

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

            val prediction = tensors[0].convertTensorToFlattenFloatArray()

            val activations = mutableListOf<Any>()
            if (visualizationIsEnabled && tensors.size > 1) {
                for (i in 1 until tensors.size) {
                    activations.add(tensors[i].convertTensorToMultiDimArray())
                }
            }

            tensors.forEach { it.close() }
            return Pair(prediction, activations.toList())
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
                if (layer.hasActivation && layer != layers.last()) runner.fetch(defaultActivationName(layer))
            }
        }
        return runner.run()
    }

    private fun calculateXYShapes(batch: DataBatch): Pair<LongArray, LongArray> {
        val batchSize = batch.size

        val xBatchShape = calculateXShape(batchSize)

        val yBatchShape = calculateYShape(batchSize)

        if (batchSize > 0) {
            batchValidation(batch, xBatchShape, yBatchShape)
        }

        return Pair(xBatchShape, yBatchShape)
    }

    private fun calculateYShape(batchSize: Int) = longArrayOf(
        batchSize.toLong(),
        numberOfClasses
    )

    private fun batchValidation(
        batch: DataBatch,
        xBatchShape: LongArray,
        yBatchShape: LongArray
    ) {
        check(
            TensorShape(xBatchShape).numElements().toInt() == batch.x.size * batch.x[0].size
        )
        {
            "The calculated [from the Model] data batch shape ${xBatchShape.contentToString()} doesn't match actual data buffer size ${
                batch.x.size * batch.x[0].size
            }. Please, check input data."
        }
        check(
            TensorShape(yBatchShape).numElements().toInt() == batch.y.size * numberOfClasses.toInt()
        )
        {
            "The calculated [from the model] label batch shape ${yBatchShape.contentToString()} doesn't match actual data buffer size ${
                batch.y.size * numberOfClasses.toInt()
            }. " +
                    "\nPlease, check the input label data or correct number of classes [number of neurons] in last Dense layer, if you have a classification problem." +
                    "\nHighly likely, you have different number of classes presented in data and described in model as desired output."
        }
    }

    private fun calculateXShape(batchSize: Int): LongArray {
        val inputLayer = layers.first() as Input

        val xTensorShape = inputLayer.input.asOutput().shape()

        return longArrayOf(
            batchSize.toLong(),
            *tail(xTensorShape)
        )
    }

    /**
     * Returns KGraph.
     *
     * NOTE: Be careful, this is direct access to the model graph, not a copy.
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
        if (saveOptimizerState) {
            check(isOptimizerVariableInitialized) { "The optimizer variables are not initialized yet. Initialize the optimizer variables with init() method or load optimizer weights to use this method." }
        }

        val pathToModelDirectory = modelDirectory.absolutePath
        when (writingMode) {
            WritingMode.FAIL_IF_EXISTS -> {
                check(!modelDirectory.exists()) { "The directory exists on path $pathToModelDirectory, please be careful it could contain valuable model! Change this mode to OVERRIDE if you want to override this directory." }
                Files.createDirectories(modelDirectory.toPath())
                modelDirectory.mkdir()
            }
            WritingMode.OVERRIDE -> {
                if (modelDirectory.exists()) {
                    modelDirectory.deleteRecursively()
                }
                Files.createDirectories(modelDirectory.toPath())
                modelDirectory.mkdir()
            }
            WritingMode.APPEND -> {
                if (!modelDirectory.exists()) {
                    Files.createDirectories(modelDirectory.toPath())
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
        Files.createDirectories(Paths.get(pathToModelDirectory))
        file.writeBytes(kGraph.tfGraph.toGraphDef())
    }

    /** Saves variables and optimizer state if [saveOptimizerState] is enabled in txt format to the [pathToModelDirectory] directory.*/
    protected fun saveVariables(pathToModelDirectory: String, saveOptimizerState: Boolean) {
        val pair = getVariablesAndTensors(saveOptimizerState)
        val variables = pair.first
        val modelWeights = pair.second

        Files.createDirectories(Paths.get(pathToModelDirectory))
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

        Files.createDirectories(modelDirectory.toPath())
        // Load variables names
        val file = File("${modelDirectory.absolutePath}/variableNames.txt")

        if (!file.exists()) throw FileNotFoundException(
            "File 'variableNames.txt' is not found. This file must be in the model directory. " +
                    "It is generated during Sequential model saving with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
        )

        val variableNames = file.readLines()
        // TODO: common code could be refactored with the link to the function (load variable)
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

    /**
     * Return layer by [layerName].
     *
     * @param [layerName] Should be existing layer name. Throws an error otherwise.
     */
    public infix fun getLayer(layerName: String): Layer {
        return layersByName[layerName] ?: error("No such layer $layerName in the model.")
    }

    override fun toString(): String {
        return "GraphTrainableModel(numberOfLayers=${layers.size}) ${super.toString()}"
    }

    public override fun summary(): ModelSummary {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }

        val (trainableLayers, frozenLayers) = layers.partition { it.isTrainable }

        return ModelSummary(
            type = this::class.simpleName.toString(),
            name = name,
            layersSummaries = layers.map { layer ->
                LayerSummary(
                    name = layer.name,
                    type = layer::class.simpleName.toString(),
                    outputShape = layer.outputShape,
                    paramsCount = layer.paramCount.toLong(),
                    inboundLayers = layer.inboundLayers.map { it.name }
                )
            },
            trainableParamsCount = trainableLayers.sumOf { it.paramCount.toLong() },
            frozenParamsCount = frozenLayers.sumOf { it.paramCount.toLong() },
        )
    }
}
