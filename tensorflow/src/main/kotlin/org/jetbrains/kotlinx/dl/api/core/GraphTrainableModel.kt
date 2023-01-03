/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import mu.KLogger
import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.exception.RepeatableLayerNameException
import org.jetbrains.kotlinx.dl.api.core.history.*
import org.jetbrains.kotlinx.dl.api.core.layer.*
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
import org.jetbrains.kotlinx.dl.api.core.summary.TfModelSummary
import org.jetbrains.kotlinx.dl.api.core.util.*
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToFlattenFloatArray
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToMultiDimArray
import org.jetbrains.kotlinx.dl.api.inference.keras.saveModelConfiguration
import org.jetbrains.kotlinx.dl.dataset.DataBatch
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.impl.util.argmax
import org.jetbrains.kotlinx.dl.impl.util.use
import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Variable
import java.io.File
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

    /** TensorFlow wrapped computational graph. */
    public val kGraph: KGraph = KGraph(tfGraph)

    /** The namespace wrapper for all TensorFlow graph operations. */
    protected val tf: Ops = Ops.create(tfGraph)

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
    private lateinit var metricOps: List<Operand<Float>>

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

            @Suppress("LeakingThis")
            layer.parentModel = this
        }
    }

    /**
     * Returns a list of layer variables in this model.
     */
    private fun layerVariables(): List<KVariable> = layers.variables()

    /**
     * Returns a list of non-trainable, 'frozen' layer variables in this model.
     */
    private fun frozenLayerVariables(): List<KVariable> = layers.frozenVariables()

    override fun reshape(vararg dims: Long) {
        throw UnsupportedOperationException("Reshaping model $this is not supported.")
    }

    override fun compile(optimizer: Optimizer, loss: Losses, metric: Metrics) {
        compile(optimizer, Losses.convert(loss), Metric.convert(metric))
    }

    override fun compile(optimizer: Optimizer, loss: LossFunction, metric: Metric) {
        compile(optimizer, loss, listOf(metric))
    }

    override fun compile(optimizer: Optimizer, loss: LossFunction, metrics: List<Metric>) {
        check(!isModelCompiled) { "The model is compiled already. Graph is created. Create new model and compile it." }

        this.loss = loss
        this.metrics = metrics
        this.optimizer = optimizer

        training = tf.withName("training").placeholder(
            Boolean::class.javaObjectType,
            Placeholder.shape(Shape.scalar())
        )
        numberOfLossesOp = tf.withName("numberOfLosses").placeholder(
            getDType(),
            Placeholder.shape(Shape.scalar())
        )

        val (input, output) = buildLayers(training, numberOfLossesOp)
        xOp = input
        yPredOp = output

        // should be after outputShape calculation
        numberOfClasses = when (val lastLayer = layers.last()) {
            is Dense -> lastLayer.outputSize.toLong()
            is ActivationLayer -> lastLayer.outputShape.tail().last()  // valid for mobileNet/DenseNet
            else -> 1
        }

        yTrueOp = tf.placeholder(getDType()) as Operand<Float>
        lossOp = buildLossFunction(loss)
        targets = optimizer.prepareTargets(kGraph, layers.trainableVariables().map { it.variable }, tf, lossOp)

        predictionOp = when (loss) {
            is SoftmaxCrossEntropyWithLogits -> tf.withName(OUTPUT_NAME).nn.softmax(yPredOp)
            else -> tf.withName(OUTPUT_NAME).identity(yPredOp)
        }

        metricOps = metrics.map { it.apply(tf, predictionOp, yTrueOp, numberOfLossesOp) }

        isModelCompiled = true
    }

    private fun buildLossFunction(loss: LossFunction): Operand<Float> {
        val basicLoss = loss.apply(tf, yPredOp, yTrueOp, numberOfLossesOp)
        var totalLoss = basicLoss
        // TODO: probably regularization output should be divided on numberOfLossesOp and changed together with loss before averaging
        layers.trainableVariables().forEach { variable ->
            variable.regularizer?.let { regularizer ->
                totalLoss = tf.math.add(totalLoss, regularizer.apply(tf, variable.variable))
            }
        }
        return tf.withName(TRAINING_LOSS).identity(totalLoss)
    }


    override fun compile(optimizer: Optimizer, loss: Losses, metric: Metric) {
        compile(optimizer, Losses.convert(loss), metric)
    }

    override fun compile(optimizer: Optimizer, loss: LossFunction, metric: Metrics) {
        compile(optimizer, loss, Metric.convert(metric))
    }

    /** Common method for building model static graph layer by layer via calling build() method on each layer in correct order. */
    protected abstract fun buildLayers(
        training: Operand<Boolean>,
        numberOfLosses: Operand<Float>
    ): Pair<Placeholder<Float>, Operand<Float>>

    override fun fit(
        trainingDataset: Dataset,
        validationDataset: Dataset,
        epochs: Int,
        trainBatchSize: Int,
        validationBatchSize: Int,
        callbacks: List<Callback>
    ): TrainingHistory {
        return internalFit(
            trainBatchSize,
            epochs,
            trainingDataset,
            true,
            validationDataset,
            validationBatchSize,
            callbacks
        )
    }

    override fun fit(dataset: Dataset, epochs: Int, batchSize: Int, callbacks: List<Callback>): TrainingHistory {
        return internalFit(batchSize, epochs, dataset, false, null, null, callbacks)
    }

    /**
     * Initializes kGraph variables.
     *
     * NOTE: The model becomes initialized after this method call. The flag [isModelInitialized] is set to True.
     */
    public fun init() {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(!isModelInitialized) { "Model is initialized already!" }
        check(!isOptimizerVariableInitialized) { "Optimizer variables are initialized already!" }

        logger.debug { "Initialization of TensorFlow Graph variables." }
        layers.initializeVariables(session)
        isModelInitialized = true
    }

    /**
     * It ignores that model is initialized already and call initializers under the hood to re-initialize [kGraph] variables.
     *
     * NOTE: The model becomes initialized after this method call.
     * The flag [isModelInitialized] is set to True and the flag [isOptimizerVariableInitialized] is set to False.
     * As a result, when the method ```fit()``` will be called, optimizer variables are re-initialized.
     */
    public fun reset() {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }

        logger.debug { "Initialization of TensorFlow Graph variables." }
        layers.initializeVariables(session)
        isModelInitialized = true
        isOptimizerVariableInitialized = false
    }

    private fun internalFit(
        trainBatchSize: Int,
        epochs: Int,
        trainingDataset: Dataset,
        validationIsEnabled: Boolean,
        validationDataset: Dataset?,
        validationBatchSize: Int?,
        fitCallbacks: List<Callback>
    ): TrainingHistory {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }

        if (!isModelInitialized) {
            logger.debug { "Initialization of TensorFlow Graph variables." }
            layers.initializeVariables(session)
            isModelInitialized = true
        }

        val trainingHistory = TrainingHistory()

        if (!isOptimizerVariableInitialized) {
            logger.debug { "Initialization of optimizer variables." }
            kGraph.initializeOptimizerVariables(session)
            isOptimizerVariableInitialized = true
        }

        // callback binding
        fitCallbacks.forEach { it.model = this }
        fitCallbacks.forEach { it.onTrainBegin() }

        for (epoch in 1..epochs) {
            if (stopTraining) break

            fitCallbacks.forEach { it.onEpochBegin(epoch, trainingHistory) }

            var batchCounter = 0
            var averageTrainingLossAccum = 0.0f
            val averageTrainingMetricAccum = FloatArray(metrics.size) { 0.0f }

            for (batch in trainingDataset.batchSequence(trainBatchSize)) {
                if (stopTraining) break

                fitCallbacks.forEach { it.onTrainBatchBegin(batchCounter, trainBatchSize, trainingHistory) }

                val (lossValue, metricValues) = getLossAndMetricValues(batch, true)
                if (lossValue.isNaN() || lossValue == Float.POSITIVE_INFINITY || lossValue == Float.NEGATIVE_INFINITY) {
                    logger.debug { "Loss function value is NaN. You could use TerminateOnNaN callback to stop it earlier." }
                }

                averageTrainingLossAccum += lossValue
                metrics.indices.forEach { i -> averageTrainingMetricAccum[i] += metricValues[i] }

                val batchTrainingEvent = BatchTrainingEvent(epoch, batchCounter, lossValue.toDouble(),
                                                            averageTrainingMetricAccum.map { it.toDouble() })
                trainingHistory.appendBatch(batchTrainingEvent)

                // TODO: create map (metric name and metric value)
                logger.debug { "Batch stat: { lossValue: $lossValue metricValues: $metricValues }" }

                fitCallbacks.forEach {
                    it.onTrainBatchEnd(batchCounter, trainBatchSize, batchTrainingEvent, trainingHistory)
                }
                batchCounter++
            }

            val avgTrainingMetricValue = FloatArray(metrics.size) {
                averageTrainingMetricAccum[it] / batchCounter
            }

            val avgLossValue = (averageTrainingLossAccum / batchCounter)

            val epochTrainingEvent = EpochTrainingEvent(
                epoch,
                avgLossValue.toDouble(),
                avgTrainingMetricValue.map { it.toDouble() }.toMutableList(),
                Double.NaN,
                List(metrics.size) { Double.NaN }
            )

            if (validationIsEnabled) {
                val evaluationResult = evaluate(validationDataset!!, validationBatchSize!!, listOf())
                val validationMetricValues = metrics.map { evaluationResult.metrics[Metric.convertBack(it)] }.toList()
                // TODO: probably I should it by name, not by type
                val validationLossValue = evaluationResult.lossValue
                epochTrainingEvent.valLossValue = validationLossValue
                epochTrainingEvent.valMetricValues = validationMetricValues
                logger.info { "epochs: $epoch loss: $avgLossValue metric: ${avgTrainingMetricValue.contentToString()} val loss: $validationLossValue val metrics: $validationMetricValues" } // TODO: check printing for validation
            } else {
                logger.info { "epochs: $epoch loss: $avgLossValue metric: ${avgTrainingMetricValue.contentToString()}" }
            }
            trainingHistory.appendEpoch(epochTrainingEvent)
            fitCallbacks.forEach { it.onEpochEnd(epoch, epochTrainingEvent, trainingHistory) }
        }
        fitCallbacks.forEach { it.onTrainEnd(trainingHistory) }
        return trainingHistory
    }

    /**
     * Returns the loss value and metric value on train batch.
     */
    private fun getLossAndMetricValues(batch: DataBatch, isTraining: Boolean): Pair<Float, List<Float>> {
        val (xBatchShape, yBatchShape) = calculateXYShapes(batch)
        return Tensor.create(xBatchShape, serializeToBuffer(batch.x))
            .use { xTensor ->
                Tensor.create(yBatchShape, serializeLabelsToBuffer(batch.y, numberOfClasses))
                    .use { yTensor ->
                        Tensor.create(TensorShape(yBatchShape).numElements().toFloat())
                            .use { numberOfLossesTensor ->
                                Tensor.create(isTraining).use { isTrainingTensor ->
                                    val runner = session.runner()
                                        .feed(xOp.asOutput(), xTensor)
                                        .feed(yTrueOp.asOutput(), yTensor)
                                        .feed(numberOfLossesOp.asOutput(), numberOfLossesTensor)
                                        .feed(training.asOutput(), isTrainingTensor)

                                    runner.fetch(TRAINING_LOSS)

                                    metricOps.forEach { runner.fetch(it) }

                                    if (isTraining) {
                                        targets.forEach { runner.addTarget(it) }
                                    }

                                    runner.run().use { tensors ->
                                        check(tensors.size == metricOps.size + 1) { "${metricOps.size} metrics are monitored, but ${tensors.size - 1} metrics are returned!" }
                                        tensors.first().floatValue() to tensors.drop(1).map { it.floatValue() }
                                    }
                                }
                            }
                    }
            }
    }

    override fun evaluate(dataset: Dataset, batchSize: Int, callbacks: List<Callback>): EvaluationResult {
        checkModelInitialized()

        val evaluationHistory = History()

        callbacks.forEach { it.model = this }
        callbacks.forEach { it.onTestBegin() }

        var batchCounter = 0
        var averageLossAccum = 0.0f
        val averageMetricAccum = FloatArray(metrics.size) { 0.0f }

        for (batch in dataset.batchSequence(batchSize)) {
            callbacks.forEach { it.onTestBatchBegin(batchCounter, batchSize, evaluationHistory) }

            val (lossValue, metricValues) = getLossAndMetricValues(batch, false)

            averageLossAccum += lossValue
            metrics.indices.forEach { i -> averageMetricAccum[i] += metricValues[i] }

            val batchEvent = BatchEvent(batchCounter, lossValue.toDouble(), averageMetricAccum.map { it.toDouble() })
            evaluationHistory.appendBatch(batchEvent)

            callbacks.forEach {
                it.onTestBatchEnd(batchCounter, batchSize, batchEvent, evaluationHistory)
            }

            batchCounter++
        }

        val avgMetricValue = FloatArray(metrics.size) { averageMetricAccum[it] / batchCounter }
        val avgLossValue = (averageLossAccum / batchCounter).toDouble()

        callbacks.forEach { it.onTestEnd(evaluationHistory) }
        val metricValues = metrics.withIndex().associate { (index, metric) ->
            Metric.convertBack(metric) to avgMetricValue[index].toDouble()  // TODO: Metrics -> Metric class
        }

        return EvaluationResult(avgLossValue, metricValues)
    }

    override fun predict(dataset: Dataset, batchSize: Int, callbacks: List<Callback>): IntArray {
        require(dataset.xSize() % batchSize == 0) { "The number of elements in the dataset must be a multiple of batch size." }
        checkModelInitialized()

        val predictions = IntArray(dataset.xSize()) { Int.MIN_VALUE }
        val buffer = Array(batchSize) { FloatArray(numberOfClasses.toInt()) { 0.0f } }
        predictOnDataset(dataset, batchSize, callbacks) { batchCounter, tensors ->
            tensors.first().copyTo(buffer)
            buffer.forEachIndexed { index, data ->
                predictions[batchSize * batchCounter + index] = data.argmax()
            }
        }
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

    override fun predictSoftly(dataset: Dataset, batchSize: Int, callbacks: List<Callback>): Array<FloatArray> {
        require(dataset.xSize() % batchSize == 0) { "The number of elements in the dataset must be a multiple of batch size." }
        checkModelInitialized()

        val predictions = Array(dataset.xSize()) { FloatArray(numberOfClasses.toInt()) { 0.0f } }
        val buffer = Array(batchSize) { FloatArray(numberOfClasses.toInt()) { 0.0f } }
        predictOnDataset(dataset, batchSize, callbacks) { batchCounter, tensors ->
            tensors.first().copyTo(buffer)
            buffer.copyInto(predictions, batchSize * batchCounter)
        }
        return predictions
    }

    private fun predictOnDataset(dataset: Dataset,
                                 batchSize: Int,
                                 callbacks: List<Callback>,
                                 block: (Int, List<Tensor<*>>) -> Unit
    ) {
        callbacks.forEach { it.model = this }
        callbacks.forEach { it.onPredictBegin() }

        val xShape = calculateXShape(batchSize)

        for ((batchCounter, batch) in dataset.batchSequence(batchSize).withIndex()) {
            callbacks.forEach { it.onPredictBatchBegin(batchCounter, batchSize) }

            Tensor.create(xShape, serializeToBuffer(batch.x)).use { xTensor ->
                Tensor.create(false).use { isTraining ->
                    session.runner()
                        .fetch(predictionOp)
                        .feed(xOp.asOutput(), xTensor)
                        .feed(training.asOutput(), isTraining)
                        .run().use { tensors ->
                            block(batchCounter, tensors)
                        }
                }
            }

            callbacks.forEach { it.onPredictBatchEnd(batchCounter, batchSize) }
        }
        callbacks.forEach { it.onPredictEnd() }
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
        checkModelInitialized()

        val xShape = calculateXShape(1)

        return Tensor.create(xShape, FloatBuffer.wrap(inputData))
            .use { xTensor ->
                val runner = session.runner().feed(xOp.asOutput(), xTensor)
                if (predictionTensorName.isEmpty()) {
                    runner.fetch(predictionOp)
                } else {
                    require(kGraph().tfGraph.operation(predictionTensorName) != null) {
                        "Output named '$predictionTensorName' not found in the TensorFlow graph."
                    }
                    runner.fetch(predictionTensorName)
                }
                if (visualizationIsEnabled) {
                    for (layer in layers.dropLast(1)) {
                        if (layer.hasActivation) runner.fetch(defaultActivationName(layer))
                    }
                }
                runner.run().use { tensors ->
                    val prediction = tensors.first().convertTensorToFlattenFloatArray()
                    val activations = tensors.drop(1).map { it.convertTensorToMultiDimArray() }

                    tensors.forEach { it.close() }
                    prediction to activations
                }
            }
    }

    private fun calculateXYShapes(batch: DataBatch): Pair<LongArray, LongArray> {
        val batchSize = batch.size

        val xBatchShape = calculateXShape(batchSize)
        val yBatchShape = calculateYShape(batchSize)

        if (batchSize > 0) {
            validateBatchShape(batch, xBatchShape, yBatchShape)
        }

        return Pair(xBatchShape, yBatchShape)
    }

    private fun calculateYShape(batchSize: Int) = longArrayOf(batchSize.toLong(), numberOfClasses)

    private fun validateBatchShape(batch: DataBatch, xBatchShape: LongArray, yBatchShape: LongArray) {
        check(TensorShape(xBatchShape).numElements().toInt() == batch.x.size * batch.x[0].size)
        {
            "The calculated [from the Model] data batch shape ${xBatchShape.contentToString()} doesn't match actual data buffer size ${
                batch.x.size * batch.x[0].size
            }. Please, check input data."
        }
        check(TensorShape(yBatchShape).numElements().toInt() == batch.y.size * numberOfClasses.toInt())
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
        return longArrayOf(batchSize.toLong(), *xTensorShape.tail())
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
        checkModelInitialized()
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
            SavingFormat.TfGraphCustomVariables -> saveInSimpleFormat(pathToModelDirectory, saveOptimizerState)
            SavingFormat.TfGraph -> saveInSavedModelFormat(pathToModelDirectory)
            is SavingFormat.JsonConfigCustomVariables -> saveInKerasFormat(
                pathToModelDirectory,
                saveOptimizerState,
                savingFormat.isKerasFullyCompatible
            )
        }
    }

    private fun saveInKerasFormat(pathToModelDirectory: String,
                                  saveOptimizerState: Boolean,
                                  isKerasFullyCompatible: Boolean
    ) {
        saveModel(pathToModelDirectory, isKerasFullyCompatible)
        saveVariables(pathToModelDirectory, saveOptimizerState)
    }

    private fun saveModel(pathToModelDirectory: String, isKerasFullyCompatible: Boolean) {
        val jsonConfig = File("$pathToModelDirectory/$MODEL_CONFIG_JSON")
        saveModelConfiguration(jsonConfig, isKerasFullyCompatible)
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
        val variablesAndTensors = getVariablesAndTensors(saveOptimizerState)

        Files.createDirectories(Paths.get(pathToModelDirectory))
        val file = File("$pathToModelDirectory/variableNames.txt")

        file.bufferedWriter().use { variableNamesFile ->
            for ((variable, tensorForCopying) in variablesAndTensors) {
                val variableName = variable.asOutput().op().name()
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

    /** Returns a list of variables paired with their data. */
    private fun getVariablesAndTensors(saveOptimizerState: Boolean): List<Pair<Variable<Float>, Tensor<*>>> {
        var variables = layerVariables().map { it.variable }
        if (saveOptimizerState) {
            variables = variables + kGraph.optimizerVariables()
        }

        val modelWeightsExtractorRunner = session.runner()
        variables.forEach(modelWeightsExtractorRunner::fetch)
        return variables.zip(modelWeightsExtractorRunner.run())
    }

    override fun loadWeights(modelDirectory: File, loadOptimizerState: Boolean) {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(!isModelInitialized) { "The model is initialized already." }

        Files.createDirectories(modelDirectory.toPath())
        loadVariablesFromTxt(modelDirectory.path) { variableName ->
            if (!isOptimizerVariable(variableName)) true
            else if (loadOptimizerState) !isVariableRelatedToFrozenLayer(variableName)
            else false
        }

        isModelInitialized = true
        if (loadOptimizerState) isOptimizerVariableInitialized = true
    }

    /** Check that the variable with the name [variableName] belongs to the frozen layer. */
    private fun isVariableRelatedToFrozenLayer(variableName: String): Boolean {
        return frozenLayerVariables().map { it.name }.any { variableName.contains(it) }
    }

    /**
     * Loads variable data for variable names in the provided collection using a provided function.
     * @param [variableNames] Variable names to load.
     * @param [getData] Function that returns variable data by variable name and shape.
     */
    protected override fun loadVariables(variableNames: Collection<String>, getData: (String, Shape) -> Any) {
        val layerVariablesByName = layerVariables().associateBy { it.name }

        for (variableName in variableNames) {
            val variableOperation = kGraph.tfGraph.operation(variableName)
            check(variableOperation != null) { "Operation $variableName is not found in static graph." }
            val variableShape = variableOperation.output<Float>(0).shape()

            val data = getData(variableName, variableShape)

            val variable = layerVariablesByName[variableName]
            if (variable != null) {
                fill(variable, data)
            } else {
                assignVariable(variableName, variableShape, data)
            }
        }
    }

    internal fun fill(variable: KVariable, data: Any) {
        variable.initializerOperation.fill(data, session)
    }

    internal fun init(variable: KVariable) {
        variable.initializerOperation.run(session)
    }

    protected fun copyWeightsTo(model: GraphTrainableModel, copyOptimizerState: Boolean) {
        // TODO: make deep copies, not just links
        model.compile(
            optimizer = this.optimizer,
            loss = this.loss,
            metrics = this.metrics
        )

        model.layers.forEach {
            it.weights = this.getLayer(it.name).weights
        }

        if (copyOptimizerState) {
            val optimizerVariables = kGraph.variableNames().filter(::isOptimizerVariable)
            copyVariablesToModel(model, optimizerVariables)
            model.isOptimizerVariableInitialized = true
        }

        model.isModelInitialized = true
    }

    /**
     * Return layer by [layerName].
     *
     * @param [layerName] Should be existing layer name. Throws an error otherwise.
     */
    public infix fun getLayer(layerName: String): Layer {
        return layersByName[layerName] ?: error("No such layer $layerName in the model.")
    }

    private fun checkModelInitialized() {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method." }
    }

    override fun toString(): String {
        return "GraphTrainableModel(numberOfLayers=${layers.size}) ${super.toString()}"
    }

    public override fun summary(): TfModelSummary {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }

        val (trainableLayers, frozenLayers) = layers.partition { it.isTrainable }

        return TfModelSummary(
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

    override fun close() {
        session.close()
        kGraph.close()
    }

    /** Helper method for preprocessing layer names and layer validation. */
    internal companion object {
        internal const val MODEL_CONFIG_JSON = "modelConfig.json"

        internal fun preProcessLayerNames(layers: Array<out Layer>) {
            for ((index, layer) in layers.withIndex()) {
                if (layer.name.isEmpty()) {
                    val simpleName = layer::class.simpleName ?: "layer"
                    layer.name = simpleName.lowercase(Locale.getDefault()) + "_" + (index + 1)
                }
            }
        }

        internal fun layerValidation(layers: List<Layer>) {
            require(layers.isNotEmpty()) { "Model should contain layers!" }
            val input = layers[0]
            require(input is Input) { "Model should start from the Input layer" }
        }
    }
}

/**
 * Freezes weights in all layers in this model, so they won't be changed during training.
 * @see [Layer.freeze]
 */
public fun GraphTrainableModel.freeze() {
    layers.forEach(Layer::freeze)
}

private fun Dataset.batchSequence(size: Int) = batchIterator(size).asSequence()
