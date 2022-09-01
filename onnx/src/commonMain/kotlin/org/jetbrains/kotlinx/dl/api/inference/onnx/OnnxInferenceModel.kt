/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx

import ai.onnxruntime.*
import ai.onnxruntime.OrtSession.SessionOptions
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.extension.argmax
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.CPU
import java.nio.*
import java.util.*


private const val RESHAPE_MISSED_MESSAGE = "Model input shape is not defined. Call reshape() to set input shape."

/**
 * Inference model built on ONNX format.
 *
 * @since 0.3
 */
public open class OnnxInferenceModel private constructor(private val modelSource: ModelSource) : InferenceModel {
    /**
     * The host object for the onnx-runtime system. Can create [session] which encapsulate
     * specific models.
     */
    private val env = OrtEnvironment.getEnvironment()

    /** Wraps an ONNX model and allows inference calls. */
    private lateinit var session: OrtSession

    /** Data shape for prediction. */
    private lateinit var inputShape: LongArray

    /** Data type for input tensor. */
    public lateinit var inputDataType: OnnxJavaType
        private set

    /** Data shape for prediction. */
    public lateinit var outputShape: LongArray
        private set

    /** Data type for output tensor. */
    public lateinit var outputDataType: OnnxJavaType
        private set

    /** Execution providers currently set for the model. */
    private lateinit var executionProvidersInUse: List<ExecutionProvider>

    /** Model name. */
    public var name: String? = null

    /**
     * Constructs an ONNX inference model from the given model file.
     */
    public constructor(modelPath: String) : this(ModelSource.File(modelPath))

    /**
     * Constructs an ONNX inference model from the byte array representing an ONNX model.
     */
    public constructor(modelBytes: ByteArray) : this(ModelSource.Bytes(modelBytes))

    /**
     * This is an interface representing a possible source for loading ONNX models.
     */
    private sealed interface ModelSource {
        /**
         * Method for building an [OrtSession] from the model source.
         */
        fun buildSession(environment: OrtEnvironment, options: SessionOptions): OrtSession

        /**
         * ONNX serialized file.
         */
        class File(private val pathToModel: String) : ModelSource {
            override fun buildSession(environment: OrtEnvironment, options: SessionOptions): OrtSession {
                return environment.createSession(pathToModel, options)
            }
        }

        /**
         * Byte array representing an ONNX model.
         */
        class Bytes(private val bytes: ByteArray) : ModelSource {
            override fun buildSession(environment: OrtEnvironment, options: SessionOptions): OrtSession {
                return environment.createSession(bytes, options)
            }
        }
    }

    /**
     * Initializes the model, if it's not initialized, or re-initializes it, depending on the execution providers.
     *
     * By default, the model is initialized with CPU execution provider with BFCArena memory allocator.
     * This method allows to set the execution provider to use.
     * If the model is already initialized, internal session will be closed and new one will be created.
     * If [executionProvidersInUse] is the same as the one passed, nothing will happen.
     * If execution provider is not supported, an exception will be thrown.
     * If empty list is passed, the model will be initialized with CPU execution provider.
     *
     * @param executionProviders list of execution providers to use.
     */
    public fun initializeWith(vararg executionProviders: ExecutionProvider = arrayOf(CPU(true))) {
        val uniqueProviders = collectProviders(executionProviders)

        if (::executionProvidersInUse.isInitialized && uniqueProviders == executionProvidersInUse) {
            return
        }

        if (::session.isInitialized) {
            session.close()
        }

        session = modelSource.buildSession(env, buildSessionOptions(uniqueProviders))

        executionProvidersInUse = uniqueProviders

        initInputOutputInfo()
    }

    private fun initInputOutputInfo() {
        val inputTensorInfo = session.inputInfo.toList()[0].second.info as TensorInfo
        if (!::inputShape.isInitialized) {
            val inputDims =
                inputTensorInfo.shape.takeLast(3).toLongArray()
            inputShape = TensorShape(1, *inputDims).dims()
        }
        inputDataType = inputTensorInfo.type

        // TODO: known bug at the https://github.com/JetBrains/KotlinDL/issues/285
        val outputTensorInfo = session.outputInfo.toList()[0].second.info as TensorInfo
        if (!::outputShape.isInitialized) {
            val outputDims = outputTensorInfo.shape.takeLast(3).toLongArray()
            // TODO: is it obsolete? anyway we should add support of multiple outputs
            outputShape = TensorShape(1, *outputDims).dims()
        }
        outputDataType = outputTensorInfo.type
    }

    private fun buildSessionOptions(uniqueProviders: List<ExecutionProvider>): SessionOptions {
        val sessionOptions = SessionOptions()
        for (provider in uniqueProviders) {
            provider.addOptionsTo(sessionOptions)
        }
        return sessionOptions
    }

    private fun collectProviders(executionProviders: Array<out ExecutionProvider>): List<ExecutionProvider> {
        for (executionProvider in executionProviders) {
            require(executionProvider.internalProviderId in OrtEnvironment.getAvailableProviders()) {
                "The optimized execution provider $executionProvider is not available in the current environment!"
            }
        }

        val uniqueProviders = executionProviders.distinct().toMutableList()

        /*
            We ensure that the CPU execution provider is always last in the list.
         */
        when (uniqueProviders.count { it is CPU }) {
            /*
                Users can explicitly add CPU provider with BFCArenaAllocator disabled,
                but if not present, it will be added automatically with BFCArenaAllocator enabled.
             */
            0 -> {
                uniqueProviders.add(CPU(true))
            }
            1 -> {
                val cpu = uniqueProviders.first { it is CPU }
                uniqueProviders.remove(cpu)
                uniqueProviders.add(cpu)
            }
            else -> throw IllegalArgumentException("Unable to use CPU(useArena = true) and CPU(useArena = false) at the same time!")
        }
        return uniqueProviders
    }

    /**
     * Chain-like setter to set up input shape.
     *
     * @param dims The input shape.
     */
    public override fun reshape(vararg dims: Long) {
        inputShape = longArrayOf(1, *dims)
    }

    override val inputDimensions: LongArray
        get() = TensorShape(inputShape).tail()

    public override fun predict(inputData: FloatArray): Int {
        return predictSoftly(inputData).argmax()
    }

    override fun predictSoftly(inputData: FloatArray, predictionTensorName: String): FloatArray {
        require(::inputShape.isInitialized) { RESHAPE_MISSED_MESSAGE }

        val outputTensorName = when {
            predictionTensorName.isEmpty() -> session.outputNames.first()
            else -> predictionTensorName
        }

        require(outputTensorName in session.outputInfo) { "There is no output with name '$outputTensorName'. The model only has following outputs - ${session.outputInfo.keys}" }

        throwIfOutputNotSupported(outputTensorName, "predictSoftly")

        val outputIdx = session.outputInfo.keys.indexOf(outputTensorName)

        return predictSoftly(inputData, outputIdx)
    }

    /**
     * Currently, some methods only support float tensors as model output.
     * This method checks if model output satisfies these requirements.
     */
    // TODO: add support for all ONNX output types (see https://github.com/Kotlin/kotlindl/issues/367)
    private fun throwIfOutputNotSupported(outputName: String, method: String) {
        val outputInfo = session.outputInfo[outputName]!!.info
        require(outputInfo !is MapInfo) { "Output $outputName is a Map, but currently method $method supports only float Tensor outputs. Please use predictRaw method instead." }
        require(outputInfo !is SequenceInfo) { "Output '$outputName' is a Sequence, but currently method $method supports only float Tensor outputs. Please use predictRaw method instead." }
        require(outputInfo is TensorInfo && outputInfo.type == OnnxJavaType.FLOAT) { "Currently method $method supports only float Tensor outputs, but output '$outputName' is not a float Tensor. Please use predictRaw method instead." }
    }

    /**
     * Predicts vector of probabilities instead of specific class in [predict] method.
     *
     * @param [inputData] The single example with unknown vector of probabilities.
     * @return Vector that represents the probability distributions of a list of potential outcomes
     */
    public fun predictSoftly(inputData: FloatArray): FloatArray {
        return predictSoftly(inputData, session.outputNames.first())
    }

    private fun predictSoftly(inputData: FloatArray, outputTensorIdx: Int): FloatArray {
        val inputTensor = createInputTensor(inputData)

        val outputTensor = session.run(Collections.singletonMap(session.inputNames.toList()[0], inputTensor))

        val outputInfo = session.outputInfo.toList()[outputTensorIdx].second.info as TensorInfo

        val outputProbs: FloatArray = when {
            outputInfo.shape.size > 1 -> (outputTensor[outputTensorIdx].value as Array<FloatArray>)[0]
            else -> outputTensor[outputTensorIdx].value as FloatArray
        }

        outputTensor.close()
        inputTensor.close()

        return outputProbs
    }

    /**
     * Returns list of multidimensional arrays with data from model outputs.
     *
     * NOTE: This operation can be quite slow for high dimensional tensors,
     * you should prefer [predictRawWithShapes] in this case.
     */
    public fun predictRaw(inputData: FloatArray): Map<String, Any> {
        require(::inputShape.isInitialized) { RESHAPE_MISSED_MESSAGE }

        val inputTensor = createInputTensor(inputData)

        val outputTensor = session.run(Collections.singletonMap(session.inputNames.toList()[0], inputTensor))

        val result = mutableMapOf<String, Any>()

        outputTensor.forEach {
            result[it.key] = it.value.value
        }

        outputTensor.close()
        inputTensor.close()

        return result.toMap()
    }

    // TODO: refactor predictRaw and predictRawWithShapes to extract the common functionality

    /**
     *  Returns list of pairs <data; shape> from model outputs.
     */
    // TODO: add tests for many available models
    // TODO: return map
    public fun predictRawWithShapes(inputData: FloatArray): List<Pair<FloatBuffer, LongArray>> {
        require(::inputShape.isInitialized) { RESHAPE_MISSED_MESSAGE }

        session.outputInfo.keys.forEach {
            throwIfOutputNotSupported(it, "predictRawWithShapes")
        }

        val preparedData = FloatBuffer.wrap(inputData)

        val tensor = OnnxTensor.createTensor(env, preparedData, inputShape)
        val output = session.run(Collections.singletonMap(session.inputNames.toList()[0], tensor))

        val result = mutableListOf<Pair<FloatBuffer, LongArray>>()

        output.forEach {
            val onnxTensorShape = (it.value.info as TensorInfo).shape
            result.add(Pair((it.value as OnnxTensor).floatBuffer, onnxTensorShape))
        }

        output.close()
        tensor.close()

        return result.toList()
    }

    /**
     * Predicts the class of [inputData].
     *
     * @param [inputData] The single example with unknown label.
     * @param [inputTensorName] The name of input tensor.
     * @param [outputTensorName] The name of output tensor.
     * @return Predicted class index.
     */
    public fun predict(inputData: FloatArray, inputTensorName: String, outputTensorName: String): Int {
        TODO("ONNX doesn't support extraction outputs from the intermediate levels of the model.")
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): OnnxInferenceModel {
        val model = OnnxInferenceModel(modelSource)
        model.name = copiedModelName
        if (::inputShape.isInitialized) {
            model.reshape(*inputDimensions)
        }
        if (::session.isInitialized) {
            model.initializeWith(*executionProvidersInUse.toTypedArray())
        }
        return model
    }

    /** Releases the ONNXRuntime - related resources. */
    override fun close() {
        session.close()
        env.close()
    }

    // TODO: make ONNX model description (see https://github.com/Kotlin/kotlindl/issues/368)
    override fun toString(): String {
        println(session.inputNames)
        println(session.inputInfo)
        println(session.outputNames)
        println(session.outputInfo)
        return "OnnxModel(session=$session)"
    }

    private fun createInputTensor(inputData: FloatArray): OnnxTensor {
        val inputTensor = when (inputDataType) {
            OnnxJavaType.FLOAT -> OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), inputShape)
            OnnxJavaType.DOUBLE -> OnnxTensor.createTensor(
                env,
                DoubleBuffer.wrap(inputData.map { it.toDouble() }.toDoubleArray()),
                inputShape
            )

            OnnxJavaType.INT8 -> OnnxTensor.createTensor(
                env,
                ByteBuffer.wrap(inputData.map { it.toInt().toByte() }.toByteArray()),
                inputShape
            )

            OnnxJavaType.INT16 -> OnnxTensor.createTensor(
                env,
                ShortBuffer.wrap(inputData.map { it.toInt().toShort() }.toShortArray()),
                inputShape
            )

            OnnxJavaType.INT32 -> OnnxTensor.createTensor(
                env,
                IntBuffer.wrap(inputData.map { it.toInt() }.toIntArray()),
                inputShape
            )

            OnnxJavaType.INT64 -> OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(inputData.map { it.toLong() }.toLongArray()),
                inputShape
            )

            OnnxJavaType.STRING -> TODO()
            OnnxJavaType.UINT8 -> OnnxTensor.createTensor(
                env,
                ByteBuffer.wrap(inputData.map { it.toInt().toUByte().toByte() }.toByteArray()),
                inputShape,
                OnnxJavaType.UINT8
            )

            OnnxJavaType.UNKNOWN -> TODO()
            else -> TODO()
        }
        return inputTensor
    }

    public companion object {
        /**
         * Loads model from serialized ONNX file.
         */
        public fun load(
            pathToModel: String,
            vararg executionProviders: ExecutionProvider = arrayOf(CPU(true))
        ): OnnxInferenceModel {
            val model = OnnxInferenceModel(pathToModel)
            model.initializeWith(*executionProviders)
            return model
        }

        /**
         * Loads model from a byte array representing an ONNX model.
         */
        public fun load(
            modelBytes: ByteArray,
            vararg executionProviders: ExecutionProvider = arrayOf(CPU(true))
        ): OnnxInferenceModel {
            val model = OnnxInferenceModel(modelBytes)
            model.initializeWith(*executionProviders)
            return model
        }
    }
}
