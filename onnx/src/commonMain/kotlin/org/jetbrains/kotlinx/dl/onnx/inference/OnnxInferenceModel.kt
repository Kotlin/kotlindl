/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference

import ai.onnxruntime.*
import ai.onnxruntime.OrtSession.SessionOptions
import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.core.floats
import org.jetbrains.kotlinx.dl.api.core.shape
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape.Companion.tail
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.summary.ModelSummary
import org.jetbrains.kotlinx.dl.api.summary.ModelWithSummary
import org.jetbrains.kotlinx.dl.impl.util.argmax
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.getFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.getValues
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.throwIfOutputNotSupported
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.CPU
import java.nio.*


/**
 * Inference model built on ONNX format.
 *
 * @since 0.3
 */
public open class OnnxInferenceModel private constructor(
    private val modelSource: ModelSource
) : InferenceModel, ExecutionProviderCompatible, ModelWithSummary {
    /**
     * The host object for the onnx-runtime system. Can create [session] which encapsulate
     * specific models.
     */
    private val env = OrtEnvironment.getEnvironment()

    /** Wraps an ONNX model and allows inference calls. */
    private lateinit var session: OrtSession

    /** Model input information */
    private lateinit var inputInfo: Map<String, NodeInfo>

    /** Model output information */
    private lateinit var outputInfo: Map<String, NodeInfo>

    /** Execution providers currently set for the model. */
    private lateinit var executionProvidersInUse: List<ExecutionProvider>

    /** Data shape of the first input tensor. Set explicitly when the model can accept variable shape input. */
    private var inputShape: LongArray? = null

    /** Data shape of the first output tensor. */
    public val outputShape: LongArray get() = outputInfo.getShape(0)

    /** Model name. */
    public var name: String? = null

    /**
     * Constructs an ONNX inference model from the given model file.
     */
    public constructor(modelPath: String) : this(ModelSource.File(modelPath))

    /**
     * Constructs an ONNX inference model from the byte array representing an ONNX model.
     */
    public constructor(modelBytes: ByteArray) : this(ModelSource.Bytes { modelBytes })

    /**
     * Constructs an ONNX inference model from the function which returns a byte array representing an ONNX model.
     */
    public constructor(loadBytes: () -> ByteArray) : this(ModelSource.Bytes(loadBytes))

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
         * Loader for a byte array representing an ONNX model.
         */
        class Bytes(private val loadBytes: () -> ByteArray) : ModelSource {
            override fun buildSession(environment: OrtEnvironment, options: SessionOptions): OrtSession {
                return environment.createSession(loadBytes(), options)
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
    public override fun initializeWith(vararg executionProviders: ExecutionProvider) {
        val uniqueProviders = collectProviders(executionProviders)

        if (::executionProvidersInUse.isInitialized && uniqueProviders == executionProvidersInUse) {
            return
        }

        if (::session.isInitialized) {
            session.close()
        }

        session = modelSource.buildSession(env, buildSessionOptions(uniqueProviders))
        executionProvidersInUse = uniqueProviders
        inputInfo = session.inputInfo
        outputInfo = session.outputInfo
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
        get() = TensorShape(inputShape ?: inputInfo.getShape(0)).tail()

    public override fun predict(inputData: FloatArray): Int {
        return predictSoftly(inputData).argmax()
    }

    override fun predictSoftly(inputData: FloatArray, predictionTensorName: String): FloatArray {
        val outputTensorName = predictionTensorName.ifEmpty { outputInfo.getName(0) }
        require(outputTensorName in outputInfo) {
            "There is no output with name '$outputTensorName'." +
                    " The model only has following outputs - ${outputInfo.keys}"
        }

        val outputInfo = outputInfo.getValue(outputTensorName).info
        throwIfOutputNotSupported(outputInfo, outputTensorName, "predictSoftly", OnnxJavaType.FLOAT)

        val shape = (inputShape ?: inputInfo.getShape(0)).tail()
        val floatData = FloatData(inputData, TensorShape(shape))

        return predictRaw(floatData) { output -> output.getFloatArray(outputTensorName) }
    }

    /**
     * Predicts vector of probabilities instead of specific class in [predict] method.
     *
     * @param [inputData] The single example with unknown vector of probabilities.
     * @return Vector that represents the probability distributions of a list of potential outcomes
     */
    public fun predictSoftly(inputData: FloatData): FloatArray {
        return predictRaw(inputData) { output ->
            output.getFloatArray(outputInfo.getName(0))
        }
    }

    /**
     * Returns list of multidimensional arrays with data from model outputs.
     *
     * NOTE: This operation can be quite slow for high dimensional tensors,
     * use [predictRaw] with custom output processing for better performance.
     */
    public fun predictRaw(inputData: FloatData): Map<String, Any> {
        return predictRaw(inputData) { it.getValues() }
    }

    /**
     * Runs prediction on a given [inputData] and calls [extractResult] function to process output.
     * For models with multiple inputs, [inputData] is passed as a first input.
     * @see OrtSessionResultConversions
     */
    public fun <R> predictRaw(inputData: FloatData, extractResult: (OrtSession.Result) -> R): R {
        return predictRaw(mapOf(inputInfo.getName(0) to inputData), extractResult)
    }

    /**
     * Runs prediction on a given [inputs] and calls [extractResult] function to process output.
     * @see OrtSessionResultConversions
     */
    public fun <R> predictRaw(inputs: Map<String, FloatData>, extractResult: (OrtSession.Result) -> R): R {
        val tensors = inputs.toTensors()
        try {
            return session.run(tensors).use { output -> extractResult(output) }
        } finally {
            tensors.values.forEach(OnnxTensor::close)
        }
    }

    private fun Map<String, FloatData>.toTensors(): Map<String, OnnxTensor> {
        // do the checks separately to avoid exceptions during tensor creation
        val triples = mapValues { (name, floatData) ->
            require(name in inputInfo) {
                "There is no input with name '$name'." +
                        " The model only has following inputs: ${inputInfo.keys}"
            }

            val modelShape = inputInfo.getShape(name)
            val dataShape = longArrayOf(1L, *floatData.shape.dims())
            require(modelShape.matches(dataShape)) {
                "Data shape for input $name does not match the model input shape. Data shape: $dataShape, model shape: $modelShape"
            }
            require(dataShape.all { it >= 0 }) { "Data shape is not defined: $dataShape" }

            Triple(floatData.floats, dataShape, inputInfo.getType(name))
        }

        return triples.mapValues { (_, triple) ->
            val (data, shape, type) = triple
            env.createTensor(data, type, shape)
        }
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): OnnxInferenceModel {
        val model = OnnxInferenceModel(modelSource)
        model.name = copiedModelName
        if (inputShape != null) {
            model.reshape(*inputDimensions)
        }
        if (::session.isInitialized) {
            model.initializeWith(*executionProvidersInUse.toTypedArray())
        }
        return model
    }

    /** Releases the ONNXRuntime - related resources. */
    override fun close() {
        if (::session.isInitialized) {
            session.close()
        }
        env.close()
    }

    override fun summary(): ModelSummary {
        val inputSummaries = inputInfo.mapValues { (_, node) -> node.info.summary() }.toList()
        val outputSummaries = outputInfo.mapValues { (_, node) -> node.info.summary() }.toList()
        return OnnxModelSummary(inputSummaries, outputSummaries)
    }

    override fun toString(): String {
        return "OnnxModel(session=$session)"
    }

    public companion object {
        private fun Map<String, NodeInfo>.getName(index: Int): String {
            return asIterable().elementAt(index).key
        }

        private fun Map<String, NodeInfo>.getShape(index: Int): LongArray {
            val (name, nodeInfo) = asIterable().elementAt(index)
            return nodeInfo.getShape(name)
        }

        private fun Map<String, NodeInfo>.getShape(name: String): LongArray {
            return getValue(name).getShape(name)
        }

        private fun NodeInfo.getShape(name: String): LongArray {
            throwIfOutputNotSupported(info, name, "getShape")
            val shape = (info as TensorInfo).shape
            if (shape[0] >= 0) return shape
            return (listOf(1L) + shape.drop(1)).toLongArray() // use batch size 1
        }

        private fun Map<String, NodeInfo>.getType(name: String): OnnxJavaType {
            val nodeInfo = getValue(name)
            throwIfOutputNotSupported(nodeInfo.info, name, "getType")
            return (nodeInfo.info as TensorInfo).type
        }

        private fun LongArray.matches(other: LongArray): Boolean {
            if (size != other.size) return false
            for (i in indices) {
                val value = get(i)
                val otherValue = other[i]
                if (value < 0 || otherValue < 0) continue
                if (value != otherValue) return false
            }
            return true
        }

        private fun OrtEnvironment.createTensor(
            data: FloatArray,
            dataType: OnnxJavaType,
            shape: LongArray
        ): OnnxTensor {
            checkTensorMatchesInputShape(data, shape)

            val inputTensor = when (dataType) {
                OnnxJavaType.FLOAT -> OnnxTensor.createTensor(this, FloatBuffer.wrap(data), shape)
                OnnxJavaType.DOUBLE -> OnnxTensor.createTensor(
                    this,
                    DoubleBuffer.wrap(data.map { it.toDouble() }.toDoubleArray()),
                    shape
                )

                OnnxJavaType.INT8 -> OnnxTensor.createTensor(
                    this,
                    ByteBuffer.wrap(data.map { it.toInt().toByte() }.toByteArray()),
                    shape
                )

                OnnxJavaType.INT16 -> OnnxTensor.createTensor(
                    this,
                    ShortBuffer.wrap(data.map { it.toInt().toShort() }.toShortArray()),
                    shape
                )

                OnnxJavaType.INT32 -> OnnxTensor.createTensor(
                    this,
                    IntBuffer.wrap(data.map { it.toInt() }.toIntArray()),
                    shape
                )

                OnnxJavaType.INT64 -> OnnxTensor.createTensor(
                    this,
                    LongBuffer.wrap(data.map { it.toLong() }.toLongArray()),
                    shape
                )

                OnnxJavaType.STRING -> TODO()
                OnnxJavaType.UINT8 -> OnnxTensor.createTensor(
                    this,
                    ByteBuffer.wrap(data.map { it.toInt().toUByte().toByte() }.toByteArray()),
                    shape,
                    OnnxJavaType.UINT8
                )

                OnnxJavaType.UNKNOWN -> TODO()
                else -> TODO()
            }
            return inputTensor
        }

        private fun checkTensorMatchesInputShape(data: FloatArray, inputShape: LongArray) {
            val numOfElements = inputShape.reduce { acc, dim -> acc * dim }.toInt()

            if (data.size == numOfElements) return

            if (inputShape.size == 4 &&
                inputShape[0] == 1L &&
                (inputShape[1] == 3L || inputShape[3] == 3L) &&
                (data.size * 3 == numOfElements)
            ) {
                throw IllegalArgumentException(
                    "The number of elements (N=${data.size}) in the input tensor does not match the model input shape - "
                        .plus("${inputShape.contentToString()}.")
                        .plus(" It looks like you are trying to use a 1-channel (grayscale) image as an input, but the model expects a 3-channel image.")
                )
            }

            throw IllegalArgumentException(
                "The number of elements (N=${data.size}) in the input tensor does not match the model input shape - "
                    .plus("${inputShape.contentToString()}.")
            )
        }

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
