/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import mu.KLogger
import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.extension.argmax
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import java.nio.FloatBuffer
import java.util.*


/**
 * Inference model built on ONNX format.
 */
public open class OnnxInferenceModel : InferenceModel() {
    /** Logger for the model. */
    private val logger: KLogger = KotlinLogging.logger {}

    /**
     * The host object for the onnx-runtime system. Can create [session] which encapsulate
     * specific models.
     */
    private lateinit var env: OrtEnvironment

    /** Wraps an ONNX model and allows inference calls. */
    private lateinit var session: OrtSession

    /** Data shape for prediction. */
    public lateinit var shape: LongArray
        private set

    public companion object {
        /**
         * Loads model from SavedModelBundle format.
         */
        public fun load(pathToModel: String): OnnxInferenceModel {
            val model = OnnxInferenceModel()

            model.env = OrtEnvironment.getEnvironment()
            model.session = model.env.createSession(pathToModel, OrtSession.SessionOptions())

            return model
        }
    }

    /**
     * Chain-like setter to set up input shape.
     *
     * @param dims The input shape.
     */
    public override fun reshape(vararg dims: Long) {
        this.shape = TensorShape(1, *dims).dims()
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): TensorFlowInferenceModel {
        TODO("Not yet implemented")
    }

    public override fun predict(inputData: FloatArray): Int {
        return predictSoftly(inputData).argmax()
    }

    override fun predictSoftly(inputData: FloatArray, predictionTensorName: String): FloatArray {
        return predictSoftly(inputData)
    }

    public fun predictSoftly(inputData: FloatArray): FloatArray {
        require(::shape.isInitialized) { "Reshape functions is missed! Define and set up the reshape function to transform initial data to the model input." }

        val preparedData = FloatBuffer.wrap(inputData)

        val tensor = OnnxTensor.createTensor(env, preparedData, shape)
        val output = session.run(Collections.singletonMap(session.inputNames.toList()[0], tensor))

        val outputProbs = output[0].value as Array<FloatArray>

        output.close()
        tensor.close()

        return outputProbs[0]
    }

    /**
     * Returns list of multidimensional arrays with data from model outputs.
     *
     * NOTE: This operation can be quite slow for high dimensional tensors,
     * you should prefer [predictRawWithShapes] in this case.
     */
    public fun predictRaw(inputData: FloatArray): List<Array<*>> {
        require(::shape.isInitialized) { "Reshape functions is missed! Define and set up the reshape function to transform initial data to the model input." }

        val preparedData = FloatBuffer.wrap(inputData)

        val tensor = OnnxTensor.createTensor(env, preparedData, shape)
        val output = session.run(Collections.singletonMap(session.inputNames.toList()[0], tensor))
        logger.debug { "Number of model's output tensors: ${output.size()}" }
        val result = mutableListOf<Array<*>>()

        output.forEach {
            logger.debug { "Output tensor: ${it.key}" }
            logger.debug { "Shape of the output: ${(it.value.info as TensorInfo).shape.contentToString()}" }
            result.add(it.value.value as Array<*>)
        }

        output.close()
        tensor.close()

        return result.toList()
    }

    // TODO: refactor predictRaw and predictRawWithShapes to extract the common functionality

    /**
     *  Returns list of pairs <data; shape> from model outputs.
     */
    public fun predictRawWithShapes(inputData: FloatArray): List<Pair<FloatBuffer, LongArray>> {
        require(::shape.isInitialized) { "Reshape functions is missed! Define and set up the reshape function to transform initial data to the model input." }

        val preparedData = FloatBuffer.wrap(inputData)

        val tensor = OnnxTensor.createTensor(env, preparedData, shape)
        val output = session.run(Collections.singletonMap(session.inputNames.toList()[0], tensor))
        logger.debug { "Number of model's output tensors: ${output.size()}" }
        val result = mutableListOf<Pair<FloatBuffer, LongArray>>()

        output.forEach {
            logger.debug { "Output tensor: ${it.key}" }
            val onnxTensorShape = (it.value.info as TensorInfo).shape
            logger.debug { "Shape of the output: ${onnxTensorShape.contentToString()}" }
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

    override fun close() {
        session.close()
        env.close()
    }

    // TODO: make ONNX model description
    override fun toString(): String {
        println(session.inputNames)
        println(session.inputInfo)
        println(session.outputNames)
        println(session.outputInfo)
        return "OnnxModel(session=$session)"
    }
}
