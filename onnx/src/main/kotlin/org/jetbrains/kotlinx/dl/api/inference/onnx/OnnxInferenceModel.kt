/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.Dataset
import java.nio.FloatBuffer
import java.util.*


/**
 * Inference model built on SavedModelBundle format to predict on images.
 */
public open class OnnxInferenceModel : InferenceModel() {

    private lateinit var env: OrtEnvironment
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
        require(::shape.isInitialized) { "Reshape functions is missed! Define and set up the reshape function to transform initial data to the model input." }

        val preparedData = FloatBuffer.wrap(inputData)

        val tensor = OnnxTensor.createTensor(env, preparedData, shape)
        val output = session.run(Collections.singletonMap(session.inputNames.toList()[0], tensor))

        val outputProbs = output[0].value as Array<FloatArray>
        val predLabel = pred(outputProbs[0])
        return predLabel
    }

    override fun predictSoftly(inputData: FloatArray, predictionTensorName: String): FloatArray {
        TODO("Not yet implemented")
    }

    public fun rawPredict(inputData: FloatArray): Any {
        require(::shape.isInitialized) { "Reshape functions is missed! Define and set up the reshape function to transform initial data to the model input." }

        val preparedData = FloatBuffer.wrap(inputData)

        val tensor = OnnxTensor.createTensor(env, preparedData, shape)
        val output = session.run(Collections.singletonMap(session.inputNames.toList()[0], tensor))

        val result = mutableListOf<Array<*>>()

        output.forEach {
            println("Shape of output ${(it.value.info as TensorInfo).shape.contentToString()}")
            result.add(it.value.value as Array<*>)
        }

        return result
        // ((((output as Result).list as java.util.ArrayList<*>)[0] as OnnxTensor).info as TensorInfo).shape = [1, 7, 7, 2048]
    }


    /**
     * Find the maximum probability and return it's index.
     *
     * @param probabilities The probabilites.
     * @return The index of the max.
     */
    private fun pred(probabilities: FloatArray): Int {
        var maxVal = Float.NEGATIVE_INFINITY
        var idx = 0
        for (i in probabilities.indices) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i]
                idx = i
            }
        }
        return idx
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
        return 0
    }

    /**
     * Predicts labels for all [images].
     *
     * NOTE: Slow method, executed on client side, not in TensorFlow.
     *
     * @param [dataset] Dataset.
     */
    public fun predictAll(dataset: Dataset): List<Int> {
        val predictedLabels: MutableList<Int> = mutableListOf()

        for (i in 0 until dataset.xSize()) {
            val predictedLabel = predict(dataset.getX(i))
            predictedLabels.add(predictedLabel)
        }

        return predictedLabels
    }

    /**
     * Predicts labels for all [images].
     *
     * NOTE: Slow method, executed on client side, not in TensorFlow.
     *
     * @param [inputTensorName] The name of input tensor.
     * @param [outputTensorName] The name of output tensor.
     * @param [dataset] Dataset.
     */
    public fun predictAll(dataset: Dataset, inputTensorName: String, outputTensorName: String): List<Int> {
        val predictedLabels: MutableList<Int> = mutableListOf()

        for (i in 0 until dataset.xSize()) {
            val predictedLabel = predict(dataset.getX(i), inputTensorName, outputTensorName)
            predictedLabels.add(predictedLabel)
        }

        return predictedLabels
    }

    /**
     * Evaluates [dataset] via [metric].
     *
     * NOTE: Slow method, executed on client side, not in TensorFlow.
     */
    public fun evaluate(
        dataset: Dataset,
        metric: Metrics
    ): Double {
        return if (metric == Metrics.ACCURACY) {
            var counter = 0
            for (i in 0 until dataset.xSize()) {
                val predictedLabel = predict(dataset.getX(i))
                if (predictedLabel == dataset.getY(i).toInt())
                    counter++
            }

            (counter.toDouble() / dataset.xSize())
        } else {
            Double.NaN
        }
    }


    override fun close() {
        session.close()
    }


    override fun toString(): String {
        println(session.inputNames)
        println(session.inputInfo)
        println(session.outputNames)
        println(session.outputInfo)
        return "OnnxModel(session=$session)"
    }
}
