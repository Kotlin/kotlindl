/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.datasets.Dataset
import java.util.*


/**
 * Inference model built on SavedModelBundle format to predict on images.
 */
public open class OnnxModel : AutoCloseable {

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession

    public companion object {
        /**
         * Loads model from SavedModelBundle format.
         */
        public fun load(pathToModel: String): OnnxModel {
            val model = OnnxModel()

            model.env = OrtEnvironment.getEnvironment()
            model.session = model.env.createSession(pathToModel, OrtSession.SessionOptions())

            return model
        }
    }

    /** Function for conversion from flat float array to input tensor. */
    public lateinit var reshapeFunction: (FloatArray) -> Array<*>

    public fun predict(inputData: FloatArray): Int {
        val preparedData = reshapeFunction(inputData)
        val tensor = OnnxTensor.createTensor(env, preparedData)
        val output = session.run(Collections.singletonMap(session.inputNames.toList()[0], tensor))
        val outputProbs = output[0].value as Array<FloatArray>
        val predLabel = pred(outputProbs[0])
        return predLabel
    }

    public fun predict(inputData: FloatArray, output: String): Int {
        val preparedData = reshapeFunction(inputData)
        val tensor = OnnxTensor.createTensor(env, preparedData)
        val output = session.run(Collections.singletonMap(session.inputNames.toList()[0], tensor), setOf(output))
        val outputProbs = output[0].value as Array<FloatArray>
        val predLabel = pred(outputProbs[0])
        return predLabel
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
                if (predictedLabel == dataset.getLabel(i))
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

    /**
     * Chain-like setter to set up [reshapeFunction] function.
     *
     * @param reshapeFunction The approach to convert raw input data to multidimensional array of floats.
     */
    public fun reshape(reshapeFunction: (FloatArray) -> Array<*>) {
        this.reshapeFunction = reshapeFunction
    }

    override fun toString(): String {
        println(session.inputNames)
        println(session.inputInfo)
        println(session.outputNames)
        println(session.outputInfo)
        return "OnnxModel(session=$session)"
    }
}

