/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.savedmodel

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.tensorflow.SavedModelBundle
import org.tensorflow.Tensor

/**
 * Inference model built on SavedModelBundle format to predict on images.
 */
public open class SavedModel : InferenceModel() {
    /** SavedModelBundle.*/
    private lateinit var bundle: SavedModelBundle

    public companion object {
        /**
         * Loads model from SavedModelBundle format.
         */
        public fun load(pathToModel: String): SavedModel {
            val model = SavedModel()

            model.bundle = SavedModelBundle.load(pathToModel, "serve")
            model.session = model.bundle.session()
            val graph = model.bundle.graph()
            val graphDef = graph.toGraphDef()
            model.kGraph = KGraph(graphDef)
            return model
        }
    }

    override fun predict(inputData: FloatArray): Int {
        require(reshapeFunction != null) { "Reshape functions is missed!" }

        val preparedData = reshapeFunction(inputData)
        val tensor = Tensor.create(preparedData)

        tensor.use {
            val runner = session.runner()
            return runner.feed(input.tfName, it)
                .fetch(output.tfName)
                .run()[0]
                .copyTo(LongArray(1))[0].toInt()
        }
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
        require(reshapeFunction != null) { "Reshape functions is missed!" }

        val preparedData = reshapeFunction(inputData)
        val tensor = Tensor.create(preparedData)

        tensor.use {
            val runner = session.runner()
            return runner.feed(inputTensorName, it)
                .fetch(outputTensorName)
                .run()[0]
                .copyTo(LongArray(1))[0].toInt()
        }
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
        super.close()
        bundle.close()
    }
}

/*public fun prepareModelForInference(init: SavedModel.() -> Unit): SavedModel =
    SavedModel()
        .apply(init)*/
