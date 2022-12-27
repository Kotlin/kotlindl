/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.savedmodel

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.util.serializeToBuffer
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.impl.util.use
import org.tensorflow.SavedModelBundle
import org.tensorflow.Tensor

/**
 * Inference model built on SavedModelBundle format to predict on images.
 */
public open class SavedModel : TensorFlowInferenceModel() {
    /** SavedModelBundle.*/
    private lateinit var bundle: SavedModelBundle

    override fun predict(inputData: FloatArray): Int {
        require(isShapeInitialized) { "Data shape is missed!" }

        val preparedData = serializeToBuffer(inputData)
        val tensor = Tensor.create(shape, preparedData)

        tensor.use {
            val runner = session.runner()
            return runner.feed(input.tfName, it)
                .fetch(output.tfName)
                .run().use { tensors ->
                    tensors.first().copyTo(LongArray(1))[0].toInt()
                }
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
        require(isShapeInitialized) { "Data shape is missed!" }

        val preparedData = serializeToBuffer(inputData)
        val tensor = Tensor.create(shape, preparedData)

        tensor.use {
            val runner = session.runner()
            return runner.feed(inputTensorName, it)
                .fetch(outputTensorName)
                .run().use { tensors ->
                    tensors.first().copyTo(LongArray(1))[0].toInt()
                }
        }
    }

    /**
     * Predicts labels for all observation in [dataset].
     *
     * NOTE: Slow method, executed on client side, not in TensorFlow.
     *
     * @param [inputTensorName] The name of input tensor.
     * @param [outputTensorName] The name of output tensor.
     * @param [dataset] Dataset.
     */
    public fun predict(dataset: OnHeapDataset, inputTensorName: String, outputTensorName: String): List<Int> {
        val predictedLabels: MutableList<Int> = mutableListOf()

        for (i in 0 until dataset.xSize()) {
            val predictedLabel = predict(dataset.getX(i), inputTensorName, outputTensorName)
            predictedLabels.add(predictedLabel)
        }

        return predictedLabels
    }

    override fun close() {
        super.close()
        bundle.close()
    }

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
}
