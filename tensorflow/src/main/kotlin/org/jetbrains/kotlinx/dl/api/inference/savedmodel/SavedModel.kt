/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.savedmodel

import org.jetbrains.kotlinx.dl.api.core.util.PLACEHOLDER
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.impl.util.use
import org.tensorflow.SavedModelBundle

/**
 * Inference model built on SavedModelBundle format to predict on images.
 *
 * @property [bundle] SavedModelBundle.
 */
public open class SavedModel(private val bundle: SavedModelBundle) :
    TensorFlowInferenceModel(bundle.graph(), bundle.session()) {

    init {
        input = PLACEHOLDER
        isModelInitialized = true
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
            val predictedLabel = predict(dataset.getX(i).first, inputTensorName, outputTensorName)
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
            return SavedModel(SavedModelBundle.load(pathToModel, "serve"))
        }
    }
}
