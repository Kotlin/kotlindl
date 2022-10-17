/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

/**
 * The basic interface for all models which defines the basic functions required for inference tasks only.
 */
public interface InferenceModel : AutoCloseable {
    /** Input specification for this model. */
    public val inputDimensions: LongArray

    /**
     * Predicts the class of [inputData].
     *
     * @param [inputData] The single example with unknown label.
     * @return Predicted class index.
     */
    public fun predict(inputData: FloatArray): Int

    /**
     * Predicts vector of probabilities instead of specific class in [predict] method.
     *
     * @param [inputData] The single example with unknown vector of probabilities.
     * @param [predictionTensorName] The name of prediction tensor. It could be changed, if you need to get alternative outputs from model graph.
     * @return Vector that represents the probability distributions of a list of potential outcomes
     */
    public fun predictSoftly(inputData: FloatArray, predictionTensorName: String = ""): FloatArray

    /**
     * Chain-like setter to set up input shape.
     *
     * @param [dims] The input shape.
     */
    public fun reshape(vararg dims: Long)

    /**
     * Creates a copy.
     *
     * @param [copiedModelName] Set up this name to make a copy with a new name.
     * @return A copied inference model.
     */
    public fun copy(
        copiedModelName: String? = null,
        saveOptimizerState: Boolean = false,
        copyWeights: Boolean = true
    ): InferenceModel
}
