/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

import mu.KLogger

public abstract class InferenceModel: AutoCloseable {
    /** Model name. */
    public var name: String? = null

    /**
     * Predicts the class of [inputData].
     *
     * @param [inputData] The single example with unknown label.
     * @return Predicted class index.
     */
    public abstract fun predict(inputData: FloatArray): Int

    /**
     * Predicts vector of probabilities instead of specific class in [predict] method.
     *
     * @param [inputData] The single example with unknown vector of probabilities.
     * @param [predictionTensorName] The name of prediction tensor. It could be changed, if you need to get alternative outputs from intermediate parts of the TensorFlow graphs.
     * @return Vector that represents the probability distributions of a list of potential outcomes
     */
    public abstract fun predictSoftly(inputData: FloatArray, predictionTensorName: String = ""): FloatArray

    /**
     * Chain-like setter to set up input shape.
     *
     * @param [dims] The input shape.
     */
    public abstract fun reshape(vararg dims: Long)

    /**
     * Creates a copy.
     *
     * @param [copiedModelName] Set up this name to make a copy with a new name.
     * @return A copied inference model.
     */
    // TODO: add tests to the tests
    public abstract fun copy(
        copiedModelName: String? = null,
        saveOptimizerState: Boolean = false, // TODO, check this case
        copyWeights: Boolean = true
    ): TensorFlowInferenceModel
}
