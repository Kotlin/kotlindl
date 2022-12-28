/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

import org.jetbrains.kotlinx.dl.api.core.FloatData

/**
 * The basic interface for all models which defines the basic functions required for inference tasks only.
 *
 * @param [R] type of inference result, produced by this model.
 * @see InferenceResultConverter
 */
public interface InferenceModel<R> : AutoCloseable {
    /** Input specification for this model. */
    public val inputDimensions: LongArray

    /**
     * Provides methods for converting inference result from this model to the common data types.
     */
    public val resultConverter: InferenceResultConverter<R>

    /**
     * Run inference on the provided [inputData] and pass inference result to the [extractResult] function.
     */
    public fun <T> predict(inputData: FloatData, extractResult: (R) -> T): T

    /**
     * Run inference on the provided [inputs], calculate the specified [outputs] and pass inference result to the [extractResult] function.
     */
    public fun <T> predict(inputs: Map<String, FloatData>, outputs: List<String>, extractResult: (R) -> T): T

    /**
     * Creates a copy of this model.
     *
     * @return A copied inference model.
     */
    public fun copy(): InferenceModel<R>
}
