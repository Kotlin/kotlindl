/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

/**
 * Represents the tensor preprocessing stage of the [Preprocessing].
 * Consists of the operations implementing [Preprocessor] which are applied to the tensor one by one.
 *
 * Supported operations include:
 * - [normalize],
 * - [rescale].
 *
 * @see Normalizing
 * @see Rescaling
 */
public class TensorPreprocessing {
    /** Internal state of the [TensorPreprocessing]. The list of [Preprocessor].*/
    internal val operations = mutableListOf<Preprocessor>()

    /** Adds an [operation] to the [operations].*/
    public fun addOperation(operation: Preprocessor) {
        operations.add(operation)
    }
}

/** Applies [Rescaling] preprocessor to the tensor to scale each value by a given coefficient. */
public fun TensorPreprocessing.rescale(block: Rescaling.() -> Unit) {
    addOperation(Rescaling().apply(block))
}

/** Applies [Normalizing] preprocessor to the tensor to normalize it with given mean and std values. */
public fun TensorPreprocessing.normalize(block: Normalizing.() -> Unit) {
    addOperation(Normalizing().apply(block))
}






