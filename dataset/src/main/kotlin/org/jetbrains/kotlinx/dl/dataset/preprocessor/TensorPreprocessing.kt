/*
 * Copyright 2020-2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

/**
 * The whole tensor preprocessing pipeline DSL.
 *
 * It supports operations that implement [Preprocessor], e.g. [Rescaling] preprocessor.
 *
 * It's a part of the [org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing] pipeline DSL.
 */
public class TensorPreprocessing {
    /** Internal state of the [TensorPreprocessing]. The list of [Preprocessor].*/
    internal val operations = mutableListOf<Preprocessor>()

    /** Adds an [operation] to the [operations].*/
    public fun addOperation(operation: Preprocessor) {
        operations.add(operation)
    }
}

/** */
public fun TensorPreprocessing.rescale(block: Rescaling.() -> Unit) {
    addOperation(Rescaling().apply(block))
}

public fun TensorPreprocessing.normalize(block: Normalizing.() -> Unit) {
    addOperation(Normalizing().apply(block))
}






