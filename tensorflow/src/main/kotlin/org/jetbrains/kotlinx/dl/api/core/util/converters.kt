/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

import org.tensorflow.ndarray.StdArrays
import org.tensorflow.types.TFloat32
import java.nio.Buffer
import java.nio.FloatBuffer

/**
 * Creates a float tensor from a (possibly nested) float array.
 *
 * TensorFlow Java 1.x removed the reflective `Tensor.create(Object)` factory that the 1.15 API
 * provided, so the supported ranks are dispatched explicitly here. Rank is determined by walking
 * the first element of each level, which matches how the weights loaded from Keras HDF5 files are
 * shaped (they are always rectangular).
 */
@Suppress("UNCHECKED_CAST")
public fun createFloatTensor(data: Any): TFloat32 = when (data) {
    is Float -> TFloat32.scalarOf(data)
    is FloatArray -> TFloat32.tensorOf(StdArrays.ndCopyOf(data))
    is Array<*> -> when (val second = data.firstOrNull()) {
        is FloatArray -> TFloat32.tensorOf(StdArrays.ndCopyOf(data as Array<FloatArray>))
        is Array<*> -> when (second.firstOrNull()) {
            is FloatArray -> TFloat32.tensorOf(StdArrays.ndCopyOf(data as Array<Array<FloatArray>>))
            is Array<*> -> TFloat32.tensorOf(StdArrays.ndCopyOf(data as Array<Array<Array<FloatArray>>>))
            else -> throw IllegalArgumentException("Unsupported tensor data of rank higher than 4.")
        }
        else -> throw IllegalArgumentException("Unsupported tensor data element: ${second?.let { it::class }}")
    }
    else -> throw IllegalArgumentException("Cannot create a float tensor from ${data::class}.")
}


/** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
public fun serializeToBuffer(src: Array<FloatArray>, start: Int, length: Int): FloatBuffer {
    val buffer = FloatBuffer.allocate(length * src[0].size)
    for (i in start until start + length) {
        buffer.put(src[i])
    }
    return (buffer as Buffer).rewind() as FloatBuffer
}

/** Converts [src] to [FloatBuffer]. */
public fun serializeToBuffer(src: Array<FloatArray>): FloatBuffer {
    val buffer = FloatBuffer.allocate(src.size * src[0].size)
    for (element in src) {
        buffer.put(element)
    }
    return (buffer as Buffer).rewind() as FloatBuffer
}

/** Converts [src] to [FloatBuffer]. */
public fun serializeToBuffer(src: FloatArray): FloatBuffer {
    val buffer = FloatBuffer.allocate(src.size)
    buffer.put(src)
    return (buffer as Buffer).rewind() as FloatBuffer
}

/** Converts [src] to [FloatBuffer]. */
public fun serializeLabelsToBuffer(src: FloatArray, amountOfClasses: Long): FloatBuffer {
    val oneHotEncodedLabels = Array(src.size) {
        FloatArray(amountOfClasses.toInt()) { 0.0f }
    }
    for (i in src.indices) {
        val label = src[i]

        if (amountOfClasses == 1L) {
            oneHotEncodedLabels[i][0] = label // for regression tasks
        } else {
            require(0.0f <= label && label < amountOfClasses) { "Label with index $i has value $label and is out of range [0.0, $amountOfClasses) for classification task." }

            oneHotEncodedLabels[i][label.toInt()] = 1f
        } //TODO: should be implemented with the real OHE here
    }

    val buffer = FloatBuffer.allocate(oneHotEncodedLabels.size * oneHotEncodedLabels[0].size)
    for (element in oneHotEncodedLabels) {
        buffer.put(element)
    }
    return (buffer as Buffer).rewind() as FloatBuffer
}

