/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.util

/**
 * Flattens the given array of float values.
 * @return flattened array
 */
public fun Array<*>.flattenFloats(): FloatArray {
    val result = mutableListOf<Float>()

    fun flatten(array: Any?): Unit = when (array) {
        is FloatArray -> array.forEach { result.add(it) }
        is Array<*> -> array.forEach { flatten(it) }
        else -> throw IllegalArgumentException("Cannot flatten object: '$array'")
    }

    flatten(this)

    return result.toFloatArray()
}