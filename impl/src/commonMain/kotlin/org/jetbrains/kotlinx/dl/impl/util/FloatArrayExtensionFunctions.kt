/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.util

// TODO: add the same method but with 3d channel navigation
/** */
public fun FloatArray.set3D(
    rowIndex: Int,
    columnIndex: Int,
    channelIndex: Int,
    width: Int,
    channels: Int,
    value: Float
) {
    this[width * rowIndex * channels + columnIndex * channels + channelIndex] = value
}

/** */
public fun FloatArray.get3D(rowIndex: Int, columnIndex: Int, channelIndex: Int, width: Int, channels: Int): Float {
    return this[width * rowIndex * channels + columnIndex * channels + channelIndex]
}

/** */
public fun FloatArray.set2D(rowIndex: Int, columnIndex: Int, width: Int, value: Float) {
    this[width * rowIndex + columnIndex] = value
}

/** */
public fun FloatArray.get2D(rowIndex: Int, columnIndex: Int, width: Int): Float {
    return this[width * rowIndex + columnIndex]
}

/**
 * Returns the index of the maximum element in the given FloatArray.
 * TODO: Should be replaced with Multik in future.
 */
public fun FloatArray.argmax(): Int =
    maxOrNull()?.let { max -> indexOfFirst { it == max } } ?: -1
