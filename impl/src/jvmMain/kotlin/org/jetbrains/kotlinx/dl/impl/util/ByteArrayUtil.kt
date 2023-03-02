/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.util

/** Normalizes [bytes] via division on 255 to get values in range '[0; 1)'.*/
public fun toNormalizedVector(bytes: ByteArray): FloatArray {
    return FloatArray(bytes.size) { ((bytes[it].toInt() and 0xFF)).toFloat() / 255f }
}

/** Converts [bytes] to [FloatArray]. */
public fun toRawVector(bytes: ByteArray): FloatArray {
    return FloatArray(bytes.size) { ((bytes[it].toInt() and 0xFF).toFloat()) }
}