/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.extension

// TODO: add the same method but with 3d channel navigation
public fun FloatArray.set3D(x: Int, y: Int, z: Int, width: Int, channels: Int, value: Float) {
    this[width * x * channels + y * channels + z] = value
}

public fun FloatArray.get3D(x: Int, y: Int, z: Int, width: Int, channels: Int): Float {
    return this[width * x * channels + y * channels + z]
}

public fun FloatArray.set2D(x: Int, y: Int, width: Int, value: Float) {
    this[width * x + y] = value
}

public fun FloatArray.get2D(x: Int, y: Int, width: Int): Float {
    return this[width * x + y]
}
