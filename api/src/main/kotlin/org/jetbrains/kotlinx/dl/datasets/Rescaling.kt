/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets

public class Rescaling(public var scalingCoefficient: Float = 255f) : Preprocessor {
    override fun apply(data: FloatArray): FloatArray {
        for (i in data.indices) {
            data[i] = data[i] / scalingCoefficient
        }
        return data
    }
}
