/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets

public class Rotate(degrees: Degrees) : ImagePreprocessor {
    override fun apply(image: FloatArray): FloatArray {
        return image
    }
}
