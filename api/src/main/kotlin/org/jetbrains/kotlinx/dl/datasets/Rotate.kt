/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets

import java.awt.image.BufferedImage

//TODO: refactor to float value if affine transformation will work
public class Rotate(public var degrees: Degrees = Degrees.R_90) : ImagePreprocessor {
    override fun apply(image: BufferedImage, inputShape: ImageShape): Pair<BufferedImage, ImageShape> {
        return Pair(image, inputShape)
    }
}
