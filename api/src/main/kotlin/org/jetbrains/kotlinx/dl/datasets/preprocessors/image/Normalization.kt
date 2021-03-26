/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets.preprocessors.image

import org.jetbrains.kotlinx.dl.datasets.preprocessors.ImageShape
import java.awt.image.BufferedImage

// TODO: https://en.wikipedia.org/wiki/Normalization_(image_processing)
public class Normalization(newMin: Float, newMax: Float) : ImagePreprocessor {
    override fun apply(image: BufferedImage, inputShape: ImageShape): Pair<BufferedImage, ImageShape> {
        TODO("Not yet implemented")
    }

}
