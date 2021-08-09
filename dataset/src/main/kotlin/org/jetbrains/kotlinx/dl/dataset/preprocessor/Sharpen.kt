/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.preprocessInput

/**
 * Applies the final image preprocessing that is specific for each of available models trained on ImageNet according chosen [modelType].
 *
 * @property [modelType] One the supported models pre-trained on ImageNet.
 */
public class Sharpen(public var modelType: ModelType = ModelType.VGG_16) : Preprocessor {
    override fun apply(data: FloatArray, inputShape: ImageShape): FloatArray {
        val tensorShape = longArrayOf(inputShape.width!!, inputShape.height!!, inputShape.channels)
        return preprocessInput(data, tensorShape, modelType)
    }
}
