/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType

/**
 * Applies the final image preprocessing that is specific for each of available models trained on ImageNet according chosen [modelTypePreprocessing].
 *
 * @property [modelTypePreprocessing] One the supported models pre-trained on ImageNet.
 */
public class Sharpen(public var modelTypePreprocessing: ModelType<*, *> = TFModels.CV.VGG16()) : Preprocessor {
    override fun apply(data: FloatArray, inputShape: ImageShape): FloatArray {
        val tensorShape = longArrayOf(inputShape.width!!, inputShape.height!!, inputShape.channels!!)
        return modelTypePreprocessing.preprocessInput(data, tensorShape)
    }
}


/** Image DSL Preprocessing extension.*/
public fun TensorPreprocessing.sharpen(sharpBlock: Sharpen.() -> Unit) {
    addOperation(Sharpen().apply(sharpBlock))
}
