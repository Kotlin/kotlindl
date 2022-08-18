/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.preprocessing.PreprocessingPipeline

/**
 * Applies the final image preprocessing that is specific for each of available models trained on ImageNet according chosen [modelTypePreprocessing].
 *
 * @property [modelTypePreprocessing] One the supported models pre-trained on ImageNet.
 */
public class Sharpen(public var modelTypePreprocessing: ModelType<*, *> = TFModels.CV.VGG16()) :
    Operation<Pair<FloatArray, TensorShape>, Pair<FloatArray, TensorShape>> {
    override fun apply(input: Pair<FloatArray, TensorShape>): Pair<FloatArray, TensorShape> {
        val (data, tensorShape) = input
        return modelTypePreprocessing.preprocessInput(data, tensorShape.dims()) to tensorShape
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        throw UnsupportedOperationException("Currently it is not possible to get final shape of sharpen operation")
    }
}


/** Image DSL Preprocessing extension.*/
public fun <I> Operation<I, Pair<FloatArray, TensorShape>>.sharpen(sharpBlock: Sharpen.() -> Unit): Operation<I, Pair<FloatArray, TensorShape>> =
    PreprocessingPipeline(this, Sharpen().apply(sharpBlock))
