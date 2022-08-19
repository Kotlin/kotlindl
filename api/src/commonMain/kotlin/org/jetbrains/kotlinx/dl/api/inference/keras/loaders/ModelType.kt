/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.loaders

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Identity
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation

/**
 * Basic interface for models loaded from S3.
 * @param T the type of the basic model for common functionality.
 * @param U the type of the pre-trained model for usage in Easy API.
 */
public interface ModelType<T : InferenceModel, U : InferenceModel> {
    /** Relative path to model for local and S3 buckets storages. */
    public val modelRelativePath: String

    /**
     * If true it means that the second dimension is related to number of channels in image has short notation as `NCWH`,
     * otherwise, channels are at the last position and has a short notation as `NHWC`.
     */
    public val channelsFirst: Boolean

    /**
     * An expected channels order for the input image.
     * Note: the wrong choice of this parameter can significantly impact the model's performance.
     */
    public val inputColorMode: ColorMode

    /**
     * Preprocessing [Operation] specific for this model type.
     */
    public val preprocessor: Operation<Pair<FloatArray, TensorShape>, Pair<FloatArray, TensorShape>>
        get() = Identity()

    /** Returns the specially prepared pre-trained model of the type U. */
    public fun pretrainedModel(modelHub: ModelHub): U

    /** Loads the model, identified by this name, from the [modelHub]. */
    public fun model(modelHub: ModelHub): T {
        return modelHub.loadModel(this)
    }
}
