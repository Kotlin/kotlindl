/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.loaders

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.preprocessing.Identity
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation

/**
 * Basic interface for models loaded from S3.
 * @param T the type of the basic model for common functionality.
 * @param U the type of the pre-trained model for usage in Easy API.
 */
public interface ModelType<T : InferenceModel, U : InferenceModel> {
    /** Relative path to model for local and S3 buckets storages. */
    public val modelRelativePath: String

    /**
     * Preprocessing [Operation] specific for this model type.
     */
    public val preprocessor: Operation<FloatData, FloatData>
        get() = Identity()

    /** Returns the specially prepared pre-trained model of the type U. */
    public fun pretrainedModel(modelHub: ModelHub): U

    /** Loads the model, identified by this name, from the [modelHub]. */
    public fun model(modelHub: ModelHub): T {
        return modelHub.loadModel(this)
    }
}
