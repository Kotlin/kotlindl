/*
 * Copyright 2022-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.loaders

import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModelBase
import java.io.File

/**
 * Base [ModelType] for TensorFlow models.
 *
 * @see TFModels
 * @see TFModelHub
 */
public interface TFModelType<T : TensorFlowInferenceModelBase, U> : ModelType<T, U> {
    /**
     * Loads model configuration from the provided [jsonFile].
     */
    public fun loadModelConfiguration(jsonFile: File): T
}