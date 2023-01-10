/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.savedmodel

import org.jetbrains.kotlinx.dl.api.core.util.PLACEHOLDER
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.tensorflow.SavedModelBundle

/**
 * Inference model built on SavedModelBundle format to predict on images.
 *
 * @property [bundle] SavedModelBundle.
 */
public open class SavedModel(private val bundle: SavedModelBundle) :
    TensorFlowInferenceModel(bundle.graph(), bundle.session()) {

    init {
        input = PLACEHOLDER
        isModelInitialized = true
    }

    override fun close() {
        super.close()
        bundle.close()
    }

    public companion object {
        /**
         * Loads model from SavedModelBundle format.
         */
        public fun load(pathToModel: String): SavedModel {
            return SavedModel(SavedModelBundle.load(pathToModel, "serve"))
        }
    }
}
