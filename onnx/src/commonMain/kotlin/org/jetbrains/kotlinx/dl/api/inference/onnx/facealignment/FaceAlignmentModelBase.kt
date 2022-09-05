/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment

import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxPreTrainedModel

/**
 * Base class for face alignment models.
 */
public abstract class FaceAlignmentModelBase<I> : OnnxPreTrainedModel<I, List<Landmark>> {
    /**
     * Name of the output tensor.
     */
    protected abstract val outputName: String

    override fun convert(output: Map<String, Any>): List<Landmark> {
        val landMarks = mutableListOf<Landmark>()
        val floats = (output[outputName] as Array<*>)[0] as FloatArray
        for (i in floats.indices step 2) {
            landMarks.add(Landmark(floats[i], floats[i + 1]))
        }

        return landMarks
    }

    /**
     * Detects [Landmark] objects on the given [image].
     */
    public fun detectLandmarks(image: I): List<Landmark> = predict(image)
}