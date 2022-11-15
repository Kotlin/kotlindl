/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference.facealignment

import ai.onnxruntime.OrtSession
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxHighLevelModel
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.getFloatArray

/**
 * Base class for face alignment models.
 */
public abstract class FaceAlignmentModelBase<I>(override val modelKindDescription: String? = null) :
    OnnxHighLevelModel<I, List<Landmark>> {
    /**
     * Name of the output tensor.
     */
    protected abstract val outputName: String

    override fun convert(output: OrtSession.Result): List<Landmark> {
        val landMarks = mutableListOf<Landmark>()
        val floats = output.getFloatArray(outputName)
        for (i in floats.indices step 2) {
            landMarks.add(Landmark((1 + floats[i]) / 2, (1 + floats[i + 1]) / 2))
        }

        return landMarks
    }

    /**
     * Detects [Landmark] objects on the given [image].
     */
    public fun detectLandmarks(image: I): List<Landmark> = predict(image)
}