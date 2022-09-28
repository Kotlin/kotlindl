/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment

import android.graphics.Bitmap
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.CameraXCompatibleModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.preprocessing.*
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape

/**
 * The light-weight API for solving Face Alignment task.
 *
 * @param [internalModel] model used to make predictions
 */
public class Fan2D106FaceAlignmentModel(override val internalModel: OnnxInferenceModel) : FaceAlignmentModelBase<Bitmap>(),
    CameraXCompatibleModel, InferenceModel by internalModel {

    override val outputName: String = "fc1"
    override var targetRotation: Float = 0f

    override val preprocessing: Operation<Bitmap, Pair<FloatArray, TensorShape>> = pipeline<Bitmap>()
        .resize {
            outputWidth = internalModel.inputDimensions[2].toInt()
            outputHeight = internalModel.inputDimensions[1].toInt()
        }
        .rotate { degrees = targetRotation }
        .toFloatArray { layout = TensorLayout.NCHW }

    override fun copy(copiedModelName: String?, saveOptimizerState: Boolean, copyWeights: Boolean): InferenceModel {
        return Fan2D106FaceAlignmentModel(internalModel.copy(copiedModelName, saveOptimizerState, copyWeights))
    }
}