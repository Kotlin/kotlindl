/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment

import android.graphics.Bitmap
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.CameraXCompatibleModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.preprocessing.*
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape

/**
 * Face detection model implementation.
 *
 * @see ONNXModels.FaceDetection.UltraFace320
 * @see ONNXModels.FaceDetection.UltraFace640
 */
public class FaceDetectionModel(override val internalModel: OnnxInferenceModel) : FaceDetectionModelBase<Bitmap>(),
    CameraXCompatibleModel, InferenceModel by internalModel {
    override var targetRotation: Float = 0f
    override val preprocessing: Operation<Bitmap, Pair<FloatArray, TensorShape>>
        get() = pipeline<Bitmap>()
            .rotate { degrees = targetRotation }
            .resize {
                outputWidth = internalModel.inputDimensions[2].toInt()
                outputHeight = internalModel.inputDimensions[1].toInt()
            }
            .toFloatArray { layout = TensorLayout.NCHW }
            .call(ONNXModels.FaceDetection.defaultPreprocessor)

    override fun copy(copiedModelName: String?, saveOptimizerState: Boolean, copyWeights: Boolean): InferenceModel {
        return FaceDetectionModel(internalModel.copy(copiedModelName, saveOptimizerState, copyWeights))
    }
}