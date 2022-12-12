/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference.facealignment

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.preprocessing.TensorLayout
import org.jetbrains.kotlinx.dl.impl.preprocessing.camerax.toBitmap
import org.jetbrains.kotlinx.dl.impl.preprocessing.resize
import org.jetbrains.kotlinx.dl.impl.preprocessing.rotate
import org.jetbrains.kotlinx.dl.impl.preprocessing.toFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.CameraXCompatibleModel
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.doWithRotation

/**
 * The light-weight API for solving Face Alignment task.
 *
 * @param [internalModel] model used to make predictions
 */
public class Fan2D106FaceAlignmentModel(
    override val internalModel: OnnxInferenceModel,
    modelKindDescription: String? = null
) : FaceAlignmentModelBase<Bitmap>(modelKindDescription), CameraXCompatibleModel, InferenceModel by internalModel {
    override val outputName: String = "fc1"
    override var targetRotation: Int = 0

    override val preprocessing: Operation<Bitmap, FloatData> = pipeline<Bitmap>()
        .resize {
            outputWidth = internalModel.inputDimensions[2].toInt()
            outputHeight = internalModel.inputDimensions[1].toInt()
        }
        .rotate { degrees = targetRotation.toFloat() }
        .toFloatArray { layout = TensorLayout.NCHW }

    override fun copy(copiedModelName: String?, saveOptimizerState: Boolean, copyWeights: Boolean): InferenceModel {
        return Fan2D106FaceAlignmentModel(
            internalModel.copy(copiedModelName, saveOptimizerState, copyWeights),
            modelKindDescription
        )
    }
}

/**
 * Detects [Landmark] objects on the given [imageProxy].
 */
public fun FaceAlignmentModelBase<Bitmap>.detectLandmarks(imageProxy: ImageProxy): List<Landmark> {
    if (this is CameraXCompatibleModel) {
        return doWithRotation(imageProxy.imageInfo.rotationDegrees) {
            detectLandmarks(imageProxy.toBitmap())
        }
    }
    return detectLandmarks(imageProxy.toBitmap(applyRotation = true))
}