/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference.facealignment

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.preprocessing.*
import org.jetbrains.kotlinx.dl.impl.preprocessing.camerax.toBitmap
import org.jetbrains.kotlinx.dl.onnx.inference.CameraXCompatibleModel
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.doWithRotation

/**
 * Face detection model implementation.
 *
 * @see ONNXModels.FaceDetection.UltraFace320
 * @see ONNXModels.FaceDetection.UltraFace640
 */
public class FaceDetectionModel(
    override val internalModel: OnnxInferenceModel,
    modelKindDescription: String? = null
) : FaceDetectionModelBase<Bitmap>(modelKindDescription), CameraXCompatibleModel, InferenceModel by internalModel {
    override var targetRotation: Int = 0
    override val preprocessing: Operation<Bitmap, FloatData>
        get() = pipeline<Bitmap>()
            .rotate { degrees = targetRotation.toFloat() }
            .resize {
                outputWidth = internalModel.inputDimensions[2].toInt()
                outputHeight = internalModel.inputDimensions[1].toInt()
            }
            .toFloatArray { layout = TensorLayout.NCHW }
            .call(ONNXModels.FaceDetection.defaultPreprocessor)

    override fun copy(copiedModelName: String?, saveOptimizerState: Boolean, copyWeights: Boolean): InferenceModel {
        return FaceDetectionModel(
            internalModel.copy(copiedModelName, saveOptimizerState, copyWeights),
            modelKindDescription
        )
    }
}

/**
 * Detects [topK] faces on the given [imageProxy]. If [topK] is negative all detected faces are returned.
 * @param [iouThreshold] threshold IoU value for the non-maximum suppression applied during postprocessing
 */
public fun FaceDetectionModelBase<Bitmap>.detectFaces(
    imageProxy: ImageProxy,
    topK: Int = 5,
    iouThreshold: Float = 0.5f
): List<DetectedObject> {
    if (this is CameraXCompatibleModel) {
        return doWithRotation(imageProxy.imageInfo.rotationDegrees) {
            detectFaces(imageProxy.toBitmap(), topK, iouThreshold)
        }
    }
    return detectFaces(imageProxy.toBitmap(applyRotation = true), topK, iouThreshold)
}