/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference.objectdetection

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.dataset.Coco
import org.jetbrains.kotlinx.dl.impl.preprocessing.TensorLayout
import org.jetbrains.kotlinx.dl.impl.preprocessing.camerax.toBitmap
import org.jetbrains.kotlinx.dl.impl.preprocessing.resize
import org.jetbrains.kotlinx.dl.impl.preprocessing.rotate
import org.jetbrains.kotlinx.dl.impl.preprocessing.toFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.CameraXCompatibleModel
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.doWithRotation

/**
 * Special model class for detection objects on images with built-in preprocessing and post-processing.
 * Suitable for models with SSD like output decoding.
 *
 * It internally uses [ONNXModels.ObjectDetection.EfficientDetLite0] or other SSDLike models trained on the COCO dataset.
 *
 * @param [internalModel] model used to make predictions
 *
 * @since 0.5
 */
public class SSDLikeModel(
    override val internalModel: OnnxInferenceModel, metadata: SSDLikeModelMetadata,
    modelKindDescription: String? = null
) : SSDLikeModelBase<Bitmap>(metadata, modelKindDescription), CameraXCompatibleModel, InferenceModel by internalModel {

    override val classLabels: Map<Int, String> = Coco.V2017.labels(zeroIndexed = true)

    override var targetRotation: Int = 0

    override val preprocessing: Operation<Bitmap, FloatData>
        get() = pipeline<Bitmap>()
            .resize {
                outputHeight = internalModel.inputDimensions[0].toInt()
                outputWidth = internalModel.inputDimensions[1].toInt()
            }
            .rotate { degrees = targetRotation.toFloat() }
            .toFloatArray { layout = TensorLayout.NHWC }

    override fun close() {
        internalModel.close()
    }
}

/**
 * Returns the detected object for the given image sorted by the score.
 * Internal preprocessing is updated to rotate image to match target orientation.
 * After prediction, internal preprocessing is restored to the original state.
 *
 * @param [imageProxy] Input image.
 * @param [topK] The number of the detected objects with the highest score to be returned.
 * @return List of [DetectedObject] sorted by score.
 */
public fun ObjectDetectionModelBase<Bitmap>.detectObjects(imageProxy: ImageProxy, topK: Int = 3): List<DetectedObject> =
    when (this) {
        is CameraXCompatibleModel -> {
            doWithRotation(imageProxy.imageInfo.rotationDegrees) { detectObjects(imageProxy.toBitmap(), topK) }
        }

        else -> detectObjects(imageProxy.toBitmap(applyRotation = true), topK)
    }
