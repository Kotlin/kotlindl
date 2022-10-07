/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference.objectdetection

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.dataset.Coco
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException

/**
 * Special model class for detection objects on images
 * with built-in preprocessing and post-processing.
 *
 * It internally uses [ONNXModels.ObjectDetection.EfficientDetD0] or other EfficientDet models trained on the COCO dataset.
 *
 * @param [internalModel] model used to make predictions
 *
 * @since 0.4
 */
public class EfficientDetObjectDetectionModel(
    override val internalModel: OnnxInferenceModel,
    modelKindDescription: String? = null
) : EfficientDetObjectDetectionModelBase<BufferedImage>(modelKindDescription), InferenceModel by internalModel {

    override val preprocessing: Operation<BufferedImage, FloatData>
        get() = pipeline<BufferedImage>()
            .resize {
                outputHeight = inputDimensions[0].toInt()
                outputWidth = inputDimensions[1].toInt()
            }
            // the channels of input of EfficientDet models should be in RGB order
            // model is quite sensitive for this
            .convert { colorMode = ColorMode.RGB }
            .toFloatArray { }
    override val classLabels: Map<Int, String> = Coco.V2017.labels(zeroIndexed = false)

    /**
     * Constructs the object detection model from a given path.
     * @param [pathToModel] path to model
     */
    public constructor(pathToModel: String) : this(OnnxInferenceModel(pathToModel))

    /**
     * Returns the detected object for the given image file sorted by the score.
     *
     * NOTE: this method includes the EfficientDet - related preprocessing.
     *
     * @param [imageFile] File, should be an image.
     * @param [topK] The number of the detected objects with the highest score to be returned.
     * @return List of [DetectedObject] sorted by score.
     */
    @Throws(IOException::class)
    public fun detectObjects(imageFile: File, topK: Int = 5): List<DetectedObject> {
        return detectObjects(ImageConverter.toBufferedImage(imageFile), topK)
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): EfficientDetObjectDetectionModel {
        return EfficientDetObjectDetectionModel(
            internalModel.copy(copiedModelName, saveOptimizerState, copyWeights),
            modelKindDescription
        )
    }
}
