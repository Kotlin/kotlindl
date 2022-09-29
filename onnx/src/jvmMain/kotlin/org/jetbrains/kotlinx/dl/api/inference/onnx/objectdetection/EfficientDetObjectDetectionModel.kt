/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.Coco
import org.jetbrains.kotlinx.dl.dataset.CocoVersion.V2017
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape
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
public class EfficientDetObjectDetectionModel(override val internalModel: OnnxInferenceModel) :
    EfficientDetObjectDetectionModelBase<BufferedImage>(), InferenceModel by internalModel {

    override val preprocessing: Operation<BufferedImage, Pair<FloatArray, TensorShape>>
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
        return EfficientDetObjectDetectionModel(internalModel.copy(copiedModelName, saveOptimizerState, copyWeights))
    }
}
