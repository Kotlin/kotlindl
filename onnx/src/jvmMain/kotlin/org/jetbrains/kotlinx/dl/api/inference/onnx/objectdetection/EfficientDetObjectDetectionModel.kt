/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.handler.cocoCategories
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException

private const val OUTPUT_NAME = "detections:0"

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
public class EfficientDetObjectDetectionModel(private val internalModel: OnnxInferenceModel) : InferenceModel by internalModel {
    private val preprocessing: Operation<BufferedImage, Pair<FloatArray, TensorShape>>
        get() = pipeline<BufferedImage>()
            .resize {
                outputHeight = inputDimensions[0].toInt()
                outputWidth = inputDimensions[1].toInt()
            }
            // the channels of input of EfficientDet models should be in RGB order
            // model is quite sensitive for this
            .convert { colorMode = ColorMode.RGB }
            .toFloatArray {  }

    /**
     * Constructs the object detection model from a given path.
     * @param [pathToModel] path to model
     */
    public constructor(pathToModel: String): this(OnnxInferenceModel(pathToModel))

    /**
     * Returns the detected object for the given image file sorted by the score.
     *
     * @param [inputData] Preprocessed data from the image file.
     * @return List of [DetectedObject] sorted by score.
     */
    public fun detectObjects(inputData: FloatArray): List<DetectedObject> {
        val rawPrediction = internalModel.predictRaw(inputData)
        val foundObjects = mutableListOf<DetectedObject>()
        val items = (rawPrediction[OUTPUT_NAME] as Array<Array<FloatArray>>)[0]

        for (i in items.indices) {
            val probability = items[i][5]
            if (probability != 0.0f) {
                val detectedObject = DetectedObject(
                    classLabel = cocoCategories[items[i][6].toInt()]!!,
                    probability = probability,
                    // left, bot, right, top
                    xMin = minOf(items[i][2] / inputDimensions[1], 1.0f),
                    yMax = minOf(items[i][3] / inputDimensions[0], 1.0f),
                    xMax = minOf(items[i][4] / inputDimensions[1], 1.0f),
                    yMin = minOf(items[i][1] / inputDimensions[0], 1.0f)
                )
                foundObjects.add(detectedObject)
            }
        }

        foundObjects.sortByDescending { it.probability }
        return foundObjects
    }

    /**
     * Returns the detected object for the given image sorted by the score.
     *
     * NOTE: this method includes the EfficientDet - related preprocessing.
     *
     * @param [image] Input image.
     * @return List of [DetectedObject] sorted by score.
     */
    public fun detectObjects(image: BufferedImage): List<DetectedObject> {
        return detectObjects(preprocessing.apply(image).first)
    }

    /**
     * Returns the detected object for the given image file sorted by the score.
     *
     * NOTE: this method includes the EfficientDet - related preprocessing.
     *
     * @param [imageFile] File, should be an image.
     * @return List of [DetectedObject] sorted by score.
     */
    @Throws(IOException::class)
    public fun detectObjects(imageFile: File): List<DetectedObject> {
        return detectObjects(ImageConverter.toBufferedImage(imageFile))
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): EfficientDetObjectDetectionModel {
        return EfficientDetObjectDetectionModel(internalModel.copy(copiedModelName, saveOptimizerState, copyWeights))
    }
}
