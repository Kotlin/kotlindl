/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection

import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.handler.cocoCategories
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

private const val OUTPUT_NAME = "detections:0"

/**
 * Special model class for detection objects on images
 * with built-in preprocessing and post-processing.
 *
 * It internally uses [ONNXModels.ObjectDetection.EfficientDetD0] or other EfficientDet models trained on the COCO dataset.
 *
 * @since 0.4
 */
public class EfficientDetObjectDetectionModel : OnnxInferenceModel() {
    /**
     * Returns the detected object for the given image file sorted by the score.
     *
     * @param [inputData] Preprocessed data from the image file.
     * @return List of [DetectedObject] sorted by score.
     */
    public fun detectObjects(inputData: FloatArray): List<DetectedObject> {
        val rawPrediction = this.predictRaw(inputData)
        val foundObjects = mutableListOf<DetectedObject>()
        val items = (rawPrediction[OUTPUT_NAME] as Array<Array<FloatArray>>)[0]

        for (i in items.indices) {
            val probability = items[i][5]
            if (probability != 0.0f) {
                val detectedObject = DetectedObject(
                    classLabel = cocoCategories[items[i][6].toInt()]!!,
                    probability = probability,
                    // left, bot, right, top
                    xMin = minOf(items[i][2] / inputShape[2], 1.0f),
                    yMax = minOf(items[i][3] / inputShape[1], 1.0f),
                    xMax = minOf(items[i][4] / inputShape[2], 1.0f),
                    yMin = minOf(items[i][1] / inputShape[1], 1.0f)
                )
                foundObjects.add(detectedObject)
            }
        }

        foundObjects.sortByDescending { it.probability }
        return foundObjects
    }

    /**
     * Returns the detected object for the given image file sorted by the score.
     *
     * NOTE: this method includes the EfficientDet - related preprocessing.
     *
     * @param [imageFile] File, should be an image.
     * @return List of [DetectedObject] sorted by score.
     */
    public fun detectObjects(imageFile: File): List<DetectedObject> {
        val preprocessing = preprocess {
            transformImage {
                resize {
                    outputHeight = inputShape[1].toInt()
                    outputWidth = inputShape[2].toInt()
                }
                // the channels of input of EfficientDet models should be in RGB order
                // model is quite sensitive for this
                convert { colorMode = ColorMode.RGB }
            }
        }

        val (data, _) = preprocessing(imageFile)
        // we don't need special preprocessing here
        return this.detectObjects(data)
    }
}
