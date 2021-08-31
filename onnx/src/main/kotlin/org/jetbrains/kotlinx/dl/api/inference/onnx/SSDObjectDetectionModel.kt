/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.dataset.handler.cocoCategories
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

/**
 * Special model class for detection objects on images
 * with built-in preprocessing and post-processing.
 *
 * It internally uses SSD model trained on the COCO dataset.
 *
 * @since 0.3
 *
 * @see <a href="https://arxiv.org/abs/1512.02325">
 *     SSD: Single Shot MultiBox Detector.</a>
 */
public class SSDObjectDetectionModel : OnnxInferenceModel() {
    /**
     * Returns the top N detected object for the given image file.
     *
     * NOTE: this method doesn't include the SSD - related preprocessing.
     *
     * @param [inputData] Preprocessed data from the image file.
     * @param [topK] The number of the detected objects with the highest score to be returned.
     * @return List of [DetectedObject] sorted by score.
     */
    public fun detectObjects(inputData: FloatArray, topK: Int = 5): List<DetectedObject> {
        val rawPrediction = this.predictRaw(inputData)

        val foundObjects = mutableListOf<DetectedObject>()
        val boxes = rawPrediction[0][0] as Array<FloatArray>
        val classIndices = rawPrediction[1][0] as LongArray
        val probabilities = rawPrediction[2][0] as FloatArray
        val numberOfFoundObjects = boxes.size

        for (i in 0 until numberOfFoundObjects) {
            val detectedObject = DetectedObject(
                classLabel = cocoCategories[classIndices[i].toInt()]!!,
                probability = probabilities[i],
                // left, bot, right, top
                xMin = boxes[i][0],
                yMax = boxes[i][1],
                xMax = boxes[i][2],
                yMin = boxes[i][3]
            )
            foundObjects.add(detectedObject)
        }

        if (topK > 0) {
            foundObjects.sortByDescending { it.probability }
            return foundObjects.take(topK)
        }

        return foundObjects
    }

    /**
     * Returns the top N detected object for the given image file.
     *
     * NOTE: this method includes the SSD - related preprocessing.
     *
     * @param [imageFile] File, should be an image.
     * @param [topK] The number of the detected objects with the highest score to be returned.
     * @return List of [DetectedObject] sorted by score.
     */
    public fun detectObjects(imageFile: File, topK: Int = 5): List<DetectedObject> {
        val preprocessing: Preprocessing = preprocess {
            transformImage {
                load {
                    pathToData = imageFile
                    imageShape = ImageShape(224, 224, 3)
                    colorMode = ColorOrder.BGR
                }
                resize {
                    outputHeight = 1200
                    outputWidth = 1200
                }
            }
        }

        val (data, shape) = preprocessing()

        val preprocessedData = ONNXModels.ObjectDetection.SSD.preprocessInput(
            data,
            longArrayOf(shape.width!!, shape.height!!, shape.channels) // TODO: refactor to the imageShape
        )

        return this.detectObjects(preprocessedData, topK)
    }
}
