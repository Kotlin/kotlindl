/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection

import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.handler.cocoCategories
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

/**
 * Special model class for detection objects on images
 * with built-in preprocessing and post-processing.
 *
 * It internally uses [ONNXModels.ObjectDetection.SSD] trained on the COCO dataset.
 *
 * @since 0.3
 */
public class SSDObjectDetectionModel : OnnxInferenceModel() {
    /**
     * Returns the top N detected object for the given image file sorted by the score.
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
        val boxes = (rawPrediction["bboxes"] as Array<Array<FloatArray>>)[0]
        val classIndices = (rawPrediction["labels"] as Array<LongArray>)[0]
        val probabilities = (rawPrediction["scores"] as Array<FloatArray>)[0]
        val numberOfFoundObjects = boxes.size

        for (i in 0 until numberOfFoundObjects) {
            val detectedObject = DetectedObject(
                classLabel = cocoCategories[classIndices[i].toInt()]!!,
                probability = probabilities[i],
                // left, bot, right, top
                xMin = boxes[i][0],
                yMin = boxes[i][1],
                xMax = boxes[i][2],
                yMax = boxes[i][3]
            )
            foundObjects.add(detectedObject)
        }

        foundObjects.sortByDescending { it.probability }

        if (topK > 0) {
            return foundObjects.take(topK)
        }

        return foundObjects
    }

    /**
     * Returns the top N detected object for the given image file sorted by the score.
     *
     * NOTE: this method includes the SSD - related preprocessing.
     *
     * @param [imageFile] File, should be an image.
     * @param [topK] The number of the detected objects with the highest score to be returned.
     * @return List of [DetectedObject] sorted by score.
     */
    public fun detectObjects(imageFile: File, topK: Int = 5): List<DetectedObject> {
        val preprocessing: Preprocessing = preprocess {
            load {
                pathToData = imageFile
                imageShape = ImageShape(null, null, 3)
            }
            transformImage {
                resize {
                    outputHeight = 1200
                    outputWidth = 1200
                }
                convert { colorMode = ColorMode.BGR }
            }
        }

        val (data, shape) = preprocessing()

        val preprocessedData = ONNXModels.ObjectDetection.SSD.preprocessInput(
            data,
            longArrayOf(shape.width!!, shape.height!!, shape.channels!!) // TODO: refactor to the imageShape
        )

        return this.detectObjects(preprocessedData, topK)
    }
}
