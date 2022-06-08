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

private const val OUTPUT_BOXES = "detection_boxes:0"
private const val OUTPUT_CLASSES = "detection_classes:0"
private const val OUTPUT_SCORES = "detection_scores:0"
private const val OUTPUT_NUMBER_OF_DETECTIONS = "num_detections:0"

/**
 * Special model class for detection objects on images
 * with built-in preprocessing and post-processing.
 *
 * It internally uses [ONNXModels.ObjectDetection.SSDMobileNetV1] model trained on the COCO dataset.
 *
 * @since 0.4
 */
public class SSDMobileNetV1ObjectDetectionModel : OnnxInferenceModel() {
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
        val boxes = (rawPrediction[OUTPUT_BOXES] as Array<Array<FloatArray>>)[0]
        val classIndices = (rawPrediction[OUTPUT_CLASSES] as Array<FloatArray>)[0]
        val probabilities = (rawPrediction[OUTPUT_SCORES] as Array<FloatArray>)[0]
        val numberOfFoundObjects = (rawPrediction[OUTPUT_NUMBER_OF_DETECTIONS] as FloatArray)[0].toInt()

        for (i in 0 until numberOfFoundObjects) {
            val detectedObject = DetectedObject(
                classLabel = cocoCategories[classIndices[i].toInt()]!!,
                probability = probabilities[i],
                // top, left, bottom, right
                yMin = boxes[i][0],
                xMin = boxes[i][1],
                yMax = boxes[i][2],
                xMax = boxes[i][3]
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
        val preprocessing = preprocess {
            transformImage {
                resize {
                    outputHeight = this@SSDMobileNetV1ObjectDetectionModel.inputShape[1].toInt()
                    outputWidth = this@SSDMobileNetV1ObjectDetectionModel.inputShape[2].toInt()
                }
                convert { colorMode = ColorMode.RGB }
            }
        }

        val (data, shape) = preprocessing(imageFile)

        val preprocessedData = ONNXModels.ObjectDetection.SSDMobileNetV1.preprocessInput(
            data,
            longArrayOf(shape.width!!, shape.height!!, shape.channels!!)
        )

        return this.detectObjects(preprocessedData, topK)
    }
}
