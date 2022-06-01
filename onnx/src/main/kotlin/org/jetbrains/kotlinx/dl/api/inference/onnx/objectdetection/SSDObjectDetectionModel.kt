/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection

import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.handler.cocoCategoriesForSSD
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

private const val OUTPUT_BOXES = "bboxes"
private const val OUTPUT_LABELS = "labels"
private const val OUTPUT_SCORES = "scores"
private const val INPUT_SIZE = 1200

/**
 * Special model class for detection objects on images
 * with built-in preprocessing and post-processing.
 *
 * It internally uses [ONNXModels.ObjectDetection.SSD] trained on the COCO dataset.
 *
 * Note that output class labels do not correspond to ids in COCO annotations.
 * If you want to evaluate this model on the COCO validation/test set, you need to convert class predictions using appropriate mapping.
 *
 *
 *  @see <a href="https://github.com/lji72/inference/blob/tf_ssd_resent34_align_onnx/others/cloud/single_stage_detector/tensorflow/dataset_config/coco_labelmap.txt">
 *     Example mapping</a>
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
    public fun  detectObjects(inputData: FloatArray, topK: Int = 5): List<DetectedObject> {
        val rawPrediction = this.predictRaw(inputData)

        val foundObjects = mutableListOf<DetectedObject>()
        val boxes = (rawPrediction[OUTPUT_BOXES] as Array<Array<FloatArray>>)[0]
        val classIndices = (rawPrediction[OUTPUT_LABELS] as Array<LongArray>)[0]
        val probabilities = (rawPrediction[OUTPUT_SCORES] as Array<FloatArray>)[0]
        val numberOfFoundObjects = boxes.size

        for (i in 0 until numberOfFoundObjects) {
            val detectedObject = DetectedObject(
                classLabel = cocoCategoriesForSSD[classIndices[i].toInt()]!!,
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
        val preprocessing = preprocess {
            transformImage {
                resize {
                    outputHeight = INPUT_SIZE
                    outputWidth = INPUT_SIZE
                }
                convert { colorMode = ColorMode.RGB }
            }
        }

        val (data, shape) = preprocessing(imageFile)

        val preprocessedData = ONNXModels.ObjectDetection.SSD.preprocessInput(
            data,
            longArrayOf(shape.width!!, shape.height!!, shape.channels!!)
        )

        return this.detectObjects(preprocessedData, topK)
    }
}
