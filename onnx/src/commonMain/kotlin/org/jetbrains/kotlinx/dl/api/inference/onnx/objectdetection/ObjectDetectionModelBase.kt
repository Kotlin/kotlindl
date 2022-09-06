/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection

import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxHighLevelModel

/**
 * Base class for object detection models.
 */
public abstract class ObjectDetectionModelBase<I> : OnnxHighLevelModel<I, List<DetectedObject>> {
    /**
     * Class labels from the dataset used for training.
     */
    protected abstract val classLabels: Map<Int, String>

    /**
     * Returns the detected object for the given image sorted by the score.
     *
     * @param [image] Input image.
     * @param [topK] The number of the detected objects with the highest score to be returned.
     * @return List of [DetectedObject] sorted by score.
     */
    public fun detectObjects(image: I, topK: Int = 5): List<DetectedObject> {
        val objects = predict(image).sortedByDescending { it.probability }
        if (topK > 0) {
            return objects.take(topK)
        }
        return objects
    }
}

/**
 * Base class for object detection models based on EfficientDet architecture.
 */
public abstract class EfficientDetObjectDetectionModelBase<I> : ObjectDetectionModelBase<I>() {

    override fun convert(output: Map<String, Any>): List<DetectedObject> {
        val foundObjects = mutableListOf<DetectedObject>()
        val items = (output[OUTPUT_NAME] as Array<Array<FloatArray>>)[0]

        for (i in items.indices) {
            val probability = items[i][5]
            if (probability != 0.0f) {
                val detectedObject = DetectedObject(
                    classLabel = classLabels[items[i][6].toInt()]!!,
                    probability = probability,
                    // left, bot, right, top
                    xMin = minOf(items[i][2] / internalModel.inputDimensions[1], 1.0f),
                    yMax = minOf(items[i][3] / internalModel.inputDimensions[0], 1.0f),
                    xMax = minOf(items[i][4] / internalModel.inputDimensions[1], 1.0f),
                    yMin = minOf(items[i][1] / internalModel.inputDimensions[0], 1.0f)
                )
                foundObjects.add(detectedObject)
            }
        }
        return foundObjects
    }

    private companion object {
        private const val OUTPUT_NAME = "detections:0"
    }
}

/**
 * Base class for object detection model based on SSD-MobilNet architecture.
 */
public abstract class SSDMobileNetObjectDetectionModelBase<I> : ObjectDetectionModelBase<I>() {

    override fun convert(output: Map<String, Any>): List<DetectedObject> {
        val foundObjects = mutableListOf<DetectedObject>()
        val boxes = (output[OUTPUT_BOXES] as Array<Array<FloatArray>>)[0]
        val classIndices = (output[OUTPUT_CLASSES] as Array<FloatArray>)[0]
        val probabilities = (output[OUTPUT_SCORES] as Array<FloatArray>)[0]
        val numberOfFoundObjects = (output[OUTPUT_NUMBER_OF_DETECTIONS] as FloatArray)[0].toInt()

        for (i in 0 until numberOfFoundObjects) {
            val detectedObject = DetectedObject(
                classLabel = classLabels[classIndices[i].toInt()]!!,
                probability = probabilities[i],
                // top, left, bottom, right
                yMin = boxes[i][0],
                xMin = boxes[i][1],
                yMax = boxes[i][2],
                xMax = boxes[i][3]
            )
            foundObjects.add(detectedObject)
        }
        return foundObjects
    }

    private companion object {
        private const val OUTPUT_BOXES = "detection_boxes:0"
        private const val OUTPUT_CLASSES = "detection_classes:0"
        private const val OUTPUT_SCORES = "detection_scores:0"
        private const val OUTPUT_NUMBER_OF_DETECTIONS = "num_detections:0"
    }
}

/**
 * Base class for object detection model based on SSD architecture.
 */
public abstract class SSDObjectDetectionModelBase<I> : ObjectDetectionModelBase<I>() {

    override fun convert(output: Map<String, Any>): List<DetectedObject> {
        val boxes = (output[OUTPUT_BOXES] as Array<Array<FloatArray>>)[0]
        val classIndices = (output[OUTPUT_LABELS] as Array<LongArray>)[0]
        val probabilities = (output[OUTPUT_SCORES] as Array<FloatArray>)[0]
        val numberOfFoundObjects = boxes.size

        val foundObjects = mutableListOf<DetectedObject>()
        for (i in 0 until numberOfFoundObjects) {
            val detectedObject = DetectedObject(
                classLabel = classLabels[classIndices[i].toInt()]!!,
                probability = probabilities[i],
                // left, bot, right, top
                xMin = boxes[i][0],
                yMin = boxes[i][1],
                xMax = boxes[i][2],
                yMax = boxes[i][3]
            )
            foundObjects.add(detectedObject)
        }
        return foundObjects
    }

    private companion object {
        private const val OUTPUT_BOXES = "bboxes"
        private const val OUTPUT_LABELS = "labels"
        private const val OUTPUT_SCORES = "scores"
    }
}