/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference.objectdetection

import ai.onnxruntime.OrtSession
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxHighLevelModel
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.get2DFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.getFloatArray

/**
 * Base class for object detection models.
 */
public abstract class ObjectDetectionModelBase<I>(override val modelKindDescription: String? = null) :
    OnnxHighLevelModel<I, List<DetectedObject>> {
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
public abstract class EfficientDetObjectDetectionModelBase<I>(modelType: String? = null) :
    ObjectDetectionModelBase<I>(modelType) {

    override fun convert(output: OrtSession.Result): List<DetectedObject> {
        val foundObjects = mutableListOf<DetectedObject>()
        val items = output.get2DFloatArray(OUTPUT_NAME)

        for (i in items.indices) {
            val probability = items[i][5]
            if (probability != 0.0f) {
                val detectedObject = DetectedObject(
                    xMin = minOf(items[i][2] / internalModel.inputDimensions[1], 1.0f),
                    xMax = minOf(items[i][4] / internalModel.inputDimensions[1], 1.0f),
                    // left, bot, right, top
                    yMin = minOf(items[i][1] / internalModel.inputDimensions[0], 1.0f),
                    yMax = minOf(items[i][3] / internalModel.inputDimensions[0], 1.0f),
                    probability = probability,
                    label = classLabels[items[i][6].toInt()]
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
 * Base class for object detection model based on SSD architecture.
 * @param [metadata] SSD-like model metadata. Used for decoding the output.
 */
public abstract class SSDLikeModelBase<I>(
    protected val metadata: SSDLikeModelMetadata,
    modelType: String? = null
) : ObjectDetectionModelBase<I>(modelType) {
    override fun convert(output: OrtSession.Result): List<DetectedObject> {
        val boxes = output.get2DFloatArray(metadata.outputBoxesName)
        val classIndices = output.getFloatArray(metadata.outputClassesName)
        val probabilities = output.getFloatArray(metadata.outputScoresName)
        val numberOfFoundObjects = boxes.size

        val foundObjects = mutableListOf<DetectedObject>()
        for (i in 0 until numberOfFoundObjects) {
            val detectedObject = DetectedObject(
                xMin = boxes[i][metadata.xMinIdx],
                xMax = boxes[i][metadata.xMinIdx + 2],
                // left, bot, right, top
                yMin = boxes[i][metadata.yMinIdx],
                yMax = boxes[i][metadata.yMinIdx + 2],
                probability = probabilities[i],
                label = classLabels[classIndices[i].toInt()]
            )
            foundObjects.add(detectedObject)
        }
        return foundObjects
    }
}

/**
 * This class aggregates the metadata of the SSD-like model used for decoding the output.
 * The class is exists mostly for reducing code duplication.
 */
public data class SSDLikeModelMetadata(
    /**
     * The name of the output tensor with the bounding boxes.
     */
    public val outputBoxesName: String,
    /**
     * The name of the output tensor with the class indices.
     */
    public val outputClassesName: String,
    /**
     * The name of the output tensor with classes confidence scores.
     */
    public val outputScoresName: String,
    /**
     * The index of the yMin coordinate in the bounding box encoded representation.
     */
    public val yMinIdx: Int,
    /**
     * The index of the xMin coordinate in the bounding box encoded representation.
     */
    public val xMinIdx: Int
)
