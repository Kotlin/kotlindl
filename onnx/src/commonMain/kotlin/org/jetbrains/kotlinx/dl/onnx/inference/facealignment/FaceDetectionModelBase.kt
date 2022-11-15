/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference.facealignment

import ai.onnxruntime.OrtSession
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxHighLevelModel
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.get2DFloatArray
import java.lang.Float.min
import kotlin.math.max

/**
 * Base class for face detection models.
 */
public abstract class FaceDetectionModelBase<I>(override val modelKindDescription: String? = null) :
    OnnxHighLevelModel<I, List<DetectedObject>> {

    override fun convert(output: OrtSession.Result): List<DetectedObject> {
        val scores = output.get2DFloatArray("scores")
        val boxes = output.get2DFloatArray("boxes")

        if (scores.isEmpty()) return emptyList()

        val result = mutableListOf<DetectedObject>()
        for (classIndex in 1 until scores[0].size) {
            for ((box, classScores) in boxes.zip(scores)) {
                val score = classScores[classIndex]
                if (score > THRESHOLD) {
                    result.add(DetectedObject(box[0], box[2], box[1], box[3], score))
                }
            }
        }
        return result
    }

    /**
     * Detects [topK] faces on the given [image]. If [topK] is negative all detected faces are returned.
     * @param [iouThreshold] threshold IoU value for the non-maximum suppression applied during postprocessing
     */
    public fun detectFaces(image: I, topK: Int = 5, iouThreshold: Float = 0.5f): List<DetectedObject> {
        val detectedObjects = predict(image)
        return suppressNonMaxBoxes(detectedObjects, topK, iouThreshold)
    }

    public companion object {
        private const val THRESHOLD = 0.7
        private const val EPS = Float.MIN_VALUE

        /**
         * Performs non-maximum suppression to filter out boxes with the IoU greater than threshold.
         * @param [boxes] boxes to filter
         * @param [topK] how many boxes to include in the result. Negative or zero means to include everything.
         * @param [threshold] threshold IoU value
         */
        public fun suppressNonMaxBoxes(
            boxes: List<DetectedObject>,
            topK: Int = -1,
            threshold: Float = 0.5f
        ): List<DetectedObject> {
            val sortedBoxes = boxes.toMutableList().apply { sortByDescending { it.probability } }
            val result = mutableListOf<DetectedObject>()
            while (sortedBoxes.isNotEmpty()) {
                val box = sortedBoxes.removeFirst()
                result.add(box)
                if (topK > 0 && result.size >= topK) break

                sortedBoxes.removeIf { iou(box, it) >= threshold }
            }
            return result
        }

        /**
         * Computes the intersection over union value for the [box1] and [box2].
         */
        public fun iou(box1: DetectedObject, box2: DetectedObject): Float {
            val xMin = max(box1.xMin, box2.xMin)
            val yMin = max(box1.yMin, box2.yMin)
            val xMax = min(box1.xMax, box2.xMax)
            val yMax = min(box1.yMax, box2.yMax)
            val overlap = (xMax - xMin) * (yMax - yMin)
            return overlap / (box1.area() + box2.area() - overlap + EPS)
        }

        private fun DetectedObject.area() = (xMax - xMin) * (yMax - yMin)
    }
}