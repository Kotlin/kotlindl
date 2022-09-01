/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection

import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxPreTrainedModel
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult
import org.jetbrains.kotlinx.dl.api.inference.posedetection.PoseLandmark
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation

/**
 * Base class for pose detection models for detecting multiple poses per image.
 */
public abstract class MultiPoseDetectionModelBase<I> : OnnxPreTrainedModel<I, MultiPoseDetectionResult> {
    /**
     * Name of the output tensor.
     */
    protected abstract val outputName: String

    /**
     * Dictionary that maps from joint names to keypoint indices.
     */
    protected abstract val keyPointsLabels: Map<Int, String>

    /**
     * Pairs of points which define body edges.
     */
    protected abstract val edgeKeyPoints: List<Pair<Int, Int>>

    override fun convert(output: Map<String, Any>): MultiPoseDetectionResult {
        val rawPoseLandMarks = (output[outputName] as Array<Array<FloatArray>>)[0]

        val poses = rawPoseLandMarks.map { floats ->
            val foundPoseLandmarks = mutableListOf<PoseLandmark>()

            for (keyPointIdx in 0..16) {
                val poseLandmark = PoseLandmark(
                    poseLandmarkLabel = keyPointsLabels[keyPointIdx]!!,
                    x = floats[3 * keyPointIdx + 1],
                    y = floats[3 * keyPointIdx],
                    probability = floats[3 * keyPointIdx + 2]
                )
                foundPoseLandmarks.add(poseLandmark)
            }

            // [ymin, xmin, ymax, xmax, score]
            val detectedObject = DetectedObject(
                classLabel = CLASS_LABEL,
                probability = floats[55],
                yMin = floats[53],
                xMin = floats[52],
                yMax = floats[51],
                xMax = floats[54]
            )

            val foundPoseEdges = buildPoseEdges(foundPoseLandmarks, edgeKeyPoints)
            val detectedPose = DetectedPose(foundPoseLandmarks, foundPoseEdges)

            detectedObject to detectedPose
        }

        return MultiPoseDetectionResult(poses)
    }

    /**
     * Detects poses for the given [image] with the given [confidence].
     * @param [confidence] confidence value to use
     */
    public fun detectPoses(image: I, confidence: Float = 0.1f): MultiPoseDetectionResult {
        val result = predict(image)
        val filteredPoses = result.multiplePoses.filter { (detectedObject, _) ->
            detectedObject.probability > confidence
        }
        return MultiPoseDetectionResult(filteredPoses)
    }

    private companion object {
        private const val CLASS_LABEL = "person"
    }
}