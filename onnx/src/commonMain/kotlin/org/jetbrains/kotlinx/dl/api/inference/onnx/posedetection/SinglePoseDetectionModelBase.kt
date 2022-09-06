/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection

import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxHighLevelModel
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.PoseEdge
import org.jetbrains.kotlinx.dl.api.inference.posedetection.PoseLandmark
import kotlin.math.min

/**
 * Base class for pose detection models for detecting a single pose per image.
 */
public abstract class SinglePoseDetectionModelBase<I> : OnnxHighLevelModel<I, DetectedPose> {

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

    override fun convert(output: Map<String, Any>): DetectedPose {
        val rawPoseLandMarks = (output[outputName] as Array<Array<Array<FloatArray>>>)[0][0]

        val foundPoseLandmarks = mutableListOf<PoseLandmark>()
        for (i in rawPoseLandMarks.indices) {
            val poseLandmark = PoseLandmark(
                poseLandmarkLabel = keyPointsLabels[i]!!,
                x = rawPoseLandMarks[i][1],
                y = rawPoseLandMarks[i][0],
                probability = rawPoseLandMarks[i][2]
            )
            foundPoseLandmarks.add(i, poseLandmark)
        }

        val foundPoseEdges = buildPoseEdges(foundPoseLandmarks, edgeKeyPoints)

        return DetectedPose(foundPoseLandmarks, foundPoseEdges)
    }

    /**
     * Detects a pose for the given [image].
     * @param [image] input image.
     */
    public fun detectPose(image: I): DetectedPose = predict(image)
}

internal fun buildPoseEdges(foundPoseLandmarks: List<PoseLandmark>, edgeKeyPoints: List<Pair<Int, Int>>): List<PoseEdge> {
    val foundPoseEdges = mutableListOf<PoseEdge>()
    edgeKeyPoints.forEach {
        val startPoint = foundPoseLandmarks[it.first]
        val endPoint = foundPoseLandmarks[it.second]
        foundPoseEdges.add(
            PoseEdge(
                poseEdgeLabel = startPoint.poseLandmarkLabel + "_" + endPoint.poseLandmarkLabel,
                probability = min(startPoint.probability, endPoint.probability),
                start = startPoint,
                end = endPoint
            )
        )
    }
    return foundPoseEdges
}