/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.posedetection

import org.jetbrains.kotlinx.dl.api.inference.FlatShape

/**
 * This data class represents the human's pose detected on the given image.
 *
 * @property [landmarks] The list of detected [PoseLandmark]s for the given image.
 * @property [edges] The list of edges connecting the detected [PoseLandmark]s.
 */
public data class DetectedPose(
    val landmarks: List<PoseLandmark>,
    val edges: List<PoseEdge>
) : FlatShape<DetectedPose> {
    override fun map(mapping: (Float, Float) -> Pair<Float, Float>): DetectedPose {
        return DetectedPose(landmarks.map { it.map(mapping) }, edges.map { it.map(mapping) })
    }
}
