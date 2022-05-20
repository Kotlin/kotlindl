/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.posedetection

/**
 * This data class represents the human's pose detected on the given image.
 *
 * @property [poseLandmarks] The list of detected [PoseLandmark]s for the given image.
 * @property [edges] The list of edges connecting the detected [PoseLandmark]s.
 */
public data class DetectedPose(
    val poseLandmarks: List<PoseLandmark>,
    val edges: List<PoseEdge>
)
