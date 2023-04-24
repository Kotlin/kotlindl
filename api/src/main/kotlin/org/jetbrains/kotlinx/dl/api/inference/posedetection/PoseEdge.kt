/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.posedetection

import org.jetbrains.kotlinx.dl.api.inference.FlatShape

/**
 * This data class represents the line connecting two points [PoseLandmark] of human's pose or edge.
 *
 * @property [label] The predicted label of the edge.
 * @property [probability] The probability of the predicted class.
 * @property [start] The probability of the predicted class.
 * @property [end] The probability of the predicted class.
 */
public data class PoseEdge(
    val start: PoseLandmark,
    val end: PoseLandmark,
    val probability: Float,
    val label: String,
) : FlatShape<PoseEdge> {
    override fun map(mapping: (Float, Float) -> Pair<Float, Float>): PoseEdge {
        return PoseEdge(start.map(mapping), end.map(mapping), probability, label)
    }
}
