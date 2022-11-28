/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.posedetection

import org.jetbrains.kotlinx.dl.api.inference.FlatShape

/**
 * This data class represents one point of the detected human's pose.
 *
 * @property [x] The value of `x` coordinate.
 * @property [y] The value of `y` coordinate.
 * @property [probability] The probability of the predicted class.
 * @property [label] The predicted pose landmark label.
 */
public data class PoseLandmark(
    val x: Float,
    val y: Float,
    val probability: Float,
    val label: String,
) : FlatShape<PoseLandmark> {
    override fun map(mapping: (Float, Float) -> Pair<Float, Float>): PoseLandmark {
        val (x1, y1) = mapping(x, y)
        return PoseLandmark(x1, y1, probability, label)
    }
}
