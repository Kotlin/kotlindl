/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.objectdetection

import org.jetbrains.kotlinx.dl.api.inference.FlatShape
import java.lang.Float.max
import java.lang.Float.min

/**
 * This data class represents the detected object on the given image.
 *
 * @property [xMin] The minimum X coordinate for the bounding box containing the predicted object.
 * @property [xMax] The maximum X coordinate for the bounding box containing the predicted object.
 * @property [yMin] The minimum Y coordinate for the bounding box containing the predicted object.
 * @property [yMax] The maximum Y coordinate for the bounding box containing the predicted object.
 * @property [probability] The probability of the predicted class.
 * @property [label] The predicted class's name
 */
public data class DetectedObject(
    val xMin: Float,
    val xMax: Float,
    val yMin: Float,
    val yMax: Float,
    val probability: Float,
    val label: String? = null
) : FlatShape<DetectedObject> {
    override fun map(mapping: (Float, Float) -> Pair<Float, Float>): DetectedObject {
        val (x1, y1) = mapping(xMin, yMin)
        val (x2, y2) = mapping(xMax, yMax)
        return DetectedObject(min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2), probability, label)
    }
}
