/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.objectdetection

/**
 * This data class represents the detected object on the given image.
 *
 * @property [classLabel] The predicted class's name
 * @property [probability] The probability of the predicted class.
 * @property [xMax] The maximum X coordinate for the bounding box containing the predicted object.
 * @property [xMin] The minimum X coordinate for the bounding box containing the predicted object.
 * @property [yMax] The maximum Y coordinate for the bounding box containing the predicted object.
 * @property [yMin] The minimum Y coordinate for the bounding box containing the predicted object.
 */
public data class DetectedObject(
    val classLabel: String,
    val probability: Float,
    val xMax: Float,
    val xMin: Float,
    val yMax: Float,
    val yMin: Float
)
