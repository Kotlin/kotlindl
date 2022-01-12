package org.jetbrains.kotlinx.dl.api.inference.posedetection

/**
 * @property [poseLandmarkLabel] The predicted pose landmark label.
 * @property [probability] The probability of the predicted class.
 */
public data class PoseLandmark (
    val poseLandmarkLabel: String,
    val probability: Float,
    val x: Float,
    val y: Float,
)
