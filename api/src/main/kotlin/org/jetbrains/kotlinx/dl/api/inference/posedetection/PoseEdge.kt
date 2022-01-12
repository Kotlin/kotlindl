package org.jetbrains.kotlinx.dl.api.inference.posedetection

/**
 * @property [poseEdgeLabel] The predicted pose edge label.
 * @property [probability] The probability of the predicted class.
 */
public data class PoseEdge(
    val poseEdgeLabel: String,
    val probability: Float,
    val start: PoseLandmark,
    val end: PoseLandmark,
)
