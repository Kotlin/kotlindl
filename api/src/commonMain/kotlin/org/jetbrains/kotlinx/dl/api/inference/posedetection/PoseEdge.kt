package org.jetbrains.kotlinx.dl.api.inference.posedetection

/**
 * This data class represents the line connecting two points [PoseLandmark] of human's pose.
 *
 * @property [poseEdgeLabel] The predicted pose edge label.
 * @property [probability] The probability of the predicted class.
 * @property [start] The probability of the predicted class.
 * @property [end] The probability of the predicted class.
 */
public data class PoseEdge(
    val poseEdgeLabel: String,
    val probability: Float,
    val start: PoseLandmark,
    val end: PoseLandmark,
)
