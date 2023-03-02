package org.jetbrains.kotlinx.dl.api.inference.posedetection

import org.jetbrains.kotlinx.dl.api.inference.FlatShape

/**
 * This data class represents the line connecting two points [PoseLandmark] of human's pose.
 *
 * @property [label] The predicted pose edge label.
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
