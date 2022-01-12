package org.jetbrains.kotlinx.dl.api.inference.posedetection

public data class DetectedPose (
    val poseLandmarks: List<PoseLandmark>,
    val edges: List<PoseEdge>
)
