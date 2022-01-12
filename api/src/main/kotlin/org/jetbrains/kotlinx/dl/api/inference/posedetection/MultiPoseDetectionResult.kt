package org.jetbrains.kotlinx.dl.api.inference.posedetection

import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject

public data class MultiPoseDetectionResult (
    val multiplePoses: MutableList<Pair<DetectedObject, DetectedPose>>
)
