/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization

import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.text.TextPaint
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult

/**
 * Draw given [detectedObject] on the [Canvas] using [paint] for the bounding box and [labelPaint] for the label.
 */
fun Canvas.drawObject(
    detectedObject: DetectedObject,
    bounds: PreviewImageBounds,
    paint: Paint,
    labelPaint: TextPaint
) {
    val rect = RectF(
        bounds.toViewX(detectedObject.xMin), bounds.toViewY(detectedObject.yMin),
        bounds.toViewX(detectedObject.xMax), bounds.toViewY(detectedObject.yMax)
    )
    val frameWidth = paint.strokeWidth * detectedObject.probability

    drawRect(rect, Paint(paint).apply { strokeWidth = frameWidth })

    val label = "${detectedObject.classLabel} : " + "%.2f".format(detectedObject.probability)
    drawText(label, rect.left, rect.top - labelPaint.fontMetrics.descent - frameWidth / 2, labelPaint)
}

/**
 * Draw given [detectedObjects] on the [Canvas] using [paint] for the bounding box and [labelPaint] for the label.
 */
fun Canvas.drawObjects(
    detectedObjects: List<DetectedObject>,
    bounds: PreviewImageBounds,
    paint: Paint,
    labelPaint: TextPaint
) {
    detectedObjects.forEach { drawObject(it, bounds, paint, labelPaint) }
}

/**
 * Draw given [detectedPose] on the [Canvas] using [landmarkPaint] and [landmarkRadius] for the pose vertices,
 * and [edgePaint] for the pose edges.
 */
fun Canvas.drawPose(
    detectedPose: DetectedPose,
    bounds: PreviewImageBounds,
    landmarkPaint: Paint, edgePaint: Paint,
    landmarkRadius: Float
) {
    detectedPose.edges.forEach { edge ->
        drawLine(
            bounds.toViewX(edge.start.x), bounds.toViewY(edge.start.y),
            bounds.toViewX(edge.end.x), bounds.toViewY(edge.end.y),
            edgePaint
        )
    }

    detectedPose.poseLandmarks.forEach { landmark ->
        drawCircle(bounds.toViewX(landmark.x), bounds.toViewY(landmark.y), landmarkRadius, landmarkPaint)
    }
}

/**
 * Draw given [detectedPoses] on the [Canvas] using [landmarkPaint] and [landmarkRadius] for the pose vertices,
 * [edgePaint] for the poses edges, [objectPaint] for the bounding box and [labelPaint] for the label.
 */
fun Canvas.drawMultiplePoses(
    detectedPoses: MultiPoseDetectionResult,
    bounds: PreviewImageBounds,
    landmarkPaint: Paint,
    edgePaint: Paint,
    objectPaint: Paint,
    labelPaint: TextPaint,
    landmarkRadius: Float
) {
    detectedPoses.multiplePoses.forEach { (detectedObject, detectedPose) ->
        drawPose(detectedPose, bounds, landmarkPaint, edgePaint, landmarkRadius)
        drawObject(detectedObject, bounds, objectPaint, labelPaint)
    }
}

/**
 * Draw given [landmarks] on the [Canvas] using [paint] and [radius].
 */
fun Canvas.drawLandmarks(landmarks: List<Landmark>,
                         bounds: PreviewImageBounds,
                         paint: Paint, radius: Float
) {
    landmarks.forEach { landmark ->
        drawCircle(bounds.toViewX(landmark.xRate), bounds.toViewY(landmark.yRate), radius, paint)
    }
}