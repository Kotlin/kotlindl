/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
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
 *
 * If the preview image coordinates do not match the [Canvas] coordinates,
 * [bounds] of the image preview should be provided.
 *
 * @see [PreviewImageBounds]
 */
fun Canvas.drawObject(
    detectedObject: DetectedObject,
    paint: Paint,
    labelPaint: TextPaint,
    bounds: PreviewImageBounds = bounds()
) {
    val rect = RectF(
        bounds.toViewX(detectedObject.xMin), bounds.toViewY(detectedObject.yMin),
        bounds.toViewX(detectedObject.xMax), bounds.toViewY(detectedObject.yMax)
    )
    val frameWidth = paint.strokeWidth * detectedObject.probability

    drawRect(rect, Paint(paint).apply { strokeWidth = frameWidth })

    if (detectedObject.label != null) {
        val label = "${detectedObject.label} : " + "%.2f".format(detectedObject.probability)
        drawText(label, rect.left, rect.top - labelPaint.fontMetrics.descent - frameWidth / 2, labelPaint)
    }
}

/**
 * Draw given [detectedObjects] on the [Canvas] using [paint] for the bounding box and [labelPaint] for the label.
 *
 * If the preview image coordinates do not match the [Canvas] coordinates,
 * [bounds] of the image preview should be provided.
 *
 * @see [PreviewImageBounds]
 */
fun Canvas.drawObjects(
    detectedObjects: List<DetectedObject>,
    paint: Paint,
    labelPaint: TextPaint,
    bounds: PreviewImageBounds = bounds()
) {
    detectedObjects.forEach { drawObject(it, paint, labelPaint, bounds) }
}

/**
 * Draw given [detectedPose] on the [Canvas] using [landmarkPaint] and [landmarkRadius] for the pose vertices,
 * and [edgePaint] for the pose edges.
 *
 * If the preview image coordinates do not match the [Canvas] coordinates,
 * [bounds] of the image preview should be provided.
 *
 * @see [PreviewImageBounds]
 */
fun Canvas.drawPose(
    detectedPose: DetectedPose,
    landmarkPaint: Paint,
    edgePaint: Paint, landmarkRadius: Float,
    bounds: PreviewImageBounds = bounds()
) {
    detectedPose.edges.forEach { edge ->
        drawLine(
            bounds.toViewX(edge.start.x), bounds.toViewY(edge.start.y),
            bounds.toViewX(edge.end.x), bounds.toViewY(edge.end.y),
            edgePaint
        )
    }

    detectedPose.landmarks.forEach { landmark ->
        drawCircle(bounds.toViewX(landmark.x), bounds.toViewY(landmark.y), landmarkRadius, landmarkPaint)
    }
}

/**
 * Draw given [detectedPoses] on the [Canvas] using [landmarkPaint] and [landmarkRadius] for the pose vertices,
 * [edgePaint] for the poses edges, [objectPaint] for the bounding box and [labelPaint] for the label.
 *
 * If the preview image coordinates do not match the [Canvas] coordinates,
 * [bounds] of the image preview should be provided.
 *
 * @see [PreviewImageBounds]
 */
fun Canvas.drawMultiplePoses(
    detectedPoses: MultiPoseDetectionResult,
    landmarkPaint: Paint,
    edgePaint: Paint,
    objectPaint: Paint,
    labelPaint: TextPaint,
    landmarkRadius: Float,
    bounds: PreviewImageBounds = bounds()
) {
    detectedPoses.poses.forEach { (detectedObject, detectedPose) ->
        drawPose(detectedPose, landmarkPaint, edgePaint, landmarkRadius, bounds)
        drawObject(detectedObject, objectPaint, labelPaint, bounds)
    }
}

/**
 * Draw given [landmarks] on the [Canvas] using [paint] and [radius].
 *
 * If the preview image coordinates do not match the [Canvas] coordinates,
 * [bounds] of the image preview should be provided.
 *
 * @see [PreviewImageBounds]
 */
fun Canvas.drawLandmarks(landmarks: List<Landmark>,
                         paint: Paint, radius: Float,
                         bounds: PreviewImageBounds = bounds()
) {
    landmarks.forEach { landmark -> drawLandmark(landmark, paint, radius, bounds) }
}

/**
 * Draw a given [landmark] on the [Canvas] using [paint] and [radius].
 *
 * If the preview image coordinates do not match the [Canvas] coordinates,
 * [bounds] of the image preview should be provided.
 *
 * @see [PreviewImageBounds]
 */
fun Canvas.drawLandmark(landmark: Landmark, paint: Paint, radius: Float,
                        bounds: PreviewImageBounds = bounds()
) {
    drawCircle(bounds.toViewX(landmark.x), bounds.toViewY(landmark.y), radius, paint)
}

/**
 * Create [PreviewImageBounds] originating in the top-left corner of this [Canvas] object and matching its dimensions.
 */
fun Canvas.bounds() = PreviewImageBounds(0f, 0f, width.toFloat(), height.toFloat())