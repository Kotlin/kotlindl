/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.swing

import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult
import org.jetbrains.kotlinx.dl.visualization.swing.ImagePanel.Companion.createImagePanel
import java.awt.*
import java.awt.geom.Ellipse2D
import java.awt.geom.Line2D
import java.awt.geom.Rectangle2D
import java.awt.image.BufferedImage
import javax.swing.JPanel


/**
 * Create a component with the given [bufferedImage] and [detectedObjects] drawn on top of it.
 */
fun createDetectedObjectsPanel(
    bufferedImage: BufferedImage,
    detectedObjects: List<DetectedObject>
): JPanel = createImagePanel(bufferedImage) {
    drawObjects(detectedObjects, bufferedImage.width, bufferedImage.height)
}

/**
 * Create a component with the given [bufferedImage] and [detectedPose] drawn on top of it.
 */
fun createDetectedPosePanel(
    bufferedImage: BufferedImage,
    detectedPose: DetectedPose
): JPanel = createImagePanel(bufferedImage) {
    drawPose(detectedPose, bufferedImage.width, bufferedImage.height)
}

/**
 * Create a component with the given [bufferedImage] and [multiPoseDetectionResult] drawn on top of it.
 */
fun createMultipleDetectedPosesPanel(
    bufferedImage: BufferedImage,
    multiPoseDetectionResult: MultiPoseDetectionResult
): JPanel = createImagePanel(bufferedImage) {
    drawMultiplePoses(multiPoseDetectionResult, bufferedImage.width, bufferedImage.height)
}

/**
 * Create a component with the given [bufferedImage] and [landmarks] drawn on top of it.
 */
fun createDetectedLandmarksPanel(
    bufferedImage: BufferedImage, landmarks: List<Landmark>
): JPanel = createImagePanel(bufferedImage) {
    drawLandmarks(landmarks, bufferedImage.width, bufferedImage.height)
}

private fun Graphics2D.drawObject(
    detectedObject: DetectedObject,
    width: Int,
    height: Int,
    objectColor: Color = Color.RED,
    labelColor: Color = Color.ORANGE
) {
    setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)

    val x = detectedObject.xMin * width
    val y = detectedObject.yMin * height

    val frameWidth = 6f * detectedObject.probability
    color = objectColor
    stroke = BasicStroke(frameWidth)
    draw(Rectangle2D.Float(x, y, detectedObject.xMax * width - x, detectedObject.yMax * height - y))

    if (detectedObject.label != null) {
        val label = "${detectedObject.label} : " + "%.2f".format(detectedObject.probability)
        color = labelColor
        font = font.deriveFont(Font.BOLD)
        drawString(label, x, y - fontMetrics.maxDescent - frameWidth / 2)
    }
}

private fun Graphics2D.drawObjects(detectedObjects: List<DetectedObject>, width: Int, height: Int) {
    setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    detectedObjects.forEach { drawObject(it, width, height) }
}

private fun Graphics2D.drawPose(
    detectedPose: DetectedPose, width: Int, height: Int,
    landmarkColor: Color = Color.RED,
    edgeColor: Color = Color.MAGENTA
) {
    setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)

    color = edgeColor
    stroke = BasicStroke(2f)
    detectedPose.edges.forEach { (start, end, _, _) ->
        draw(
            Line2D.Float(
                width * start.x, height * start.y,
                width * end.x, height * end.y
            )
        )
    }

    val r = 3.0f
    color = landmarkColor
    detectedPose.landmarks.forEach { (x, y, _, _) ->
        fill(Ellipse2D.Float(width * x - r, height * y - r, 2 * r, 2 * r))
    }
}

private fun Graphics2D.drawMultiplePoses(
    multiPoseDetectionResult1: MultiPoseDetectionResult,
    width: Int,
    height: Int
) {
    setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    multiPoseDetectionResult1.poses.forEachIndexed { i, (detectedObject, detectedPose) ->
        drawPose(
            detectedPose, width, height,
            Color((6 - i) * 40, i * 20, i * 10),
            Color.MAGENTA
        )
        drawObject(detectedObject, width, height, Color.ORANGE)
    }
}


private fun Graphics2D.drawLandmarks(landmarks: List<Landmark>, width: Int, height: Int) {
    setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)

    val r = 3.0f
    color = Color.RED
    landmarks.forEach { (x, y) ->
        fill(Ellipse2D.Float(width * x - r, height * y - r, 2 * r, 2 * r))
    }
}