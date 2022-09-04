/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.swing

import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult
import org.jetbrains.kotlinx.dl.visualization.swing.ImagePanel.Companion.createImagePanel
import java.awt.*
import java.awt.image.BufferedImage


fun drawDetectedObjects(bufferedImage: BufferedImage, detectedObjects: List<DetectedObject>) {
    showFrame("Filters", createImagePanel(bufferedImage) {
        drawObjects(detectedObjects, bufferedImage.width, bufferedImage.height)
    })
}

fun drawDetectedPose(bufferedImage: BufferedImage, detectedPose: DetectedPose) {
    showFrame("Filters", createImagePanel(bufferedImage) {
        drawPose(detectedPose, bufferedImage.width, bufferedImage.height)
    })
}

fun drawMultiPoseLandMarks(bufferedImage: BufferedImage,
                           multiPoseDetectionResult: MultiPoseDetectionResult
) {
    showFrame("Landmarks", createImagePanel(bufferedImage) {
        drawMultiplePoses(multiPoseDetectionResult, bufferedImage.width, bufferedImage.height)
    })
}

fun drawLandMarks(bufferedImage: BufferedImage, landmarks: List<Landmark>) {
    showFrame("Landmarks", createImagePanel(bufferedImage) {
        drawLandmarks(landmarks, bufferedImage.width, bufferedImage.height)
    })
}

private fun Graphics2D.drawObject(detectedObject: DetectedObject,
                                  width: Int,
                                  height: Int,
                                  objectColor: Color = Color.RED,
                                  labelColor: Color = Color.ORANGE
) {
    val top = detectedObject.yMin * height
    val left = detectedObject.xMin * width
    val bottom = detectedObject.yMax * height
    val right = detectedObject.xMax * width
    // left, bot, right, top

    // y = columnIndex
    // x = rowIndex
    val yRect = top
    val xRect = left
    color = objectColor
    stroke = BasicStroke(6f * detectedObject.probability)
    drawRect(xRect.toInt(), yRect.toInt(), (right - left).toInt(), (bottom - top).toInt())


    color = labelColor
    font = Font("Courier New", 1, 17)
    drawString(" ${detectedObject.classLabel} : ${detectedObject.probability}", xRect.toInt(), yRect.toInt() - 8)
}

private fun Graphics2D.drawObjects(detectedObjects: List<DetectedObject>, width: Int, height: Int) {
    detectedObjects.forEach { drawObject(it, width, height) }
}

private fun Graphics2D.drawPose(detectedPose: DetectedPose, width: Int, height: Int,
                                landmarkColor: Color = Color.RED,
                                edgeColor: Color = Color.MAGENTA
) {
    color = landmarkColor
    stroke = BasicStroke(3f)
    detectedPose.poseLandmarks.forEach { landmark ->
        val xLM = width * landmark.x
        val yLM = height * landmark.y

        drawOval(xLM.toInt(), yLM.toInt(), 3, 3)
    }

    color = edgeColor
    stroke = BasicStroke(2f)
    detectedPose.edges.forEach { edge ->
        val x1 = width * edge.start.x
        val y1 = height * edge.start.y
        val x2 = width * edge.end.x
        val y2 = height * edge.end.y

        drawLine(x1.toInt(), y1.toInt(), x2.toInt(), y2.toInt())
    }
}

private fun Graphics2D.drawMultiplePoses(multiPoseDetectionResult1: MultiPoseDetectionResult,
                                         width: Int,
                                         height: Int
) {
    multiPoseDetectionResult1.multiplePoses.forEachIndexed { i, (detectedObject, detectedPose) ->
        drawPose(
            detectedPose, width, height,
            Color((6 - i) * 40, i * 20, i * 10),
            Color.MAGENTA
        )
        drawObject(detectedObject, width, height, Color.ORANGE)
    }
}


private fun Graphics2D.drawLandmarks(landmarks: List<Landmark>, width: Int, height: Int) {
    color = Color.RED
    stroke = BasicStroke(3f)
    landmarks.forEach { landmark ->
        val xLM = width * landmark.xRate
        val yLM = height * landmark.yRate

        drawOval(xLM.toInt(), yLM.toInt(), 2, 2)
    }
}