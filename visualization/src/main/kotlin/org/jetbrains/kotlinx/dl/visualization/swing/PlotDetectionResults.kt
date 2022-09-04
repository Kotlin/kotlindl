/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.swing

import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult
import java.awt.*
import java.awt.image.BufferedImage


fun drawDetectedObjects(bufferedImage: BufferedImage, detectedObjects: List<DetectedObject>) {
    showFrame("Filters", DetectedObjectJPanel(bufferedImage, detectedObjects))
}

fun drawDetectedPose(bufferedImage: BufferedImage, detectedPose: DetectedPose) {
    showFrame("Filters", DetectedPoseJPanel(bufferedImage, detectedPose))
}

fun drawMultiPoseLandMarks(bufferedImage: BufferedImage,
                           multiPoseDetectionResult: MultiPoseDetectionResult
) {
    showFrame("Landmarks", MultiPosePointsJPanel(bufferedImage, multiPoseDetectionResult))
}

fun drawLandMarks(bufferedImage: BufferedImage, landmarks: List<Landmark>) {
    showFrame("Landmarks", LandMarksJPanel(bufferedImage, landmarks))
}

class MultiPosePointsJPanel(
    bufferedImage: BufferedImage,
    private val multiPoseDetectionResult: MultiPoseDetectionResult
) : ImagePanel(bufferedImage) {

    override fun paint(graphics: Graphics) {
        super.paint(graphics)
        graphics.drawImage(bufferedImage, 0, 0, null)

        multiPoseDetectionResult.multiplePoses.forEachIndexed { i, it ->
            val onePosePoints = it.second.poseLandmarks

            for (j in onePosePoints.indices) {
                val xLM = (size.width) * (onePosePoints[j].x)
                val yLM = (size.height) * (onePosePoints[j].y)

                graphics as Graphics2D
                val stroke1: Stroke = BasicStroke(3f)
                graphics.setColor(Color((6 - i) * 40, i * 20, i * 10))
                graphics.stroke = stroke1
                graphics.drawOval(xLM.toInt(), yLM.toInt(), 3, 3)
            }

            val onePoseEdges = it.second.edges
            for (j in onePoseEdges.indices) {
                val x1 = (size.width) * (onePoseEdges[j].start.x)
                val y1 = (size.height) * (onePoseEdges[j].start.y)
                val x2 = (size.width) * (onePoseEdges[j].end.x)
                val y2 = (size.height) * (onePoseEdges[j].end.y)

                graphics as Graphics2D
                val stroke1: Stroke = BasicStroke(2f)
                graphics.setColor(Color.MAGENTA)
                graphics.stroke = stroke1
                graphics.drawLine(x1.toInt(), y1.toInt(), x2.toInt(), y2.toInt())
            }

            val detectedObject = it.first

            val top = detectedObject.yMin * bufferedImage.height
            val left = detectedObject.xMin * bufferedImage.width
            val bottom = detectedObject.yMax * bufferedImage.height
            val right = detectedObject.xMax * bufferedImage.width
            // left, bot, right, top

            // y = columnIndex
            // x = rowIndex
            val yRect = top
            val xRect = left
            graphics as Graphics2D
            val stroke: Stroke = BasicStroke(6f * detectedObject.probability)
            graphics.setColor(Color.ORANGE)
            graphics.stroke = stroke
            graphics.drawRect(xRect.toInt(), yRect.toInt(), (right - left).toInt(), (bottom - top).toInt())
        }
    }
}

class DetectedPoseJPanel(
    bufferedImage: BufferedImage,
    private val detectedPose: DetectedPose
) : ImagePanel(bufferedImage) {

    override fun paint(graphics: Graphics) {
        super.paint(graphics)
        graphics.drawImage(bufferedImage, 0, 0, null)

        detectedPose.poseLandmarks.forEach {
            val xLM = (size.width) * (it.x)
            val yLM = (size.height) * (it.y)

            graphics as Graphics2D
            val stroke1: Stroke = BasicStroke(3f)
            graphics.setColor(Color.RED)
            graphics.stroke = stroke1
            graphics.drawOval(xLM.toInt(), yLM.toInt(), 3, 3)
        }

        val onePoseEdges = detectedPose.edges
        for (j in onePoseEdges.indices) {
            val x1 = (size.width) * (onePoseEdges[j].start.x)
            val y1 = (size.height) * (onePoseEdges[j].start.y)
            val x2 = (size.width) * (onePoseEdges[j].end.x)
            val y2 = (size.height) * (onePoseEdges[j].end.y)

            graphics as Graphics2D
            val stroke1: Stroke = BasicStroke(2f)
            graphics.setColor(Color.MAGENTA)
            graphics.stroke = stroke1
            graphics.drawLine(x1.toInt(), y1.toInt(), x2.toInt(), y2.toInt())
        }
    }
}

class LandMarksJPanel(bufferedImage: BufferedImage, private val landmarks: List<Landmark>) :
    ImagePanel(bufferedImage) {

    override fun paint(graphics: Graphics) {
        super.paint(graphics)

        graphics.drawImage(bufferedImage, 0, 0, null)

        for (i in landmarks.indices) {
            val xLM = size.width * landmarks[i].xRate
            val yLM = size.height * landmarks[i].yRate

            graphics as Graphics2D
            val stroke1: Stroke = BasicStroke(3f)
            graphics.setColor(Color.RED)
            graphics.stroke = stroke1
            graphics.drawOval(xLM.toInt(), yLM.toInt(), 2, 2)
        }
    }
}

class DetectedObjectJPanel(
    bufferedImage: BufferedImage,
    private val detectedObjects: List<DetectedObject>
) : ImagePanel(bufferedImage) {

    override fun paint(graphics: Graphics) {
        super.paint(graphics)

        graphics.drawImage(bufferedImage, 0, 0, null)

        detectedObjects.forEach {
            val pixelWidth = 1
            val pixelHeight = 1

            val top = it.yMax * bufferedImage.height * pixelHeight
            val left = it.xMin * bufferedImage.width * pixelWidth
            val bottom = it.yMin * bufferedImage.height * pixelHeight
            val right = it.xMax * bufferedImage.width * pixelWidth
            // left, bot, right, top

            // y = columnIndex
            // x = rowIndex
            val yRect = bottom
            val xRect = left
            graphics.color = Color.ORANGE
            graphics.font = Font("Courier New", 1, 17)
            graphics.drawString(" ${it.classLabel} : ${it.probability}", xRect.toInt(), yRect.toInt() - 8)

            graphics as Graphics2D
            val stroke1: Stroke = BasicStroke(10f * it.probability)
            graphics.setColor(Color.RED)
            graphics.stroke = stroke1
            graphics.drawRect(xRect.toInt(), yRect.toInt(), (right - left).toInt(), (top - bottom).toInt())
        }
    }
}