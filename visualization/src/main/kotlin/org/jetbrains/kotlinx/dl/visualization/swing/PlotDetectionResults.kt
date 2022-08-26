/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.swing

import org.jetbrains.kotlinx.dl.api.extension.get3D
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.*
import java.awt.image.BufferedImage
import javax.swing.JPanel


fun drawDetectedObjects(dst: FloatArray, imageShape: ImageShape, detectedObjects: List<DetectedObject>) {
    showFrame("Filters", DetectedObjectJPanel(dst, imageShape, detectedObjects))
}

fun drawDetectedPose(dst: FloatArray, imageShape: ImageShape, detectedPose: DetectedPose) {
    showFrame("Filters", DetectedPoseJPanel(dst, imageShape, detectedPose))
}

fun drawMultiPoseLandMarks(dst: FloatArray,
                           imageShape: ImageShape,
                           multiPoseDetectionResult: MultiPoseDetectionResult
) {
    showFrame("Landmarks", MultiPosePointsJPanel(dst, imageShape, multiPoseDetectionResult))
}

fun drawLandMarks(dst: FloatArray, imageShape: ImageShape, landmarks: List<Landmark>) {
    showFrame("Landmarks", LandMarksJPanel(dst, imageShape, landmarks))
}

class MultiPosePointsJPanel(
    val image: FloatArray,
    val imageShape: ImageShape,
    private val multiPoseDetectionResult: MultiPoseDetectionResult
) : JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape)

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

            val top = detectedObject.yMin * imageShape.height!!
            val left = detectedObject.xMin * imageShape.width!!
            val bottom = detectedObject.yMax * imageShape.height!!
            val right = detectedObject.xMax * imageShape.width!!
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

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }

    override fun getMinimumSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
}

class DetectedPoseJPanel(
    val image: FloatArray,
    val imageShape: ImageShape,
    private val detectedPose: DetectedPose
) : JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape)

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

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }

    override fun getMinimumSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
}

class LandMarksJPanel(val image: FloatArray, val imageShape: ImageShape, private val landmarks: List<Landmark>) :
    JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape)

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

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }

    override fun getMinimumSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
}

class DetectedObjectJPanel(
    val image: FloatArray,
    val imageShape: ImageShape,
    private val detectedObjects: List<DetectedObject>
) : JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape)

    override fun paint(graphics: Graphics) {
        super.paint(graphics)

        graphics.drawImage(bufferedImage, 0, 0, null)

        detectedObjects.forEach {
            val pixelWidth = 1
            val pixelHeight = 1

            val top = it.yMax * imageShape.height!! * pixelHeight
            val left = it.xMin * imageShape.width!! * pixelWidth
            val bottom = it.yMin * imageShape.height!! * pixelHeight
            val right = it.xMax * imageShape.width!! * pixelWidth
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

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }

    override fun getMinimumSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
}

private fun FloatArray.toBufferedImage(imageShape: ImageShape): BufferedImage {
    val result = BufferedImage(imageShape.width!!.toInt(), imageShape.height!!.toInt(), BufferedImage.TYPE_INT_RGB)
    for (i in 0 until imageShape.height!!.toInt()) { // rows
        for (j in 0 until imageShape.width!!.toInt()) { // columns
            val r = get3D(i, j, 2, imageShape.width!!.toInt(), imageShape.channels!!.toInt()).coerceIn(0f, 1f)
            val g = get3D(i, j, 1, imageShape.width!!.toInt(), imageShape.channels!!.toInt()).coerceIn(0f, 1f)
            val b = get3D(i, j, 0, imageShape.width!!.toInt(), imageShape.channels!!.toInt()).coerceIn(0f, 1f)
            result.setRGB(j, i, Color(r, g, b).rgb)
        }
    }
    return result
}