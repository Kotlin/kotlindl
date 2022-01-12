/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.swing

import org.jetbrains.kotlinx.dl.api.extension.get3D
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.visualization.letsplot.TensorImageData
import java.awt.*
import java.awt.image.BufferedImage
import javax.swing.JFrame
import javax.swing.JPanel
import kotlin.math.max
import kotlin.math.min


class Conv2dJPanel(
    private val dst: Array<Array<Array<FloatArray>>>,
    private val colorCoefficient: Double = 2.0
) : JPanel() {
    override fun paint(g: Graphics) {

        for (k in 0 until 32) {
            for (i in dst.indices) {
                for (j in dst[i].indices) {
                    val width = 15
                    val height = 15
                    var x = 10 + i * width
                    val y = 10 + j * height + k % 8 * 150
                    when (k) {
                        in 8..15 -> {
                            x += 150
                        }
                        in 16..23 -> {
                            x += 150 * 2
                        }
                        in 24..31 -> {
                            x += 150 * 3
                        }
                    }

                    val float = dst[i][j][0][k]
                    val grey = (min(1.0f, max(float * colorCoefficient.toFloat(), 0.0f)) * 255).toInt()
                    val color = Color(grey, grey, grey)
                    g.color = color
                    g.fillRect(y, x, width, height)
                    g.color = Color.BLACK
                    g.drawRect(y, x, width, height)
                }
            }
        }
    }
}

class Conv2dJPanel1(private val dst: Array<Array<Array<FloatArray>>>) : JPanel() {
    override fun paint(g: Graphics) {

        for (k in 0 until 32) {
            for (i in dst.indices) {
                for (j in dst[i].indices) {
                    val float = dst[i][j][0][k]
                    val grey = (min(1.0f, max(float * 2, 0.0f)) * 255).toInt()
                    val color = Color(grey, grey, grey)
                    g.color = color
                    g.fillRect(10 + i * 20 + k % 8 * 105, 10 + j * 20 + k * 15, 10, 10)
                    g.color = Color.BLACK
                    g.drawRect(10 + i * 20 + k % 8 * 105, 10 + j * 20 + k * 15, 10, 10)
                }
            }
        }
    }
}

class ReluGraphics(private val dst: Array<Array<Array<FloatArray>>>) : JPanel() {
    override fun paint(g: Graphics) {
        for (k in 0 until 32) {
            for (i in dst[0].indices) {
                for (j in dst[0][i].indices) {
                    val width = 5
                    val height = 5
                    var x = 10 + i * width
                    val y = 10 + j * height + k % 8 * 150
                    when (k) {
                        in 8..15 -> {
                            x += 150
                        }
                        in 16..23 -> {
                            x += 150 * 2
                        }
                        in 24..31 -> {
                            x += 150 * 3
                        }
                    }

                    val float = dst[0][i][j][k]
                    val grey = (min(1.0f, max(float * 4, 0.0f)) * 255).toInt()
                    val color = Color(grey, grey, grey)
                    g.color = color

                    g.fillRect(y, x, width, height)
                    g.color = Color.BLACK
                    g.drawRect(y, x, width, height)
                }
            }
        }
    }
}

class ReluGraphics2(private val dst: Array<Array<Array<FloatArray>>>) : JPanel() {
    override fun paint(g: Graphics) {

        for (k in 0 until 64) {
            for (i in dst[0].indices) {
                for (j in dst[0][i].indices) {
                    val width = 7
                    val height = 7

                    var x = 10 + i * width
                    val y = 10 + j * height + k % 8 * 100 // 14 * width <= 100

                    when (k) {
                        in 8..15 -> {
                            x += 100
                        }
                        in 16..23 -> {
                            x += 100 * 2
                        }
                        in 24..31 -> {
                            x += 100 * 3
                        }
                        in 32..39 -> {
                            x += 100 * 4
                        }
                        in 40..47 -> {
                            x += 100 * 5
                        }
                        in 48..55 -> {
                            x += 100 * 6
                        }
                        in 56..63 -> {
                            x += 100 * 7
                        }
                    }

                    val float = dst[0][i][j][k]
                    val grey = (min(1.0f, max(float, 0.0f)) * 255).toInt()
                    val color = Color(grey, grey, grey)
                    g.color = color
                    g.fillRect(y, x, width, height)
                    g.color = Color.BLACK
                    g.drawRect(y, x, width, height)
                }
            }
        }
    }
}

/*class DetectedObjectJPanel(
    val dst: FloatArray,
    val imageShape: ImageShape,
    val detectedObjects: List<DetectedObject>,
) : JPanel() {
    override fun paint(g: Graphics) {
        //drawImage()
        detectedObjects.forEach {
            val pixelWidth = 1
            val pixelHeight = 1

            val top = it.yMin * imageShape.height!! * pixelHeight
            val left = it.xMin * imageShape.width!! * pixelWidth
            val bottom = it.yMax * imageShape.height!! * pixelHeight
            val right = it.xMax * imageShape.width!! * pixelWidth
            // left, bot, right, top

            // y = columnIndex
            // x = rowIndex
            val y = 50 + top
            val x = 50 + left

            g.color = Color.BLACK
            g.drawRect(x.toInt(), y.toInt(), (right - left).toInt(), (top - bottom).toInt())
            g.drawString(" ${it.classLabel} : ${it.probability}",  x.toInt(), y.toInt(),);
        }

    }

    private fun drawImage() {
        for (i in 0 until imageShape.height!!.toInt()) { // rows
            for (j in 0 until imageShape.width!!.toInt()) { // columns
                val pixelWidth = 1
                val pixelHeight = 1

                // y = columnIndex
                // x = rowIndex
                val y = 100 + i * pixelWidth
                val x = 100 + j * pixelHeight

                val r =
                    dst.get3D(i, j, 2, imageShape.width!!.toInt(), imageShape.channels.toInt())
                val g =
                    dst.get3D(i, j, 1, imageShape.width!!.toInt(), imageShape.channels.toInt())
                val b =
                    dst.get3D(i, j, 0, imageShape.width!!.toInt(), imageShape.channels.toInt())

                val color = Color(r, g, b)
                graphics.color = color
                graphics.fillRect(x, y, pixelWidth, pixelHeight)
            }
        }
    }
}
*/
fun drawDetectedObjects(dst: FloatArray, imageShape: ImageShape, detectedObjects: List<DetectedObject>) {
    val frame = JFrame("Filters")
    @Suppress("UNCHECKED_CAST")
    frame.contentPane.add(DetectedObjectJPanel(dst, imageShape, detectedObjects))
    frame.pack()
    frame.setLocationRelativeTo(null)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}

fun drawRawLandMarks(dst: FloatArray, imageShape: ImageShape, landmarks: Map<String, Any>) {
    val frame = JFrame("Landmarks")
    @Suppress("UNCHECKED_CAST")
    frame.contentPane.add(RawLandMarksJPanel(dst, imageShape, landmarks))
    frame.pack()
    frame.setLocationRelativeTo(null)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}

fun drawDetectedPose(dst: FloatArray, imageShape: ImageShape, detectedPose: DetectedPose) {
    val frame = JFrame("Filters")
    @Suppress("UNCHECKED_CAST")
    frame.contentPane.add(DetectedPoseJPanel(dst, imageShape, detectedPose))
    frame.pack()
    frame.setLocationRelativeTo(null)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}

fun drawRawPoseLandMarks(dst: FloatArray, imageShape: ImageShape, posepoints: Array<FloatArray>) {
    val frame = JFrame("Landmarks")
    @Suppress("UNCHECKED_CAST")
    frame.contentPane.add(RawPosePointsJPanel(dst, imageShape, posepoints))
    frame.pack()
    frame.setLocationRelativeTo(null)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}

fun drawMultiPoseLandMarks(dst: FloatArray, imageShape: ImageShape, multiPoseDetectionResult: MultiPoseDetectionResult) {
    val frame = JFrame("Landmarks")
    @Suppress("UNCHECKED_CAST")
    frame.contentPane.add(MultiPosePointsJPanel(dst, imageShape, multiPoseDetectionResult))
    frame.pack()
    frame.setLocationRelativeTo(null)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}

fun drawRawMultiPoseLandMarks(dst: FloatArray, imageShape: ImageShape, posepoints: Array<FloatArray>) {
    val frame = JFrame("Landmarks")
    @Suppress("UNCHECKED_CAST")
    frame.contentPane.add(RawMultiPosePointsJPanel(dst, imageShape, posepoints))
    frame.pack()
    frame.setLocationRelativeTo(null)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}

fun drawLandMarks(dst: FloatArray, imageShape: ImageShape, landmarks: List<Landmark>) {
    val frame = JFrame("Landmarks")
    @Suppress("UNCHECKED_CAST")
    frame.contentPane.add(LandMarksJPanel(dst, imageShape, landmarks))
    frame.pack()
    frame.setLocationRelativeTo(null)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}

class RawMultiPosePointsJPanel(
    val image: FloatArray,
    val imageShape: ImageShape,
    private val rawPosePoints: Array<FloatArray>
) : JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape)

    override fun paint(graphics: Graphics) {
        super.paint(graphics)
        val posePoints = mutableListOf<MutableList<Triple<Float, Float, Float>>>()
        rawPosePoints.forEachIndexed { index, floats ->
            val poseLandmarks = mutableListOf<Triple<Float, Float, Float>>()
            for (i in 0 .. 50 step 3) {
                poseLandmarks.add(i%3, Triple(floats[i + 1], floats[i], floats[i + 2]))
            }

            //if(floats[55] > 0.1) { // threshold
                posePoints.add(index, poseLandmarks)
            //}
        }

        graphics.drawImage(bufferedImage, 0, 0, null)

        for (i in 0 until posePoints.size) {
            val onePosePoints = posePoints[i]

            for (j in onePosePoints.indices) {
                val xLM = (size.width) * (onePosePoints[j].first)
                val yLM = (size.height) * (onePosePoints[j].second)

                graphics as Graphics2D
                val stroke1: Stroke = BasicStroke(3f)
                graphics.setColor(Color((6 - i) * 40, i * 20, i * 10))
                graphics.stroke = stroke1
                graphics.drawOval(xLM.toInt(), yLM.toInt(), 3, 3)
            }
        }
    }

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }

    override fun getMinimumSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
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
            val yRect = bottom
            val xRect = left
            graphics as Graphics2D
            val stroke: Stroke = BasicStroke(6f * detectedObject.probability)
            graphics.setColor(Color.ORANGE)
            graphics.stroke = stroke
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
    }

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }

    override fun getMinimumSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
}

class RawPosePointsJPanel(
    val image: FloatArray,
    val imageShape: ImageShape,
    private val rawPosePoints: Array<FloatArray>
) : JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape)

    override fun paint(graphics: Graphics) {
        super.paint(graphics)
        val posePoints = mutableListOf<Triple<Float, Float, Float>>()
        for (i in rawPosePoints.indices) {
            posePoints.add(Triple(rawPosePoints[i][1], rawPosePoints[i][0], rawPosePoints[i][2])) //(y, x, score)
        }

        graphics.drawImage(bufferedImage, 0, 0, null)

        for (i in posePoints.indices) {
            val xLM = (size.width) * (posePoints[i].first)
            val yLM = (size.height) * (posePoints[i].second)

            graphics as Graphics2D
            val stroke1: Stroke = BasicStroke(3f)
            graphics.setColor(Color.RED)
            graphics.stroke = stroke1
            graphics.drawOval(xLM.toInt(), yLM.toInt(), 3, 3)
        }
    }

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }

    override fun getMinimumSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
}

class RawLandMarksJPanel(val image: FloatArray, val imageShape: ImageShape, private val landmarks: Map<String, Any>) :
    JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape)

    override fun paint(graphics: Graphics) {
        super.paint(graphics)
        val tempLandMarks = mutableListOf<Pair<Float, Float>>()
        val floats = (landmarks.values.toTypedArray()[0] as Array<FloatArray>)[0]
        for (i in floats.indices step 2) {
            tempLandMarks.add(Pair(floats[i], floats[i + 1]))
        }


        val xCoefficient: Float = size.width.toFloat() / bufferedImage.width.toFloat()
        val yCoefficient: Float = size.height.toFloat() / bufferedImage.height.toFloat()

        graphics.drawImage(bufferedImage, 0, 0, null)

        for (i in tempLandMarks.indices) {
            val xLM = (size.width / 2) * (1 + tempLandMarks[i].first) / xCoefficient
            val yLM = (size.height / 2) * (1 + tempLandMarks[i].second) / yCoefficient

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

class LandMarksJPanel(val image: FloatArray, val imageShape: ImageShape, private val landmarks: List<Landmark>) : JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape)

    override fun paint(graphics: Graphics) {
        super.paint(graphics)

        graphics.drawImage(bufferedImage, 0, 0, null)

        for (i in landmarks.indices) {
            val xLM = (size.width / 2) * (1 + landmarks[i].xRate)
            val yLM = (size.height / 2) * (1 + landmarks[i].yRate)

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


class ImagesJPanel3(
    private val dst: FloatArray,
    val imageShape: ImageShape
) : JPanel() {
    override fun paint(graphics: Graphics) {
        for (i in 0 until imageShape.height!!.toInt()) { // rows
            for (j in 0 until imageShape.width!!.toInt()) { // columns
                val pixelWidth = 2
                val pixelHeight = 2

                // y = columnIndex
                // x = rowIndex
                val y = 100 + i * pixelWidth
                val x = 100 + j * pixelHeight

                val r =
                    dst.get3D(i, j, 2, imageShape.width!!.toInt(), imageShape.channels!!.toInt())
                val g =
                    dst.get3D(i, j, 1, imageShape.width!!.toInt(), imageShape.channels!!.toInt())
                val b =
                    dst.get3D(i, j, 0, imageShape.width!!.toInt(), imageShape.channels!!.toInt())
                val r1 = (min(1.0f, max(r * 0.8f, 0.0f)) * 255).toInt()
                val g1 = (min(1.0f, max(g * 0.8f, 0.0f)) * 255).toInt()
                val b1 = (min(1.0f, max(b * 0.8f, 0.0f)) * 255).toInt()
                val color = Color(r, g, b)
                graphics.color = color
                graphics.fillRect(x, y, pixelWidth, pixelHeight)
                graphics.color = Color.BLACK
                graphics.drawRect(x, y, pixelWidth, pixelHeight)
            }
        }
    }
}

fun drawActivations(activations: List<*>) {
    val frame = JFrame("Visualise the matrix weights on Relu")
    @Suppress("UNCHECKED_CAST")
    frame.contentPane.add(ReluGraphics(activations[0] as TensorImageData))
    frame.setSize(1500, 1500)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false

    val frame2 = JFrame("Visualise the matrix weights on Relu_1")
    @Suppress("UNCHECKED_CAST")
    frame2.contentPane.add(ReluGraphics2(activations[1] as TensorImageData))
    frame2.setSize(1500, 1500)
    frame2.isVisible = true
    frame2.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame2.isResizable = false
}

fun drawFilters(filters: Array<*>, colorCoefficient: Double = 2.0) {
    val frame = JFrame("Filters")
    @Suppress("UNCHECKED_CAST")
    frame.contentPane.add(Conv2dJPanel(filters as TensorImageData, colorCoefficient))
    frame.setSize(1000, 1000)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
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
