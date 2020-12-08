/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.production

import java.awt.Color
import java.awt.Graphics
import javax.swing.JPanel
import kotlin.math.max
import kotlin.math.min

class Conv2dJPanel(
    val dst: Array<Array<Array<FloatArray>>>,
    val colorCoefficient: Double = 2.0
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

class Conv2dJPanel1(val dst: Array<Array<Array<FloatArray>>>) : JPanel() {
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
