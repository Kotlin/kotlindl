/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.swing

import java.awt.Dimension
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.image.BufferedImage
import javax.swing.JPanel

/**
 * A [JPanel] to display an image.
 *
 * @param [bufferedImage] an image represented by a [BufferedImage].
 */
open class ImagePanel(private val bufferedImage: BufferedImage) : JPanel() {
    override fun paint(graphics: Graphics) {
        super.paint(graphics)
        val x = (size.width - bufferedImage.width) / 2
        val y = (size.height - bufferedImage.height) / 2
        graphics.drawImage(bufferedImage, x, y, null)
    }

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }

    override fun getMinimumSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }

    companion object {
        /**
         * Creates an [ImagePanel] instance which displays given [bufferedImage]
         * and allows to draw on it using the given [draw] function.
         */
        fun createImagePanel(bufferedImage: BufferedImage, draw: Graphics2D.() -> Unit): JPanel {
            return object : ImagePanel(bufferedImage) {
                override fun paint(graphics: Graphics) {
                    super.paint(graphics)
                    (graphics as Graphics2D).draw()
                }
            }
        }
    }
}
