/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import java.awt.Dimension
import java.awt.Graphics
import java.awt.image.BufferedImage
import javax.swing.JPanel

/**
 * A [JPanel] to display an image.
 *
 * @param [bufferedImage] an image represented by a [BufferedImage].
 */
class ImagePanel(private val bufferedImage: BufferedImage) : JPanel() {
    override fun paint(graphics: Graphics) {
        super.paint(graphics)
        val x = (size.width - bufferedImage.width) / 2
        val y = (size.height - bufferedImage.height) / 2
        graphics.drawImage(bufferedImage, x, y, null)
    }

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
}
