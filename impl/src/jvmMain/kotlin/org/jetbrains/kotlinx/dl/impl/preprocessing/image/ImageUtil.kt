/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import java.awt.Graphics2D
import java.awt.image.BufferedImage

internal fun BufferedImage.draw(block: (Graphics2D) -> Unit) {
    val graphics2D = createGraphics()
    try {
        block(graphics2D)
    } finally {
        graphics2D.dispose()
    }
}

internal fun BufferedImage.copy(): BufferedImage {
    val result = BufferedImage(width, height, type)
    copyData(result.raster)
    return result
}

internal fun BufferedImage.getShape(): TensorShape {
    return TensorShape(width.toLong(), height.toLong(), colorModel.numComponents.toLong())
}
