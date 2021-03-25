/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets

import java.awt.RenderingHints
import java.awt.image.BufferedImage


// ЛУЧШЕ СДЕЛАТЬ СИГНАТУРУ ByteArray->ByTeArray или Image->Image или одну общую функцию с кучей параметров - как стадии и самому их упорядочить, как мне удобно
// ИЛИ сделать все же на floats на самому заимпелменить resize с простейшей интерполяцией и коэффициентом resize, повороты сделать без афинного - простым отображением, как вариант + зеракльное отображение сделать полезно для бубудщей аугментации
// надо профилировать оба подхода - интересно сколько займет, поискать дешевые алгоритмы resize
// как вариант надо обернуть bufferedImage как сделали в той самой библиотеке

public class Resize(
    public var outputWidth: Int = 100,
    public var outputHeight: Int = 100,
    public var interpolation: InterpolationType = InterpolationType.BILINEAR
) : ImagePreprocessor {
    override fun apply(image: BufferedImage, inputShape: ImageShape): Pair<BufferedImage, ImageShape> {
        val resizedImage = BufferedImage(outputWidth, outputHeight, BufferedImage.TYPE_3BYTE_BGR)
        val graphics2D = resizedImage.createGraphics()
        graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
        //graphics2D.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED)
        //graphics2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        graphics2D.drawImage(image, 0, 0, outputWidth, outputHeight, null)
        graphics2D.dispose()

        return Pair(
            resizedImage,
            ImageShape(outputWidth.toLong(), outputHeight.toLong(), channels = inputShape.channels)
        )
    }
}
