/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.RenderingHints
import java.awt.image.BufferedImage

/**
 * This image preprocessor defines the Resize operation.
 *
 * Resize operations creates new image with the following sizes:
 *  - resizedWidth = [outputWidth]
 *  - resizedHeight = [outputHeight]
 *
 * The final image quality depends on the following tunable parameters: [interpolation], [renderingSpeed], [enableAntialiasing].
 *
 * @property [outputWidth] The output width.
 * @property [outputHeight] The output height.
 * @property [interpolation] Interpolation algorithm.
 * @property [renderingSpeed] Speed of preprocessing.
 * @property [enableAntialiasing] The image will be cropped from the right by the given number of pixels.
 */
public class Resize(
    public var outputWidth: Int = 100,
    public var outputHeight: Int = 100,
    public var interpolation: InterpolationType = InterpolationType.BILINEAR,
    public var renderingSpeed: RenderingSpeed = RenderingSpeed.MEDIUM,
    public var enableAntialiasing: Boolean = true
) : ImagePreprocessorBase() {

    override fun getOutputShape(inputShape: ImageShape): ImageShape {
        return ImageShape(outputWidth.toLong(), outputHeight.toLong(), inputShape.channels)
    }

    override fun apply(image: BufferedImage): BufferedImage {
        val resizedImage = BufferedImage(outputWidth, outputHeight, image.type)
        val graphics2D = resizedImage.createGraphics()

        val renderingHint = when (interpolation) {
            InterpolationType.BILINEAR -> RenderingHints.VALUE_INTERPOLATION_BILINEAR
            InterpolationType.NEAREST -> RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR
            InterpolationType.BICUBIC -> RenderingHints.VALUE_INTERPOLATION_BICUBIC
        }

        val renderingSpeed = when (renderingSpeed) {
            RenderingSpeed.FAST -> RenderingHints.VALUE_RENDER_SPEED
            RenderingSpeed.SLOW -> RenderingHints.VALUE_RENDER_QUALITY
            RenderingSpeed.MEDIUM -> RenderingHints.VALUE_RENDER_DEFAULT
        }

        graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION, renderingHint)
        graphics2D.setRenderingHint(RenderingHints.KEY_RENDERING, renderingSpeed)

        if (enableAntialiasing) {
            graphics2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        }

        graphics2D.drawImage(image, 0, 0, outputWidth, outputHeight, null)
        graphics2D.dispose()

        return resizedImage
    }
}
