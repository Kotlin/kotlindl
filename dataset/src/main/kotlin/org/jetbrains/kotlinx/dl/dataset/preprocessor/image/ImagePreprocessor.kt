/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.image.BufferedImage

/** Basic interface for image preprocessors. It operates on [BufferedImage]. */
public interface ImagePreprocessor {
    /**
     * Computes output image shape for the provided [inputShape].
     * @param inputShape image input shape. Null value means that input image size is not known.
     *                   This is useful for operations with a fixed output size,
     *                   which should return a non-null value in this case.
     *
     * @return output image shape
     */
    public fun getOutputShape(inputShape: ImageShape?): ImageShape? = inputShape

    /**
     * Transforms [image] with [inputShape] to the new image with the new shape.
     *
     * @return Pair <new image; new shape>.
     */
    public fun apply(image: BufferedImage, inputShape: ImageShape): Pair<BufferedImage, ImageShape>
}




