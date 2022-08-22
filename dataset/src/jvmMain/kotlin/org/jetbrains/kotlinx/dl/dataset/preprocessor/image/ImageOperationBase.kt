/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation
import java.awt.image.BufferedImage

/**
 * Base class for [Operation]<BufferedImage, BufferedImage> implementations.
 * Allows to add additional features, such as saving output with [ImageSaver].
 */
public abstract class ImageOperationBase : Operation<BufferedImage, BufferedImage> {
    internal var save: ImageSaver? = null
}
