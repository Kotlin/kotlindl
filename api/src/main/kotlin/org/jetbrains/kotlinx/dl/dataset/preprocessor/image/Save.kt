/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException
import javax.imageio.ImageIO

/**
 * This image preprocessor defines the Save operation.
 *
 * It saves the input image to the [dirLocation].
 *
 * @property [dirLocation] Could be link to the file or directory.
 * // TODO: add more docs
 *
 * NOTE: currently it supports [BufferedImage.TYPE_3BYTE_BGR] image type only.
 */
public class Save(
    public var dirLocation: File? = null,
    // TODO: add filenameStrategy: keepName, counter, withPrefix
// TODO: add filetype: PNG or JPG
) : ImagePreprocessor {
    @Throws(IOException::class)
    internal fun imageToFile(filename: String, image: BufferedImage, shape: ImageShape): File {
        val outputFile: File = if (dirLocation!!.isDirectory) {
            File("${dirLocation}\\${filename}")
        } else {
            dirLocation!!
        }
// TODO: file extension is a part of name, need to extract name without extension
        ImageIO.write(image, "jpg", outputFile)
        return outputFile
    }


    override fun apply(image: BufferedImage, inputShape: ImageShape): Pair<BufferedImage, ImageShape> {
        TODO("Not yet implemented")
    }
}
