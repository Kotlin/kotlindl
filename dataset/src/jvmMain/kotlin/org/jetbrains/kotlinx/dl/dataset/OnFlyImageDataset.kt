/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.LabelGenerator
import java.io.File
import java.io.IOException
import java.nio.FloatBuffer
import kotlin.math.truncate
import kotlin.random.Random

/**
 * This dataset keeps all data on disk and generates batches on-fly according [preprocessing] pipeline.
 *
 * @param [xFiles] files to load images from
 * @param [y] labels to use for the loaded images
 * @param [preprocessing] preprocessing to apply to the loaded images
 */
public class OnFlyImageDataset internal constructor(
    private val xFiles: Array<File>,
    private val y: FloatArray,
    private val preprocessing: Preprocessing,
) : Dataset() {

    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copyImagesToBatch(src: Array<File>, start: Int, length: Int): Array<FloatArray> {
        return Array(length) { index -> applyImagePreprocessing(src[start + index]) }
    }

    private fun applyImagePreprocessing(file: File): FloatArray {
        return preprocessing.handleFile(file).first
    }

    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copyLabelsToBatch(src: FloatArray, start: Int, length: Int): FloatArray {
        return FloatArray(length) { src[start + it] }
    }

    /** Splits datasets on two sub-datasets according [splitRatio].*/
    override fun split(splitRatio: Double): Pair<OnFlyImageDataset, OnFlyImageDataset> {
        require(splitRatio in 0.0..1.0) { "'Split ratio' argument value must be in range [0.0; 1.0]." }

        val trainDatasetLastIndex = truncate(xFiles.size * splitRatio).toInt()

        val train = OnFlyImageDataset(
            xFiles.copyOfRange(0, trainDatasetLastIndex),
            y.copyOfRange(0, trainDatasetLastIndex),
            preprocessing
        )
        val test = OnFlyImageDataset(
            xFiles.copyOfRange(trainDatasetLastIndex, xFiles.size),
            y.copyOfRange(trainDatasetLastIndex, y.size),
            preprocessing
        )

        return Pair(train, test)
    }

    public companion object {
        /** Creates binary vector with size [numClasses] from [label]. */
        @JvmStatic
        public fun toOneHotVector(numClasses: Int, label: Byte): FloatArray {
            val ret = FloatArray(numClasses)
            ret[label.toInt() and 0xFF] = 1f
            return ret
        }

        /** Normalizes [bytes] via division on 255 to get values in range '[0; 1)'.*/
        @JvmStatic
        public fun toNormalizedVector(bytes: ByteArray): FloatArray {
            return FloatArray(bytes.size) { ((bytes[it].toInt() and 0xFF)) / 255f }
        }

        /** Converts [bytes] to [FloatArray]. */
        @JvmStatic
        public fun toRawVector(bytes: ByteArray): FloatArray {
            return FloatArray(bytes.size) { ((bytes[it].toInt() and 0xFF).toFloat()) }
        }

        /**
         * Create dataset [OnFlyImageDataset] from [pathToData] and [labels] using [preprocessing] to prepare images.
         */
        @JvmStatic
        public fun create(
            pathToData: File,
            labels: FloatArray,
            preprocessing: Preprocessing = Preprocessing()
        ): OnFlyImageDataset {
            return try {
                OnFlyImageDataset(OnHeapDataset.prepareFileNames(pathToData), labels, preprocessing)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        /**
         * Create dataset [OnFlyImageDataset] from [pathToData] and [labelGenerator] using [preprocessing] to prepare images.
         */
        @JvmStatic
        public fun create(
            pathToData: File,
            labelGenerator: LabelGenerator,
            preprocessors: Preprocessing = Preprocessing()
        ): OnFlyImageDataset {
            return try {
                val xFiles = OnHeapDataset.prepareFileNames(pathToData)
                val y = OnHeapDataset.prepareY(xFiles, labelGenerator)
                OnFlyImageDataset(xFiles, y, preprocessors)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }
    }

    /** Returns amount of data rows. */
    override fun xSize(): Int {
        return xFiles.size
    }

    /** Returns row by index [idx]. */
    override fun getX(idx: Int): FloatArray {
        return applyImagePreprocessing(xFiles[idx])
    }

    /** Returns label as [FloatArray] by index [idx]. */
    override fun getY(idx: Int): Float {
        return y[idx]
    }

    override fun shuffle(): OnFlyImageDataset {
        xFiles.shuffle(Random(12L))
        y.shuffle(Random(12L))
        return this
    }

    override fun createDataBatch(batchStart: Int, batchLength: Int): DataBatch {
        return DataBatch(
            copyImagesToBatch(xFiles, batchStart, batchLength),
            copyLabelsToBatch(y, batchStart, batchLength),
            batchLength
        )
    }
}
