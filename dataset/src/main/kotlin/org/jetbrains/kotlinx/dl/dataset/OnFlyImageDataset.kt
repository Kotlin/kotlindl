/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import java.io.File
import java.io.IOException
import java.nio.FloatBuffer
import kotlin.math.truncate
import kotlin.random.Random

/**
 * This dataset keeps all data on disk and generates batches on-fly according [preprocessing] pipeline.
 *
 * @param [preprocessing] The preprocessing pipeline.
 * @param [labels] Also it could keep labels, if it could be preprocessed separately and passed to this parameter. If it's missed, it will try to generate labels on-fly according
 * [org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.LabelGenerator].
 */
public class OnFlyImageDataset internal constructor(
    private var preprocessing: Preprocessing,
    labels: FloatArray?
) : Dataset() {
    private var xFiles: Array<File>

    private var y: FloatArray

    init {
        val loading = preprocessing.imagePreprocessingStage.load
        xFiles = loading.prepareFileNames()
        y = labels ?: OnHeapDataset.prepareY(xFiles, preprocessing)
    }

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

        val train = OnFlyImageDataset(preprocessing, y.copyOfRange(0, trainDatasetLastIndex))
        train.xFiles = xFiles.copyOfRange(0, trainDatasetLastIndex)

        val test = OnFlyImageDataset(
            preprocessing,
            y.copyOfRange(trainDatasetLastIndex, y.size)
        )
        test.xFiles = xFiles.copyOfRange(trainDatasetLastIndex, xFiles.size)

        return Pair(
            train,
            test
        )
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
         * Use [preprocessors] and [labels] to prepare data
         * to create dataset [OnFlyImageDataset].
         */
        @JvmStatic
        public fun create(
            preprocessors: Preprocessing,
            labels: FloatArray
        ): OnFlyImageDataset {
            return try {
                OnFlyImageDataset(preprocessors, labels)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        /**
         * Use [preprocessors] to prepare data
         * to create dataset [OnFlyImageDataset].
         */
        @JvmStatic
        public fun create(
            preprocessors: Preprocessing
        ): OnFlyImageDataset {
            return try {
                OnFlyImageDataset(preprocessors, null)
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
