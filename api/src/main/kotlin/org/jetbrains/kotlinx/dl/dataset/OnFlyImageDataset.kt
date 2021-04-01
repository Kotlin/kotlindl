/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.FromFolders
import java.io.File
import java.io.IOException
import java.nio.FloatBuffer
import kotlin.math.truncate
import kotlin.random.Random

/**
 * Basic class to handle features [x] and labels [y].
 *
 * NOTE: Labels [y] should have shape <number of rows; number of labels> and contain exactly one 1 and other 0-es per row to be result of one-hot-encoding.
 */
public class OnFlyImageDataset internal constructor(
    private var preprocessing: Preprocessing, // TODO: maybe move to builder
    labels: FloatArray?
) : Dataset() {

    private var xFiles: Array<File> // maybe move to constructor
    private var y: FloatArray

    init {
        val loading = preprocessing.imagePreprocessingStage.load
        xFiles = loading.prepareFileNames()
        if (labels != null) {
            y = labels
        } else {
            if (preprocessing.imagePreprocessingStage.load.labelGenerator != null) {
                val labelGenerator = preprocessing.imagePreprocessingStage.load.labelGenerator as FromFolders
                val mapping = labelGenerator.mapping

                y = FloatArray(xFiles.size) { 0.0f }
                for (i in xFiles.indices) {
                    y[i] = (mapping[xFiles[i].parentFile.name]
                        ?: error("The parent directory of ${xFiles[i].absolutePath} is ${xFiles[i].parentFile.name}. No such class name in mapping $mapping")).toFloat()
                }

            } else {
                throw IllegalStateException("Label generator should be defined for Loading stage of image preprocessing.")
            }
        }
    }

    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copyImagesToBatch(src: Array<File>, start: Int, length: Int): Array<FloatArray> {
        val numOfPixels: Int = 32 * 32 * 3

        val dataForBatch = Array(length) { FloatArray(numOfPixels) { 0.0f } }
        for (i in start until start + length) {
            dataForBatch[i - start] = applyImagePreprocessing(src[i])
        }
        return dataForBatch
    }

    private fun applyImagePreprocessing(file: File): FloatArray {
        return preprocessing.handleFile(file).first
    }

    // TODO: src argument could be removed as stupid
    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copyLabelsToBatch(src: FloatArray, start: Int, length: Int): FloatArray {
        val dataForBatch = FloatArray(length) { 0.0f }
        for (i in start until start + length) {
            dataForBatch[i - start] = src[i]
        }
        return dataForBatch
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
         * Takes data from external data [features] and [labels]
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
         * Takes data from external data [features] and [labels]
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
        TODO()
        //return xFiles[idx] // TODO: convert file to image and apply all preprocessors
    }

    /** Returns label as [FloatArray] by index [idx]. */
    override fun getY(idx: Int): Float {
        return y[idx]
    }

    // TODO: check that initial data are not shuffled or return void if are shuffled
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
