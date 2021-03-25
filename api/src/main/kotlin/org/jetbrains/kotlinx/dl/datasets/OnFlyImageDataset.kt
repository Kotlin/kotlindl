/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets

import java.io.File
import java.io.IOException
import java.nio.FloatBuffer

/**
 * Basic class to handle features [x] and labels [y].
 *
 * NOTE: Labels [y] should have shape <number of rows; number of labels> and contain exactly one 1 and other 0-es per row to be result of one-hot-encoding.
 */
public class OnFlyImageDataset internal constructor(
    private val preprocessing: Preprocessing, // TODO: maybe move to builder
    private val y: Array<FloatArray>
) : Dataset() {

    private var xFiles: Array<File> // maybe move to constructor

    init {
        val loading = preprocessing.imagePreprocessing.load
        xFiles = loading.prepareFileNames()
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
        return preprocessing.handleFile(file)
    }

    // TODO: src argument could be removed as stupid
    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copyToBatch(src: Array<FloatArray>, start: Int, length: Int): Array<FloatArray> {
        val dataForBatch = Array(length) { FloatArray(src[0].size) { 0.0f } }
        for (i in start until start + length) {
            dataForBatch[i - start] = src[i].copyOf() // Creates new copy for batch data
        }
        return dataForBatch
    }


    /** Splits datasets on two sub-datasets according [splitRatio].*/
    override fun split(splitRatio: Double): Pair<OnFlyImageDataset, OnFlyImageDataset> {
        TODO() // split lists of files on two parts
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
            labels: Array<FloatArray>
        ): OnFlyImageDataset {
            return try {
                OnFlyImageDataset(preprocessors, labels)
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
    override fun getY(idx: Int): FloatArray {
        return y[idx]
    }

    /** Returns label as [Int] by index [idx]. */
    override fun getLabel(idx: Int): Int {
        val labelArray = y[idx]
        return labelArray.indexOfFirst { it == labelArray.maxOrNull()!! }
    }

    // TODO: check that initial data are not shuffled or return void if are shuffled
    override fun shuffle(): OnFlyImageDataset {
        TODO()
    }

    override fun createDataBatch(batchStart: Int, batchLength: Int): DataBatch {
        return DataBatch(
            copyImagesToBatch(xFiles, batchStart, batchLength),
            copyToBatch(y, batchStart, batchLength),
            batchLength
        )
    }
}
