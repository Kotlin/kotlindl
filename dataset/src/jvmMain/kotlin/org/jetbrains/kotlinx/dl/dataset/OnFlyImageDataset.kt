/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.generator.LabelGenerator
import org.jetbrains.kotlinx.dl.dataset.generator.LabelGenerator.Companion.prepareY
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ConvertToFloatArray
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException
import java.nio.FloatBuffer
import kotlin.math.truncate
import kotlin.random.Random

/**
 * This dataset keeps all data on disk and generates batches on the fly using provided [dataLoader].
 *
 * @param [x] sources to load data from
 * @param [y] labels to use for the loaded data
 * @param [dataLoader] data loader to load data with from the provided sources
 */
public class OnFlyImageDataset<D> internal constructor(
    private val x: Array<D>,
    private val y: FloatArray,
    private val dataLoader: DataLoader<D>,
) : Dataset() {

    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copySourcesToBatch(src: Array<D>, start: Int, length: Int): Array<FloatArray> {
        return Array(length) { index -> dataLoader.load(src[start + index]).first }
    }

    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copyLabelsToBatch(src: FloatArray, start: Int, length: Int): FloatArray {
        return FloatArray(length) { src[start + it] }
    }

    /** Splits datasets on two sub-datasets according [splitRatio].*/
    override fun split(splitRatio: Double): Pair<OnFlyImageDataset<D>, OnFlyImageDataset<D>> {
        require(splitRatio in 0.0..1.0) { "'Split ratio' argument value must be in range [0.0; 1.0]." }

        val trainDatasetLastIndex = truncate(x.size * splitRatio).toInt()

        val train = OnFlyImageDataset(
            x.copyOfRange(0, trainDatasetLastIndex),
            y.copyOfRange(0, trainDatasetLastIndex),
            dataLoader
        )
        val test = OnFlyImageDataset(
            x.copyOfRange(trainDatasetLastIndex, x.size),
            y.copyOfRange(trainDatasetLastIndex, y.size),
            dataLoader
        )

        return Pair(train, test)
    }

    /** Returns amount of data rows. */
    override fun xSize(): Int {
        return x.size
    }

    /** Returns row by index [idx]. */
    override fun getX(idx: Int): FloatArray {
        return dataLoader.load(x[idx]).first
    }

    /** Returns label as [FloatArray] by index [idx]. */
    override fun getY(idx: Int): Float {
        return y[idx]
    }

    override fun shuffle(): OnFlyImageDataset<D> {
        x.shuffle(Random(12L))
        y.shuffle(Random(12L))
        return this
    }

    override fun createDataBatch(batchStart: Int, batchLength: Int): DataBatch {
        return DataBatch(
            copySourcesToBatch(x, batchStart, batchLength),
            copyLabelsToBatch(y, batchStart, batchLength),
            batchLength
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
         * Create dataset [OnFlyImageDataset] from [pathToData] and [labels] using [preprocessing] to prepare images.
         */
        @JvmStatic
        @Throws(IOException::class)
        public fun create(
            pathToData: File,
            labels: FloatArray,
            preprocessing: Operation<BufferedImage, FloatData> = ConvertToFloatArray()
        ): OnFlyImageDataset<File> {
            return OnFlyImageDataset(OnHeapDataset.prepareFileNames(pathToData), labels, preprocessing.fileLoader())
        }

        /**
         * Create dataset [OnFlyImageDataset] from [pathToData] and [labelGenerator] using [preprocessing] to prepare images.
         */
        @JvmStatic
        @Throws(IOException::class)
        public fun create(
            pathToData: File,
            labelGenerator: LabelGenerator<File>,
            preprocessing: Operation<BufferedImage, FloatData> = ConvertToFloatArray()
        ): OnFlyImageDataset<File> {
            val xFiles = OnHeapDataset.prepareFileNames(pathToData)
            val y = labelGenerator.prepareY(xFiles)
            return OnFlyImageDataset(xFiles, y, preprocessing.fileLoader())
        }
    }
}
