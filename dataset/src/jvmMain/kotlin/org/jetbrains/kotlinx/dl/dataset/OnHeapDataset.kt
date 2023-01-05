/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.DataLoader.Companion.prepareX
import org.jetbrains.kotlinx.dl.dataset.generator.LabelGenerator
import org.jetbrains.kotlinx.dl.dataset.generator.LabelGenerator.Companion.prepareY
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ConvertToFloatArray
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException
import java.nio.FloatBuffer
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.truncate
import kotlin.random.Random

// do not remove this import
import kotlin.streams.toList

/**
 * Basic class to handle features [x] and labels [y].
 *
 * It loads the whole data from disk to the Heap Memory.
 *
 * NOTE: Labels [y] should have shape <number of rows; number of labels> and contain exactly one 1 and other 0-es per row to be result of one-hot-encoding.
 * @property [x] an array of feature vectors
 * @property [y] an array of labels
 */
public class OnHeapDataset internal constructor(public val x: Array<FloatArray>, public val y: FloatArray) : Dataset() {

    init {
        check(x.size == y.size) {
            "Number of data elements in the dataset (${x.size}) is not the same as the number of labels (${y.size})."
        }
    }

    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copyXToBatch(src: Array<FloatArray>, start: Int, length: Int): Array<FloatArray> {
        val dataForBatch = Array(length) {
            src[it + start].copyOf()
        }
        return dataForBatch
    }

    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copyLabelsToBatch(src: FloatArray, start: Int, length: Int): FloatArray {
        val dataForBatch = FloatArray(length) { 0.0f }
        for (i in start until start + length) {
            dataForBatch[i - start] = src[i]
        }
        return dataForBatch
    }

    /** Splits datasets on two sub-datasets according [splitRatio].*/
    override fun split(splitRatio: Double): Pair<OnHeapDataset, OnHeapDataset> {
        require(splitRatio in 0.0..1.0) { "'Split ratio' argument value must be in range [0.0; 1.0]." }

        val trainDatasetLastIndex = truncate(x.size * splitRatio).toInt()

        return Pair(
            OnHeapDataset(x.copyOfRange(0, trainDatasetLastIndex), y.copyOfRange(0, trainDatasetLastIndex)),
            OnHeapDataset(x.copyOfRange(trainDatasetLastIndex, x.size), y.copyOfRange(trainDatasetLastIndex, y.size))
        )
    }

    /** Returns amount of data rows. */
    override fun xSize(): Int {
        return x.size
    }

    /** Returns row by index [idx]. */
    override fun getX(idx: Int): FloatArray {
        return x[idx]
    }

    /** Returns label as [FloatArray] by index [idx]. */
    override fun getY(idx: Int): Float {
        return y[idx]
    }

    override fun shuffle(): OnHeapDataset {
        x.shuffle(Random(12L))
        y.shuffle(Random(12L))
        return this
    }

    override fun createDataBatch(batchStart: Int, batchLength: Int): DataBatch {
        return DataBatch(
            copyXToBatch(x, batchStart, batchLength),
            copyLabelsToBatch(y, batchStart, batchLength)
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

        /** Creates float [label]. */
        @JvmStatic
        public fun convertByteToFloat(label: Byte): Float {
            return (label.toInt() and 0xFF).toFloat()
        }

        /**
         * Creates an [OnHeapDataset] from [features] and [labels].
         */
        @JvmStatic
        public fun create(features: Array<FloatArray>, labels: FloatArray): OnHeapDataset {
            return OnHeapDataset(features, labels)
        }

        /**
         * Creates an [OnHeapDataset] from [pathToData] and [labels] using [preprocessing] to prepare images.
         */
        @JvmStatic
        @Throws(IOException::class)
        public fun create(
            pathToData: File,
            labels: FloatArray,
            preprocessing: Operation<BufferedImage, FloatData>
        ): OnHeapDataset {
            val xFiles = prepareFileNames(pathToData)
            val x = preprocessing.fileLoader().prepareX(xFiles)

            return OnHeapDataset(x, labels)
        }

        /**
         * Creates an [OnHeapDataset] from [pathToData] and [labelGenerator] with [preprocessing] to prepare images.
         */
        @JvmStatic
        @Throws(IOException::class)
        public fun create(
            pathToData: File,
            labelGenerator: LabelGenerator<File>,
            preprocessing: Operation<BufferedImage, FloatData> = ConvertToFloatArray()
        ): OnHeapDataset {
            val xFiles = prepareFileNames(pathToData)
            val x = preprocessing.fileLoader().prepareX(xFiles)
            val y = labelGenerator.prepareY(xFiles)

            return OnHeapDataset(x, y)
        }

        @Throws(IOException::class)
        internal fun prepareFileNames(pathToData: File): Array<File> {
            return Files.walk(pathToData.toPath())
                .filter { path: Path -> Files.isRegularFile(path) }
                .filter { path: Path -> path.toString().endsWith(".jpg") || path.toString().endsWith(".png") }
                .map { obj: Path -> obj.toFile() }
                .toList()
                .toTypedArray()
        }
    }
}


