/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.FromFolders
import java.io.File
import java.io.IOException
import java.nio.FloatBuffer
import kotlin.math.roundToInt
import kotlin.math.truncate
import kotlin.random.Random

/**
 * Basic class to handle features [x] and labels [y].
 *
 * It loads the whole data from disk to the Heap Memory.
 *
 * NOTE: Labels [y] should have shape <number of rows; number of labels> and contain exactly one 1 and other 0-es per row to be result of one-hot-encoding.
 */
public class OnHeapDataset internal constructor(private val x: Array<FloatArray>, private val y: FloatArray) :
    Dataset() {

    /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
    private fun copyXToBatch(src: Array<FloatArray>, start: Int, length: Int): Array<FloatArray> {
        val dataForBatch = Array(length) { FloatArray(src[0].size) { 0.0f } }
        for (i in start until start + length) {
            dataForBatch[i - start] = src[i].copyOf() // Creates new copy for batch data
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

        /** Normalizes [bytes] via division on 255 to get values in range '[0; 1)'.*/
        @JvmStatic
        public fun toNormalizedVector(bytes: ByteArray): FloatArray {
            return FloatArray(bytes.size) { ((bytes[it].toInt() and 0xFF)).toFloat() / 255f }
        }

        /** Converts [bytes] to [FloatArray]. */
        @JvmStatic
        public fun toRawVector(bytes: ByteArray): FloatArray {
            return FloatArray(bytes.size) { ((bytes[it].toInt() and 0xFF).toFloat()) }
        }

        /**
         * Takes data located in [trainFeaturesPath], [trainLabelsPath], [testFeaturesPath], [testLabelsPath]
         * with [numClasses], extracts data and labels via [featuresExtractor] and [labelExtractor]
         * to create pair of train and test [OnHeapDataset].
         */
        @JvmStatic
        public fun createTrainAndTestDatasets(
            trainFeaturesPath: String,
            trainLabelsPath: String,
            testFeaturesPath: String,
            testLabelsPath: String,
            numClasses: Int,
            featuresExtractor: (String) -> Array<FloatArray>,
            labelExtractor: (String, Int) -> FloatArray
        ): Pair<OnHeapDataset, OnHeapDataset> {
            return try {
                val xTrain = featuresExtractor(trainFeaturesPath)
                val yTrain = labelExtractor(trainLabelsPath, numClasses)
                val xTest = featuresExtractor(testFeaturesPath)
                val yTest = labelExtractor(testLabelsPath, numClasses)
                Pair(OnHeapDataset(xTrain, yTrain), OnHeapDataset(xTest, yTest))
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        /**
         * Takes data located in [featuresPath], [labelsPath]
         * with [numClasses], extracts data and labels via [featuresExtractor] and [labelExtractor]
         * to create pair of train and test [OnHeapDataset].
         */
        @JvmStatic
        public fun create(
            featuresPath: String,
            labelsPath: String,
            numClasses: Int,
            featuresExtractor: (String) -> Array<FloatArray>,
            labelExtractor: (String, Int) -> FloatArray
        ): OnHeapDataset {
            return try {
                val features = featuresExtractor(featuresPath)
                val labels = labelExtractor(labelsPath, numClasses)

                check(features.size == labels.size) { "The amount of labels is not equal to the amount of images." }

                OnHeapDataset(features, labels)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        /**
         * Takes data from consumers [featuresConsumer] and [labelConsumer]
         * to dataset [OnHeapDataset].
         */
        @JvmStatic
        public fun create(
            featuresConsumer: () -> Array<FloatArray>,
            labelConsumer: () -> FloatArray
        ): OnHeapDataset {
            return try {
                val features = featuresConsumer()
                val labels = labelConsumer()

                check(features.size == labels.size) { "The amount of labels is not equal to the amount of images." }

                OnHeapDataset(features, labels)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        /**
         * Takes data from external data [features] and [labels]
         * to create dataset [OnHeapDataset].
         */
        @JvmStatic
        public fun create(
            features: Array<FloatArray>,
            labels: FloatArray
        ): OnHeapDataset {
            return try {
                check(features.size == labels.size) { "The amount of labels is not equal to the amount of images." }

                OnHeapDataset(features, labels)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        /**
         * Use [preprocessors] and [labels] to prepare data
         * to create dataset [OnHeapDataset].
         */
        @JvmStatic
        public fun create(
            preprocessors: Preprocessing,
            labels: FloatArray
        ): OnHeapDataset {
            return try {
                val loading = preprocessors.imagePreprocessingStage.load
                val xFiles = loading.prepareFileNames()

                val x = prepareX(xFiles, preprocessors, preprocessors.finalShape.numberOfElements.toInt())

                OnHeapDataset(x, labels)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        private fun prepareX(
            xFiles: Array<File>,
            preprocessors: Preprocessing,
            numOfPixels: Int
        ): Array<FloatArray> {
            val x = Array(xFiles.size) { FloatArray(numOfPixels) { 0.0f } }
            for (i in xFiles.indices) {
                x[i] = preprocessors.handleFile(xFiles[i]).first
            }
            return x
        }


        /**
         * Use [preprocessors] to prepare data
         * to create dataset [OnHeapDataset].
         */
        @JvmStatic
        public fun create(
            preprocessors: Preprocessing
        ): OnHeapDataset {
            return try {
                val loading = preprocessors.imagePreprocessingStage.load
                val xFiles = loading.prepareFileNames()
                val numOfPixels = preprocessors.finalShape.numberOfElements.toInt()
                val x = prepareX(xFiles, preprocessors, numOfPixels)
                val y = prepareY(xFiles, preprocessors)

                OnHeapDataset(x, y)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        internal fun prepareY(
            xFiles: Array<File>,
            preprocessors: Preprocessing,
        ): FloatArray {
            val y: FloatArray
            if (preprocessors.imagePreprocessingStage.load.labelGenerator != null) {
                val labelGenerator = preprocessors.imagePreprocessingStage.load.labelGenerator as FromFolders
                val mapping = labelGenerator.mapping

                y = FloatArray(xFiles.size) { 0.0f }
                for (i in xFiles.indices) {
                    y[i] = (mapping[xFiles[i].parentFile.name]
                        ?: error("The parent directory of ${xFiles[i].absolutePath} is ${xFiles[i].parentFile.name}. No such class name in mapping $mapping")).toFloat()
                }

            } else {
                throw IllegalStateException("Label generator should be defined for Loading stage of image preprocessing.")
            }
            return y
        }
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
            copyLabelsToBatch(y, batchStart, batchLength),
            batchLength
        )
    }

    override fun toString(): String = buildStringRepr(x.partialToString(), y.partialToString())

    public fun fullToString(): String = buildStringRepr(x.contentDeepToString(), y.contentToString())

    private fun buildStringRepr(xString: String, yString: String): String =
        "OnHeapDataset(\nx ${x.shape} =\n${xString},\ny [${y.size}] =\n${yString}\n)"
}

/**
 * Create String representation of `FloatArray` where only a part of the data is printed to String.
 *
 * @param maxSize max number of elements of array present in its string representation
 * @param lowPercent percent of data of [maxSize] to be printed from the beginning of array data.
 * Rest will be obtained from the tail of the array in order matching the order in array
 * @return string representation of [FloatArray] in format like
 * `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, ..., 18.0, 19.0]`
 */
private fun FloatArray.partialToString(maxSize: Int = 10, lowPercent: Double = 0.8): String {
    if (size <= maxSize) {
        return contentToString()
    }

    val lowCount = (lowPercent * maxSize).roundToInt()
    val upStart = size - maxSize - 1

    return Array(maxSize + 1) {
        when {
            it < lowCount -> this[it]
            it > lowCount -> this[upStart + it]
            else -> "..."
        }
    }.joinToString(prefix = "[", postfix = "]", separator = ", ")
}

/**
 * Create String representation of `Array<FloatArray>` where only a part of the data is printed to String.
 *
 * @param maxSize max number of elements of array present in its string representation
 * @param lowPercent percent of data of [maxSize] to be printed from the beginning of array data.
 * Rest will be obtained from the tail of the array in order matching the order in array
 * @return string representation of [FloatArray] in format like
 * `[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, ..., 48.0, 49.0],
 *   [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, ..., 148.0, 149.0],
 *   [200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, ..., 248.0, 249.0],
 *   [300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0, ..., 348.0, 349.0],
 *   [400.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, ..., 448.0, 449.0],
 *   [500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0, ..., 548.0, 549.0],
 *   [600.0, 601.0, 602.0, 603.0, 604.0, 605.0, 606.0, 607.0, ..., 648.0, 649.0],
 *   [700.0, 701.0, 702.0, 703.0, 704.0, 705.0, 706.0, 707.0, ..., 748.0, 749.0],
 *   ...,
 *   [4800.0, 4801.0, 4802.0, 4803.0, 4804.0, 4805.0, 4806.0, 4807.0, ..., 4848.0, 4849.0],
 *   [4900.0, 4901.0, 4902.0, 4903.0, 4904.0, 4905.0, 4906.0, 4907.0, ..., 4948.0, 4949.0]]`
 */
private fun Array<FloatArray>.partialToString(maxSize: Int = 10, lowPercent: Double = 0.8): String {
    if (size <= maxSize) {
        return joinToString(prefix = "[", postfix = "]", separator = ",\n ") {
            it.partialToString(maxSize, lowPercent)
        }
    }

    val lowCount = (lowPercent * maxSize).roundToInt()
    val upStart = size - maxSize - 1

    return Array(maxSize + 1) {
        when {
            it < lowCount -> this[it].partialToString(maxSize, lowPercent)
            it > lowCount -> this[upStart + it].partialToString(maxSize, lowPercent)
            else -> "..."
        }
    }.joinToString(prefix = "[", postfix = "]", separator = ",\n ")
}
