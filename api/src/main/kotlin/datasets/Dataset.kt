/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package datasets

import java.io.IOException
import java.nio.Buffer
import java.nio.FloatBuffer
import kotlin.math.min
import kotlin.math.truncate

/**
 * Basic class to handle features [x] and labels [y].
 *
 * NOTE: Labels [y] should have shape <number of rows; number of labels> and contain exactly one 1 and other 0-es per row to be result of one-hot-encoding.
 */
public class Dataset internal constructor(private val x: Array<FloatArray>, private val y: Array<FloatArray>) {
    /**
     * An iterator over a [Dataset].
     */
    public inner class BatchIterator internal constructor(
        private val batchSize: Int, private val batchX: Array<FloatArray>, private val batchY: Array<FloatArray>
    ) : Iterator<DataBatch?> {

        private var batchStart = 0

        override fun hasNext(): Boolean = batchStart < batchX.size

        override fun next(): DataBatch {
            val size = min(batchSize, batchX.size - batchStart)
            val batch = DataBatch(
                serializeToBuffer(batchX, batchStart, size),
                serializeToBuffer(batchY, batchStart, size),
                size
            )
            batchStart += batchSize
            return batch
        }
    }

    /** Returns [BatchIterator] with fixed [batchSize]. */
    public fun batchIterator(batchSize: Int): BatchIterator {
        return BatchIterator(batchSize, x, y)
    }

    /** Splits datasets on two sub-datasets according [splitRatio].*/
    public fun split(splitRatio: Double): Pair<Dataset, Dataset> {
        require(splitRatio in 0.0..1.0) { "'Split ratio' argument value must be in range [0.0; 1.0]." }

        val trainDatasetLastIndex = truncate(x.size * splitRatio).toInt()

        return Pair(
            Dataset(x.copyOfRange(0, trainDatasetLastIndex), y.copyOfRange(0, trainDatasetLastIndex)),
            Dataset(x.copyOfRange(trainDatasetLastIndex, x.size), y.copyOfRange(trainDatasetLastIndex, y.size))
        )
    }

    public companion object {
        /** Creates binary vector with size [numClasses] from [label]. */
        public fun toOneHotVector(numClasses: Int, label: Byte): FloatArray {
            val ret = FloatArray(numClasses)
            ret[label.toInt() and 0xFF] = 1f
            return ret
        }

        /** Normalizes [bytes] via division on 255 to get values in range '[0; 1)'.*/
        public fun toNormalizedVector(bytes: ByteArray): FloatArray {
            return FloatArray(bytes.size) { ((bytes[it].toInt() and 0xFF)) / 255f }
        }

        /** Converts [bytes] to [FloatArray]. */
        public fun toRawVector(bytes: ByteArray): FloatArray {
            return FloatArray(bytes.size) { ((bytes[it].toInt() and 0xFF).toFloat()) }
        }

        /** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
        public fun serializeToBuffer(src: Array<FloatArray>, start: Int, length: Int): FloatBuffer {
            val buffer = FloatBuffer.allocate(length * src[0].size)
            for (i in start until start + length) {
                buffer.put(src[i])
            }
            return (buffer as Buffer).rewind() as FloatBuffer
        }

        /**
         * Takes data located in [trainFeaturesPath], [trainLabelsPath], [testFeaturesPath], [testLabelsPath]
         * with [numClasses], extracts data and labels via [featuresExtractor] and [labelExtractor]
         * to create pair of train and test [Dataset].
         */
        public fun createTrainAndTestDatasets(
            trainFeaturesPath: String,
            trainLabelsPath: String,
            testFeaturesPath: String,
            testLabelsPath: String,
            numClasses: Int,
            featuresExtractor: (String) -> Array<FloatArray>,
            labelExtractor: (String, Int) -> Array<FloatArray>
        ): Pair<Dataset, Dataset> {
            return try {
                val xTrain = featuresExtractor(trainFeaturesPath)
                val yTrain = labelExtractor(trainLabelsPath, numClasses)
                val xTest = featuresExtractor(testFeaturesPath)
                val yTest = labelExtractor(testLabelsPath, numClasses)
                Pair(Dataset(xTrain, yTrain), Dataset(xTest, yTest))
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        /**
         * Takes data located in [featuresPath], [labelsPath]
         * with [numClasses], extracts data and labels via [featuresExtractor] and [labelExtractor]
         * to create pair of train and test [Dataset].
         */
        public fun create(
            featuresPath: String,
            labelsPath: String,
            numClasses: Int,
            featuresExtractor: (String) -> Array<FloatArray>,
            labelExtractor: (String, Int) -> Array<FloatArray>
        ): Dataset {
            return try {
                val features = featuresExtractor(featuresPath)
                val labels = labelExtractor(labelsPath, numClasses)

                check(features.size == labels.size) { "The amount of labels is not equal to the amount of images." }

                Dataset(features, labels)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        /**
         * Takes data from consumers [featuresConsumer] and [labelConsumer]
         * to create pair of train and test [Dataset].
         */
        public fun create(
            featuresConsumer: () -> Array<FloatArray>,
            labelConsumer: () -> Array<FloatArray>
        ): Dataset {
            return try {
                val features = featuresConsumer()
                val labels = labelConsumer()

                check(features.size == labels.size) { "The amount of labels is not equal to the amount of images." }

                Dataset(features, labels)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }
    }

    /** Returns amount of data rows. */
    public fun xSize(): Int {
        return x.size
    }

    /** Returns row by index [idx]. */
    public fun getX(idx: Int): FloatArray {
        return x[idx]
    }

    /** Returns label as [FloatArray] by index [idx]. */
    public fun getY(idx: Int): FloatArray {
        return y[idx]
    }

    /** Returns label as [Int] by index [idx]. */
    public fun getLabel(idx: Int): Int {
        val labelArray = y[idx]
        return labelArray.indexOfFirst { it == labelArray.maxOrNull()!! }
    }
}