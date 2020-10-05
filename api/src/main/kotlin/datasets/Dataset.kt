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

    public fun batchIterator(batchSize: Int): BatchIterator {
        return BatchIterator(batchSize, x, y)
    }

    public fun split(splitRatio: Double): Pair<Dataset, Dataset> {
        require(splitRatio in 0.0..1.0) { "'Split ratio' argument value must be in range [0.0; 1.0]." }

        val trainDatasetLastIndex = truncate(x.size * splitRatio).toInt()

        return Pair(
            Dataset(x.copyOfRange(0, trainDatasetLastIndex), y.copyOfRange(0, trainDatasetLastIndex)),
            Dataset(x.copyOfRange(trainDatasetLastIndex, x.size), y.copyOfRange(trainDatasetLastIndex, y.size))
        )
    }

    public companion object {
        public fun toOneHotVector(numClasses: Int, label: Byte): FloatArray {
            val ret = FloatArray(numClasses)
            ret[label.toInt() and 0xFF] = 1f
            return ret
        }

        public fun toNormalizedVector(bytes: ByteArray): FloatArray {
            return FloatArray(bytes.size) { ((bytes[it].toInt() and 0xFF)) / 255f }
        }

        public fun toRawVector(bytes: ByteArray): FloatArray {
            return FloatArray(bytes.size) { ((bytes[it].toInt() and 0xFF).toFloat()) }
        }

        public fun serializeToBuffer(src: Array<FloatArray>, start: Int, length: Int): FloatBuffer {
            val buffer = FloatBuffer.allocate(length * src[0].size)
            for (i in start until start + length) {
                buffer.put(src[i])
            }
            return (buffer as Buffer).rewind() as FloatBuffer
        }

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

    public fun xSize(): Int {
        return x.size
    }

    public fun getX(idx: Int): FloatArray {
        return x[idx]
    }

    public fun getY(idx: Int): FloatArray {
        return y[idx]
    }

    public fun getLabel(idx: Int): Int {
        val labelArray = y[idx]
        return labelArray.indexOf(labelArray.max()!!)

    }
}