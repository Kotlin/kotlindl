package datasets

import java.io.IOException
import java.nio.Buffer
import java.nio.FloatBuffer
import kotlin.math.min
import kotlin.math.truncate

class Dataset internal constructor(
    private val x: Array<FloatArray>,
    private val y: Array<FloatArray>
) {
    inner class BatchIterator internal constructor(
        private val batchSize: Int,
        private val batchX: Array<FloatArray>,
        private val batchY: Array<FloatArray>
    ) :
        MutableIterator<DataBatch?> {
        override fun hasNext(): Boolean {
            return batchStart < totalSize()
        }

        override fun next(): DataBatch {
            val size = min(batchSize, batchX.size - batchStart)
            val batch = DataBatch(
                serializeToBuffer(
                    batchX,
                    batchStart,
                    size
                ),
                serializeToBuffer(
                    batchY,
                    batchStart,
                    size
                ),
                size
            )
            batchStart += batchSize
            return batch
        }

        private var batchStart = 0
        private fun totalSize(): Int {
            return batchX.size
        }

        override fun remove() {
            throw UnsupportedOperationException("The removal operation is not supported!")
        }

    }

    fun batchIterator(batchSize: Int): BatchIterator {
        return BatchIterator(batchSize, x, y)
    }

    fun split(splitRatio: Double): Pair<Dataset, Dataset> {
        require(splitRatio in 0.0..1.0) { "'Split ratio' argument value must be in range [0.0; 1.0]." }

        val trainDatasetLastIndex = truncate(x.size * splitRatio).toInt()

        return Pair(
            Dataset(
                x.copyOfRange(0, trainDatasetLastIndex),
                y.copyOfRange(0, trainDatasetLastIndex)
            ),
            Dataset(
                x.copyOfRange(trainDatasetLastIndex, x.size),
                y.copyOfRange(trainDatasetLastIndex, y.size)
            )
        )
    }

    companion object {
        fun toOneHotVector(numClasses: Int, label: Byte): FloatArray {
            val buf = FloatBuffer.allocate(numClasses)
            buf.put((label.toInt() and 0xFF), 1.0f)
            return buf.array()
        }

        fun toNormalizedVector(bytes: ByteArray): FloatArray {
            val floats = FloatArray(bytes.size)
            for (i in bytes.indices) {
                floats[i] = ((bytes[i].toInt() and 0xFF).toFloat()) / 255.0f
            }
            return floats
        }

        fun toRawVector(bytes: ByteArray): FloatArray {
            val floats = FloatArray(bytes.size)
            for (i in bytes.indices) {
                floats[i] = ((bytes[i].toInt() and 0xFF).toFloat())
            }
            return floats
        }

        fun serializeToBuffer(src: Array<FloatArray>, start: Int, length: Int): FloatBuffer {
            val buffer = FloatBuffer.allocate(length * src[0].size)
            for (i in start until start + length) {
                buffer.put(src[i])
            }
            return (buffer as Buffer).rewind() as FloatBuffer
        }

        fun createTrainAndTestDatasets(
            trainFeaturesPath: String,
            trainLabelsPath: String,
            testFeaturesPath: String,
            testLabelsPath: String,
            numClasses: Int,
            featuresExtractor: (String) -> Array<FloatArray>,
            labelExtractor: (String, Int) -> Array<FloatArray>
        ): Pair<Dataset, Dataset> {
            return try {
                val xTrain =
                    featuresExtractor.invoke(
                        trainFeaturesPath
                    )
                val yTrain =
                    labelExtractor.invoke(
                        trainLabelsPath,
                        numClasses
                    )
                val xTest =
                    featuresExtractor.invoke(
                        testFeaturesPath
                    )
                val yTest =
                    labelExtractor.invoke(
                        testLabelsPath,
                        numClasses
                    )
                Pair(Dataset(xTrain, yTrain), Dataset(xTest, yTest))
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        fun create(
            featuresPath: String,
            labelsPath: String,
            numClasses: Int,
            featuresExtractor: (String) -> Array<FloatArray>,
            labelExtractor: (String, Int) -> Array<FloatArray>
        ): Dataset {
            return try {
                val features =
                    featuresExtractor.invoke(
                        featuresPath
                    )
                val labels =
                    labelExtractor.invoke(
                        labelsPath,
                        numClasses
                    )

                check(features.size == labels.size) { "The amount of labels is not equal to the amount of images." }

                Dataset(features, labels)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        fun create(
            featuresConsumer: () -> Array<FloatArray>,
            labelConsumer: () -> Array<FloatArray>
        ): Dataset {
            return try {
                val features = featuresConsumer.invoke()
                val labels = labelConsumer.invoke()

                check(features.size == labels.size) { "The amount of labels is not equal to the amount of images." }

                Dataset(features, labels)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }
    }

    fun xSize(): Int {
        return x.size
    }

    fun getX(idx: Int): FloatArray {
        return x[idx]
    }

    fun getY(idx: Int): FloatArray {
        return y[idx]
    }
}