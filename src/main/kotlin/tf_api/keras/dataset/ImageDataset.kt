package tf_api.keras.dataset

import java.io.IOException
import java.nio.FloatBuffer
import kotlin.experimental.and
import kotlin.math.min
import kotlin.math.truncate

class ImageDataset internal constructor(
    private val images: Array<FloatArray>,
    private val labels: Array<FloatArray>
) {
    inner class ImageBatchIterator internal constructor(
        private val batchSize: Int,
        private val images: Array<FloatArray>,
        private val labels: Array<FloatArray>
    ) :
        MutableIterator<ImageBatch?> {
        override fun hasNext(): Boolean {
            return batchStart < totalSize()
        }

        override fun next(): ImageBatch {
            val size = min(batchSize, images.size - batchStart)
            val batch = ImageBatch(
                serializeToBuffer(
                    images,
                    batchStart,
                    size
                ),
                serializeToBuffer(
                    labels,
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
            return images.size
        }

        override fun remove() {
            TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
        }

    }

    fun batchIterator(batchSize: Int): ImageBatchIterator {
        return ImageBatchIterator(batchSize, images, labels)
    }

    fun split(validationRate: Double): Pair<ImageDataset, ImageDataset> {
        // TODO: convert to exception
        assert(validationRate in 0.0..1.0)

        val trainDatasetLastIndex = truncate(images.size * validationRate).toInt()

        return Pair(
            ImageDataset(
                images.copyOfRange(0, trainDatasetLastIndex),
                labels.copyOfRange(0, trainDatasetLastIndex)
            ),
            ImageDataset(
                images.copyOfRange(trainDatasetLastIndex, images.size),
                labels.copyOfRange(trainDatasetLastIndex, labels.size)
            )
        )
    }

    companion object {
        fun toOneHotVector(numClasses: Int, label: Byte): FloatArray {
            val buf = FloatBuffer.allocate(numClasses)
            buf.put((label and 0xFF.toByte()).toInt(), 1.0f)
            return buf.array()
        }

        fun toNormalizedVector(bytes: ByteArray): FloatArray {
            val floats = FloatArray(bytes.size)
            for (i in bytes.indices) {
                floats[i] = (bytes[i] and 0xFF.toByte()).toFloat() / 255.0f
            }
            return floats
        }

        fun serializeToBuffer(src: Array<FloatArray>, start: Int, length: Int): FloatBuffer {
            val buffer = FloatBuffer.allocate(length * src[0].size)
            for (i in start until start + length) {
                buffer.put(src[i])
            }
            return buffer.rewind() as FloatBuffer
        }

        fun createTrainAndTestDatasets(
            trainImagesPath: String,
            trainLabelsPath: String,
            testImagesPath: String,
            testLabelsPath: String,
            numClasses: Int,
            imageExtractor: (String) -> Array<FloatArray>,
            labelExtractor: (String, Int) -> Array<FloatArray>
        ): Pair<ImageDataset, ImageDataset> {
            return try {
                val trainImages =
                    imageExtractor.invoke(
                        trainImagesPath
                    )
                val trainLabels =
                    labelExtractor.invoke(
                        trainLabelsPath,
                        numClasses
                    )
                val testImages =
                    imageExtractor.invoke(
                        testImagesPath
                    )
                val testLabels =
                    labelExtractor.invoke(
                        testLabelsPath,
                        numClasses
                    )
                Pair(ImageDataset(trainImages, trainLabels), ImageDataset(testImages, testLabels))
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        fun create(
            imagesPath: String,
            labelsPath: String,
            numClasses: Int,
            imageExtractor: (String) -> Array<FloatArray>,
            labelExtractor: (String, Int) -> Array<FloatArray>
        ): ImageDataset {
            return try {
                val images =
                    imageExtractor.invoke(
                        imagesPath
                    )
                val labels =
                    labelExtractor.invoke(
                        labelsPath,
                        numClasses
                    )
                // TODO: convert to exception
                assert(images.size == labels.size)

                ImageDataset(images, labels)
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }
    }

    fun imagesSize(): Int {
        return images.size
    }

    fun getImage(idx: Int): FloatArray {
        return images[idx]
    }

    fun getImageLabel(idx: Int): FloatArray {
        return labels[idx]
    }
}