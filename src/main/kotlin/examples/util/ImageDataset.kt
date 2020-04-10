package tensorflow.training.util

import java.io.DataInputStream
import java.io.IOException
import java.nio.FloatBuffer
import java.util.*
import java.util.zip.GZIPInputStream
import kotlin.experimental.and

class ImageDataset private constructor(
    private val trainingImages: Array<FloatArray>,
    private val trainingLabels: Array<FloatArray>,
    private val validationImages: Array<FloatArray>,
    private val validationLabels: Array<FloatArray>,
    private val testImages: Array<FloatArray>,
    private val testLabels: Array<FloatArray>
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
            val size = Math.min(batchSize, images.size - batchStart)
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

    fun trainingBatchIterator(batchSize: Int): ImageBatchIterator {
        return ImageBatchIterator(batchSize, trainingImages, trainingLabels)
    }

    fun validationBatchIterator(batchSize: Int): ImageBatchIterator {
        return ImageBatchIterator(batchSize, validationImages, validationLabels)
    }

    fun testBatch(): ImageBatch {
        return ImageBatch(
            serializeToBuffer(
                testImages,
                0,
                testImages.size
            ),
            serializeToBuffer(
                testLabels,
                0,
                testLabels.size
            ),
            testImages.size
        )
    }

    companion object {
        fun create(validationSize: Int): ImageDataset {
            return try {
                val trainImages =
                    extractImages(
                        TRAIN_IMAGES_ARCHIVE
                    )
                val trainLabels =
                    extractLabels(
                        TRAIN_LABELS_ARCHIVE,
                        NUM_CLASSES
                    )
                val testImages =
                    extractImages(
                        TEST_IMAGES_ARCHIVE
                    )
                val testLabels =
                    extractLabels(
                        TEST_LABELS_ARCHIVE,
                        NUM_CLASSES
                    )
                if (validationSize > 0) {
                    ImageDataset(
                        Arrays.copyOfRange(trainImages, validationSize, trainImages.size),
                        Arrays.copyOfRange(trainLabels, validationSize, trainLabels.size),
                        Arrays.copyOfRange(trainImages, 0, validationSize),
                        Arrays.copyOfRange(trainLabels, 0, validationSize),
                        testImages,
                        testLabels
                    )
                } else ImageDataset(
                    trainImages,
                    trainLabels,
                    arrayOf(),
                    arrayOf(),
                    testImages,
                    testLabels
                )
            } catch (e: IOException) {
                throw AssertionError(e)
            }
        }

        private const val TRAIN_IMAGES_ARCHIVE = "train-images-idx3-ubyte.gz"
        private const val TRAIN_LABELS_ARCHIVE = "train-labels-idx1-ubyte.gz"
        private const val TEST_IMAGES_ARCHIVE = "t10k-images-idx3-ubyte.gz"
        private const val TEST_LABELS_ARCHIVE = "t10k-labels-idx1-ubyte.gz"
        private const val NUM_CLASSES = 10
        private const val IMAGE_ARCHIVE_MAGIC = 2051
        private const val LABEL_ARCHIVE_MAGIC = 2049
        @Throws(IOException::class)
        private fun extractImages(archiveName: String): Array<FloatArray> {
            val archiveStream = DataInputStream(
                GZIPInputStream(
                    ImageDataset::class.java.classLoader.getResourceAsStream(archiveName)
                )
            )
            val magic = archiveStream.readInt()
            require(IMAGE_ARCHIVE_MAGIC == magic) { "\"$archiveName\" is not a valid image archive" }
            val imageCount = archiveStream.readInt()
            val imageRows = archiveStream.readInt()
            val imageCols = archiveStream.readInt()
            println(
                String.format(
                    "Extracting %d images of %dx%d from %s",
                    imageCount,
                    imageRows,
                    imageCols,
                    archiveName
                )
            )
            val images =
                Array(imageCount) { FloatArray(imageRows * imageCols) }
            val imageBuffer = ByteArray(imageRows * imageCols)
            for (i in 0 until imageCount) {
                archiveStream.readFully(imageBuffer)
                images[i] =
                    toNormalizedVector(
                        imageBuffer
                    )
            }
            return images
        }

        @Throws(IOException::class)
        private fun extractLabels(archiveName: String, numClasses: Int): Array<FloatArray> {
            val archiveStream = DataInputStream(
                GZIPInputStream(
                    ImageDataset::class.java.classLoader.getResourceAsStream(archiveName)
                )
            )
            val magic = archiveStream.readInt()
            require(LABEL_ARCHIVE_MAGIC == magic) { "\"$archiveName\" is not a valid image archive" }
            val labelCount = archiveStream.readInt()
            println(String.format("Extracting %d labels from %s", labelCount, archiveName))
            val labelBuffer = ByteArray(labelCount)
            archiveStream.readFully(labelBuffer)
            val floats =
                Array(labelCount) { FloatArray(10) }
            for (i in 0 until labelCount) {
                floats[i] =
                    toOneHotVector(
                        10,
                        labelBuffer[i]
                    )
            }
            return floats
        }

        private fun toOneHotVector(numClasses: Int, label: Byte): FloatArray {
            val buf = FloatBuffer.allocate(numClasses)
            buf.put((label and 0xFF.toByte()).toInt(), 1.0f)
            return buf.array()
        }

        private fun toNormalizedVector(bytes: ByteArray): FloatArray {
            val floats = FloatArray(bytes.size)
            for (i in bytes.indices) {
                floats[i] = (bytes[i] and 0xFF.toByte()).toFloat() / 255.0f
            }
            return floats
        }

        private fun serializeToBuffer(src: Array<FloatArray>, start: Int, length: Int): FloatBuffer {
            val buffer = FloatBuffer.allocate(length * src[0].size)
            for (i in start until start + length) {
                buffer.put(src[i])
            }
            return buffer.rewind() as FloatBuffer
        }
    }

}