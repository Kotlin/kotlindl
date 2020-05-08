package examples.keras.mnist.util

import tf_api.keras.dataset.ImageDataset
import java.io.DataInputStream
import java.io.IOException
import java.util.zip.GZIPInputStream

private const val IMAGE_ARCHIVE_MAGIC = 2051
private const val LABEL_ARCHIVE_MAGIC = 2049

const val TRAIN_IMAGES_ARCHIVE = "train-images-idx3-ubyte.gz"
const val TRAIN_LABELS_ARCHIVE = "train-labels-idx1-ubyte.gz"
const val TEST_IMAGES_ARCHIVE = "t10k-images-idx3-ubyte.gz"
const val TEST_LABELS_ARCHIVE = "t10k-labels-idx1-ubyte.gz"
const val VALIDATION_SIZE = 0
const val NUM_LABELS = 10

@Throws(IOException::class)
fun extractImages(archiveName: String): Array<FloatArray> {
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
            ImageDataset.toNormalizedVector(
                imageBuffer
            )
    }
    return images
}

@Throws(IOException::class)
fun extractLabels(archiveName: String, numClasses: Int): Array<FloatArray> {
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
            ImageDataset.toOneHotVector(
                10,
                labelBuffer[i]
            )
    }
    return floats
}