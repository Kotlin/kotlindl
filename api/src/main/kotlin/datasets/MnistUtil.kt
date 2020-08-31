package datasets

import api.keras.dataset.Dataset
import java.io.DataInputStream
import java.util.zip.GZIPInputStream

private const val IMAGE_ARCHIVE_MAGIC = 2051
private const val LABEL_ARCHIVE_MAGIC = 2049

const val TRAIN_IMAGES_ARCHIVE = "train-images-idx3-ubyte.gz"
const val TRAIN_LABELS_ARCHIVE = "train-labels-idx1-ubyte.gz"
const val TEST_IMAGES_ARCHIVE = "t10k-images-idx3-ubyte.gz"
const val TEST_LABELS_ARCHIVE = "t10k-labels-idx1-ubyte.gz"
const val AMOUNT_OF_CLASSES = 10

fun extractImages(archiveName: String): Array<FloatArray> {
    val archiveStream = DataInputStream(
        GZIPInputStream(
            Dataset::class.java.classLoader.getResourceAsStream(archiveName)
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
            Dataset.toNormalizedVector(
                imageBuffer
            )
    }
    return images
}

fun extractLabels(archiveName: String, numClasses: Int): Array<FloatArray> {
    val archiveStream = DataInputStream(
        GZIPInputStream(
            Dataset::class.java.classLoader.getResourceAsStream(archiveName)
        )
    )
    val magic = archiveStream.readInt()
    require(LABEL_ARCHIVE_MAGIC == magic) { "\"$archiveName\" is not a valid image archive" }
    val labelCount = archiveStream.readInt()
    println(String.format("Extracting %d labels from %s", labelCount, archiveName))
    val labelBuffer = ByteArray(labelCount)
    archiveStream.readFully(labelBuffer)
    val floats =
        Array(labelCount) { FloatArray(numClasses) }
    for (i in 0 until labelCount) {
        floats[i] =
            Dataset.toOneHotVector(
                numClasses,
                labelBuffer[i]
            )
    }
    return floats
}