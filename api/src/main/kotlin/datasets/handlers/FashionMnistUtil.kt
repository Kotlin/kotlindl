package datasets.handlers

import datasets.Dataset
import java.io.DataInputStream
import java.io.IOException
import java.util.zip.GZIPInputStream

private const val IMAGE_ARCHIVE_MAGIC = 2051
private const val LABEL_ARCHIVE_MAGIC = 2049

const val FASHION_TRAIN_IMAGES_ARCHIVE = "fashionmnist/train-images-idx3-ubyte.gz"
const val FASHION_TRAIN_LABELS_ARCHIVE = "fashionmnist/train-labels-idx1-ubyte.gz"
const val FASHION_TEST_IMAGES_ARCHIVE = "fashionmnist/t10k-images-idx3-ubyte.gz"
const val FASHION_TEST_LABELS_ARCHIVE = "fashionmnist/t10k-labels-idx1-ubyte.gz"


@Throws(IOException::class)
fun extractFashionImages(archiveName: String): Array<FloatArray> {
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

@Throws(IOException::class)
fun extractFashionLabels(archiveName: String, numClasses: Int): Array<FloatArray> {
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