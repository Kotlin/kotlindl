package examples.keras.cifar10.util

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import tf_api.keras.dataset.ImageDataset
import java.io.DataInputStream
import java.io.IOException
import java.util.zip.GZIPInputStream

private const val IMAGE_ARCHIVE_MAGIC = 2051
private const val LABEL_ARCHIVE_MAGIC = 2049

// TODO: implement the method
@Throws(IOException::class)
fun extractCifar10Images(archiveName: String): Array<FloatArray> {
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
fun extractCifar10Labels(pathToLabels: String, numClasses: Int): Array<FloatArray> {
    val labelCount = 50000
    println(String.format("Extracting %d labels from %s", labelCount, pathToLabels))
    val labelBuffer = ByteArray(labelCount)

    val dictionary = mapOf(
        "airplane" to 1, "automobile" to 2, "bird" to 3, "cat" to 4, "deer" to 5, "dog" to 6, "frog" to 7,
        "horse" to 8,
        "ship" to 9,
        "truck" to 10
    )

    var cnt = 0
    csvReader().open(pathToLabels) {
        readAllAsSequence().forEach { row ->
            labelBuffer[cnt] = dictionary.getOrElse(row[1]) { 1 }.toByte()
            cnt++
        }
    }

    val floats =
        Array(labelCount) { FloatArray(numClasses) }
    for (i in 0 until labelCount) {
        floats[i] =
            ImageDataset.toOneHotVector(
                numClasses,
                labelBuffer[i]
            )
    }
    return floats
}