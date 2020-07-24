package examples.inference.keras


import api.inference.keras.loadConfig
import api.inference.keras.loadWeights
import api.inference.keras.recursivePrintGroup
import api.keras.dataset.ImageDataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import examples.production.drawFilters
import io.jhdf.HdfFile
import java.io.File


/**
 * This example doesn't work due to problem with reading from hdf5 file.
 *
 * Exception in thread "main" io.jhdf.exceptions.HdfException: Failed to map data buffer for dataset '/block1_conv1/block1_conv1/kernel:0'
at io.jhdf.dataset.ContiguousDatasetImpl.getDataBuffer(ContiguousDatasetImpl.java:44)
at io.jhdf.dataset.DatasetBase.getData(DatasetBase.java:121)
 */
fun main() {
    val pathToConfig = "models/imagenet/vgg.json"
    val realPathToConfig = ImageDataset::class.java.classLoader.getResource(pathToConfig).path.toString()
    val jsonConfigFile = File(realPathToConfig)

    val model = loadConfig<Float>(jsonConfigFile)

    val weightsHDFFilePath = "C:\\zaleslaw\\home\\models\\vgg\\hdf\\weights.h5"
    val file = File(weightsHDFFilePath)
    val hdfFile = HdfFile(file)
    model.use {
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()

        hdfFile.use { hdfFile ->
            recursivePrintGroup(hdfFile, hdfFile, 0)
        }

        it.loadWeights(hdfFile)


        var weights = it.layers[0].getWeights() // first conv2d layer

        drawFilters(weights[0])

    }
}




