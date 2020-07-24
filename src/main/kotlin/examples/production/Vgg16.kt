package examples.production


import api.inference.keras.loadConfig
import api.keras.dataset.ImageDataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import java.io.File

fun main() {
    val jsonConfigFilePath = "C:\\zaleslaw\\home\\models\\vgg\\modelConfig.json"
    //val jsonConfigRightPath = ImageDataset::class.java.classLoader.getResource(jsonConfigFilePath).path.toString()

    val jsonConfigFile = File(jsonConfigFilePath)

    val model = loadConfig<Float>(jsonConfigFile)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()
        it.loadVariablesFromTxtFiles("C:\\zaleslaw\\home\\models\\vgg\\")

        var floatArray = FloatArray(224 * 224)
        val imageBuffer = ByteArray(224 * 224)

        floatArray =
            ImageDataset.toNormalizedVector(
                imageBuffer
            )

        // TODO: need to rewrite predict and getactivations method for inference model (predict on image)

        val (res, activations) = it.predict(floatArray)
        println(res)

        var weights = it.layers[0].getWeights() // first conv2d layer
        println(weights.size)

        drawFilters(weights[0])

        var weights4 = it.layers[4].getWeights() // first conv2d layer
        println(weights4.size)

        drawFilters(weights4[0])
    }
}




