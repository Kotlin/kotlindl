package examples.inference.keras.vgg


import api.core.Sequential
import api.core.loss.LossFunctions
import api.core.metric.Metrics
import api.core.optimizer.Adam
import api.inference.keras.loadWeights
import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import datasets.Dataset
import io.jhdf.HdfFile
import java.io.File

fun main() {
    val jsonConfigFilePath = "C:\\zaleslaw\\home\\models\\vgg\\modelConfig.json"
    val jsonConfigFile = File(jsonConfigFilePath)
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    val pathToIndices = "/datasets/vgg/imagenet_class_index.json"

    fun parse(name: String): Any? {
        val cls = Parser::class.java
        return cls.getResourceAsStream(name)?.let { inputStream ->
            return Parser.default().parse(inputStream, Charsets.UTF_8)
        }
    }

    val classIndices = parse(pathToIndices) as JsonObject

    val imageNetClassIndices = mutableMapOf<Int, String>()

    for (key in classIndices.keys) {
        imageNetClassIndices[key.toInt()] = (classIndices[key] as JsonArray<*>)[1].toString()
    }

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()
        println(it.kGraph)
        val pathToWeights = "C:\\zaleslaw\\home\\models\\vgg\\hdf\\weights.h5"
        val file = File(pathToWeights)
        val hdfFile = HdfFile(file)

        val kernelDataPathTemplate = "/%s/%s_1/kernel:0"
        val biasDataPathTemplate = "/%s/%s_1/bias:0"
        it.loadWeights(hdfFile, kernelDataPathTemplate, biasDataPathTemplate)


        for (i in 1..8) {
            val inputStream = Dataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val floatArray = loadImageAndConvertToFloatArray(inputStream)

            // TODO: need to rewrite predict and getactivations method for inference model (predict on image)

            val (res, activations) = it.predictAndGetActivations(floatArray)
            println("Predicted object for image$i.jpg is ${imageNetClassIndices[res]}")
            //drawActivations(activations)

            val predictionVector = it.predictSoftly(floatArray, "Softmax").toMutableList()
            val predictionVector2 =
                it.predictSoftly(floatArray, "Softmax").toMutableList() // get copy of previous vector

            val top5: MutableMap<Int, Pair<String, Float>> = mutableMapOf()
            for (j in 1..5) {
                val max = predictionVector2.maxOrNull()
                val indexOfElem = predictionVector.indexOf(max!!)
                top5[j] = Pair(imageNetClassIndices[indexOfElem]!!, predictionVector[indexOfElem])
                predictionVector2.remove(max)
            }

            println(top5.toString())
        }


        /*var weights = it.layers[0].getWeights() // first conv2d layer
        println(weights.size)

        drawFilters(weights[0])

        var weights4 = it.layers[4].getWeights() // first conv2d layer
        println(weights4.size)

        drawFilters(weights4[0])*/
    }
}




