/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.keras.vgg


import api.core.Sequential
import api.core.loss.Losses
import api.core.metric.Metrics
import api.core.optimizer.Adam
import datasets.Dataset
import datasets.image.ImageConverter
import examples.production.drawActivations
import java.io.File

/** Loads weights from .txt files especially prepared. */
fun main() {
    val jsonConfigFilePath = "C:\\zaleslaw\\home\\models\\vgg\\modelConfig.json"
    val jsonConfigFile = File(jsonConfigFilePath)

    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()
        println(it.kGraph)
        it.loadWeights(File("C:\\zaleslaw\\home\\models\\vgg\\"))

        for (i in 1..8) {
            val inputStream = Dataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val floatArray = ImageConverter.toRawFloatArray(inputStream)

            // TODO: need to rewrite predict and getactivations method for inference model (predict on image)
            val (res, activations) = it.predictAndGetActivations(floatArray)
            println(res)
            drawActivations(activations)

            val predictionVector = it.predictSoftly(floatArray).toMutableList()
            val predictionVector2 = it.predictSoftly(floatArray).toMutableList()


            val top5: MutableMap<Int, Int> = mutableMapOf()
            for (j in 1..5) {
                val max = predictionVector2.maxOrNull()
                val indexOfElem = predictionVector.indexOf(max!!)
                top5[j] = indexOfElem
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






