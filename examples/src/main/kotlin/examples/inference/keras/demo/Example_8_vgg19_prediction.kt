/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.keras.demo


import api.core.Sequential
import api.core.loss.Losses
import api.core.metric.Metrics
import api.core.optimizer.Adam
import api.inference.keras.loadWeights
import datasets.Dataset
import datasets.image.ImageConverter
import io.jhdf.HdfFile
import java.io.File

fun main() {
    val jsonConfigFilePath = "C:\\zaleslaw\\home\\models\\vgg19\\modelConfig.json"
    val jsonConfigFile = File(jsonConfigFilePath)
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    val imageNetClassLabels = prepareHumanReadableClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val pathToWeights = "C:\\zaleslaw\\home\\models\\vgg19\\hdf5\\weights.h5"
        val file = File(pathToWeights)
        val hdfFile = HdfFile(file)

        it.loadWeights(hdfFile)

        for (i in 1..8) {
            val inputStream = Dataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val floatArray = ImageConverter.toRawFloatArray(inputStream)

            val res = it.predict(floatArray)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5Labels(it, floatArray, imageNetClassLabels)

            println(top5.toString())
        }
    }
}




