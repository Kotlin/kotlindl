/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelzoo.resnet.resnet50

import examples.transferlearning.modelzoo.vgg16.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.RMSProp
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelZoo
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.predictTop5Labels
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.imagePreprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocessingPipeline
import java.io.File

private const val PATH_TO_MODEL = "savedmodels/resnet50_1"
private const val PATH_TO_MODEL_2 = "savedmodels/resnet50_2"

fun main() {
    val modelZoo =
        ModelZoo(commonModelDirectory = File("cache/pretrainedModels"), modelType = ModelType.ResNet_50)
    val model = modelZoo.loadModel() as Functional

    val imageNetClassLabels = modelZoo.loadClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val hdfFile = modelZoo.loadWeights()

        it.loadWeights(hdfFile)

        for (i in 1..8) {
            val preprocessing: Preprocessing = preprocessingPipeline {
                imagePreprocessing {
                    load {
                        pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                        imageShape = ImageShape(224, 224, 3)
                        colorMode = ColorOrder.BGR
                    }
                }
            }

            val inputData = modelZoo.preprocessInput(preprocessing().first, model.inputDimensions)


            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5Labels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }

        /*

        Predicted object for image1.jpg is neck_brace
{1=(neck_brace, 0.2446676), 2=(jersey, 0.21495727), 3=(lab_coat, 0.063993305), 4=(Windsor_tie, 0.030763602), 5=(sweatshirt, 0.02347603)}
Predicted object for image2.jpg is water_ouzel
{1=(water_ouzel, 0.28694847), 2=(partridge, 0.23358065), 3=(quail, 0.13337053), 4=(black_grouse, 0.082313), 5=(ptarmigan, 0.08104088)}
Predicted object for image3.jpg is Egyptian_cat
{1=(Egyptian_cat, 0.89821666), 2=(tabby, 0.05102), 3=(tiger_cat, 0.048863027), 4=(lynx, 8.701493E-4), 5=(swab, 1.590161E-4)}
Predicted object for image4.jpg is sports_car
{1=(sports_car, 0.6798361), 2=(grille, 0.15290835), 3=(car_wheel, 0.08073072), 4=(convertible, 0.037038352), 5=(beach_wagon, 0.022067564)}
Predicted object for image5.jpg is broccoli
{1=(broccoli, 0.91358185), 2=(cauliflower, 0.08460309), 3=(pot, 7.0783374E-4), 4=(bonnet, 4.270178E-4), 5=(head_cabbage, 1.00912235E-4)}
Predicted object for image6.jpg is tench
{1=(tench, 0.2759103), 2=(rock_beauty, 0.25171873), 3=(eel, 0.06595011), 4=(axolotl, 0.058189172), 5=(electric_ray, 0.05162542)}
Predicted object for image7.jpg is hog
{1=(hog, 0.82425535), 2=(wild_boar, 0.16144408), 3=(sloth_bear, 0.0057242136), 4=(American_black_bear, 0.0030997216), 5=(brown_bear, 0.0022617055)}
Predicted object for image8.jpg is goldfish
{1=(goldfish, 0.87598634), 2=(rock_beauty, 0.041129958), 3=(barracouta, 0.026740275), 4=(tench, 0.025140727), 5=(puffer, 0.009031909)}

         */

        it.save(
            File(PATH_TO_MODEL),
            savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
            writingMode = WritingMode.OVERRIDE
        )

        it.save(
            File(PATH_TO_MODEL_2),
            savingFormat = SavingFormat.TF_GRAPH_CUSTOM_VARIABLES,
            writingMode = WritingMode.OVERRIDE
        )
    }

    /*val inferenceModel = InferenceModel.load(File(PATH_TO_MODEL_2))

    inferenceModel.use {
        for (i in 1..8) {
            val inputStream = Dataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val floatArray = ImageConverter.toRawFloatArray(inputStream)

            val xTensorShape = it.inputLayer.input.asOutput().shape()
            val tensorShape = longArrayOf(
                1,
                *tail(xTensorShape)
            )

            val inputData = preprocessInput(floatArray, tensorShape, inputType = InputType.CAFFE)
            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5Labels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }*/

    val model2 = Functional.loadModelConfiguration(File("$PATH_TO_MODEL/modelConfig.json"))

    model2.use {
        it.compile(
            optimizer = RMSProp(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.summary()

        it.loadWeights(File(PATH_TO_MODEL))

        for (i in 1..8) {
            val preprocessing: Preprocessing = preprocessingPipeline {
                imagePreprocessing {
                    load {
                        pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                        imageShape = ImageShape(224, 224, 3)
                        colorMode = ColorOrder.BGR
                    }
                }
            }

            val inputData = modelZoo.preprocessInput(preprocessing().first, model2.inputDimensions)
            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5Labels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }

        /* TODO: check it
TOO SMALL, but correct signals, what is happened? something wrong with final logits?
Different results are related to lossfunctions and logits applications at the end (need to check for all loaded models and recommend loss functions for training

In prediction MAE usually is used, but for training is softmax is used and this is a root of difference


Predicted object for image1.jpg is neck_brace
{1=(neck_brace, 0.0012758446), 2=(jersey, 0.0012384964), 3=(lab_coat, 0.0010649567), 4=(Windsor_tie, 0.00103015), 5=(sweatshirt, 0.0010226701)}
Predicted object for image2.jpg is water_ouzel
{1=(water_ouzel, 0.0013309028), 2=(partridge, 0.0012617374), 3=(quail, 0.0011414274), 4=(black_grouse, 0.0010846116), 5=(ptarmigan, 0.0010832328)}
Predicted object for image3.jpg is Egyptian_cat
{1=(Egyptian_cat, 0.002451402), 2=(tabby, 0.0010507071), 3=(tiger_cat, 0.0010484433), 4=(lynx, 9.993138E-4), 5=(swab, 9.986034E-4)}
Predicted object for image4.jpg is sports_car
{1=(sports_car, 0.0019709482), 2=(grille, 0.0011636796), 3=(car_wheel, 0.0010826474), 4=(convertible, 0.0010363625), 5=(beach_wagon, 0.001020963)}
Predicted object for image5.jpg is broccoli
{1=(broccoli, 0.0024892658), 2=(cauliflower, 0.0010865516), 3=(pot, 9.991141E-4), 4=(bonnet, 9.988336E-4), 5=(head_cabbage, 9.98508E-4)}
Predicted object for image6.jpg is tench
{1=(tench, 0.0013162919), 2=(rock_beauty, 0.0012848309), 3=(eel, 0.0010670079), 4=(axolotl, 0.0010587589), 5=(electric_ray, 0.0010518323)}
Predicted object for image7.jpg is hog
{1=(hog, 0.0022768131), 2=(wild_boar, 0.0011734703), 3=(sloth_bear, 0.0010042546), 4=(American_black_bear, 0.0010016224), 5=(brown_bear, 0.0010007834)}
Predicted object for image8.jpg is goldfish
{1=(goldfish, 0.002397591), 2=(rock_beauty, 0.001040403), 3=(barracouta, 0.0010255391), 4=(tench, 0.0010239), 5=(puffer, 0.0010075383)}

Process finished with exit code 0


         */
    }
}

