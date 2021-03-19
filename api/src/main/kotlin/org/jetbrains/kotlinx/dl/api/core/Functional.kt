/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.ActivationLayer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.LossFunction
import org.jetbrains.kotlinx.dl.api.core.loss.SoftmaxCrossEntropyWithLogits
import org.jetbrains.kotlinx.dl.api.core.metric.Metric
import org.jetbrains.kotlinx.dl.api.core.optimizer.Optimizer
import org.jetbrains.kotlinx.dl.api.core.util.OUTPUT_NAME
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.jetbrains.kotlinx.dl.api.inference.keras.loadFunctionalModelLayers
import org.jetbrains.kotlinx.dl.api.inference.keras.saveModelConfiguration
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.core.Placeholder
import java.io.File
import java.io.FileNotFoundException

/**
 * Sequential model groups a linear stack of layers into a TensorFlow Model.
 * Also, it provides training and inference features on this model.
 *
 * @property [inputLayer] the input layer with initial shapes.
 * @property [layers] the layers to describe the model design.
 * @constructor Creates a Functional model via sequence of [layers].
 */
public class Functional(vararg layers: Layer) : GraphTrainableModel(*layers) {
    public companion object {
        /**
         * Creates the [Functional] model.
         *
         * @property [input] The input layer with initial shapes.
         * @property [layers] The layers to describe the model design.
         * @return the [Functional] model.
         */
        @JvmStatic
        public fun of(vararg layers: Layer): GraphTrainableModel {
            require(layers.isNotEmpty()) { "Model should contain layers!" }
            val input = layers[0]
            require(input is Input) { "Model should start from the Input layer" }

            // TODO: check that preprocessing is correct for input layer
            preProcessLayerNames(layers)
            val model = Functional(*layers)
            postProcessLayerNames(layers, model)
            return model
        }

        /**
         * Creates the [Functional] model.
         * @property [layers] The layers to describe the model design.
         * NOTE: First layer should be input layer.
         * @return the [Functional] model.
         */
        @JvmStatic
        public fun of(layers: List<Layer>): Functional {
            require(layers.isNotEmpty()) { "Model should contain layers!" }
            val input = layers[0]
            require(input is Input) { "Model should start from the Input layer" }

            val otherLayers = layers.subList(1, layers.size)
            preProcessLayerNames(otherLayers.toTypedArray())
            val model = Functional(*layers.toTypedArray())
            postProcessLayerNames(otherLayers.toTypedArray(), model)
            return model
        }

        /**
         * Loads a [Functional] model from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [Functional] model.
         * @return Non-compiled and non-trained Sequential model.
         */
        @JvmStatic
        public fun loadModelConfiguration(configuration: File): Functional {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            return org.jetbrains.kotlinx.dl.api.inference.keras.loadFunctionalModelConfiguration(configuration)
        }

        /**
         * Loads a [Functional] model layers from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [Functional] model.
         * @return Pair of <input layer; list of layers>.
         */
        @JvmStatic
        public fun loadModelLayersFromConfiguration(configuration: File): MutableList<Layer> {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            return loadModelLayersFromConfiguration(configuration)
        }

        /**
         * Loads a [Functional] model from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return Non-compiled and non-trained Sequential model.
         */
        @JvmStatic
        public fun loadDefaultModelConfiguration(modelDirectory: File): Functional {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a 'modelConfig.json' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/modelConfig.json")

            if (!configuration.exists()) throw FileNotFoundException(
                "File 'modelConfig.json' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            return org.jetbrains.kotlinx.dl.api.inference.keras.loadFunctionalModelConfiguration(configuration)
        }

        /**
         * Loads a [Functional] model layers from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return Pair of <input layer; list of layers>.
         */
        @JvmStatic
        public fun loadModelLayersFromDefaultConfiguration(modelDirectory: File): MutableList<Layer> {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a 'modelConfig.json' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/modelConfig.json")

            if (!configuration.exists()) throw FileNotFoundException(
                "File 'modelConfig.json' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            return loadFunctionalModelLayers(configuration)
        }
    }

    override fun compile(optimizer: Optimizer, loss: LossFunction, metric: Metric, callback: Callback) {
        check(!isModelCompiled) { "The model is compiled already. Graph is created. Create new model and compile it." }

        validateModelArchitecture()

        this.loss = loss
        this.metric = metric
        this.metrics = listOf(metric) // handle multiple metrics
        this.optimizer = optimizer
        //this.callback = callback
        //this.callback.model = this // TODO: cyclic reference

        val inputLayer = layers[0] as Input

        inputLayer.build(tf)
        var inputShape: Shape? = inputLayer.computeOutputShape()

        layers.filter { it !is Input }.forEach {
            /*if (inputShape == null) {
                inputShape = it.inboundLayers[0].outputShape.toShape()
            }*/

            it.buildFromInboundLayers(tf, kGraph)
            val outputShape = it.computeOutputShapeFromInboundLayers()
            val dims = outputShape.dims()

            check(outputShape.tail().all { elem -> elem > 0 })
            {
                "The last dimensions (except first = -1) of shape of layer ${it.name} contains zero or negative dimension values: ${dims.contentToString()}.\n" +
                        "Analyze your model architecture and layer output shapes carefully to discover a problem."
            }

            it.outputShape = outputShape // it could be done inside computeOutputShapeMethods

            logger.info { "${it.name}; outputShape: $outputShape $it" }

            inputShape = null
        }

        if (layers.last() is Dense) amountOfClasses = (layers.last() as Dense).outputSize.toLong()
        else if (layers.last() is ActivationLayer) amountOfClasses =
            (layers.last() as ActivationLayer).outputShape.tail().last() // valid for mobileNet/DenseNet


        xOp = inputLayer.input
        yTrueOp = tf.placeholder(getDType()) as Operand<Float>
        numberOfLossesOp = tf.withName("numberOfLosses").placeholder(
            getDType(),
            Placeholder.shape(Shape.scalar())
        )

        training = tf.withName("training").placeholder(
            Boolean::class.javaObjectType,
            Placeholder.shape(Shape.scalar())
        )

        yPredOp = forward(xOp, inputLayer)
        lossOp = loss.apply(tf, yPredOp, yTrueOp, numberOfLossesOp)
        targets = optimizer.prepareTargets(kGraph, tf, lossOp)

        predictionOp = when (loss) {
            is SoftmaxCrossEntropyWithLogits -> tf.withName(OUTPUT_NAME).nn.softmax(yPredOp)
            else -> tf.withName(OUTPUT_NAME).identity(yPredOp)
        }

        metricOp = metric.apply(tf, predictionOp, yTrueOp, numberOfLossesOp)

        isModelCompiled = true
    }

    private fun forward(input: Operand<Float>, inputLayer: Input): Operand<Float> {
        var output: Operand<Float> = input
        val outputByLayerName = mutableMapOf<String, Operand<Float>>()
        val outputs = mutableListOf<Operand<Float>>()
        outputs.add(input)
        outputByLayerName[inputLayer.name] = input
        for (layer in layers) {
            for (inboundLayer in layer.inboundLayers) {
                outputs.add(outputByLayerName[inboundLayer.name]!!)
            }
            output = layer.forward(tf, outputs, training, numberOfLossesOp)
            outputByLayerName[layer.name] = output
            outputs.clear()
        }
        return output
    }

    override fun save(
        modelDirectory: File,
        savingFormat: SavingFormat,
        saveOptimizerState: Boolean,
        writingMode: WritingMode
    ) {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method." }
        //TODO: work wrong for cases in resnet50_prediction_save_load
        // check(isOptimizerVariableInitialized) { "The optimizer variables are not initialized yet. Initialize the optimizer variables with init() method or load optimizer weights to use this method." }

        val pathToModelDirectory = modelDirectory.absolutePath
        when (writingMode) {
            WritingMode.FAIL_IF_EXISTS -> {
                check(!modelDirectory.exists()) { "The directory exists on path $pathToModelDirectory, please be careful it could contain valuable model! Change this mode to OVERRIDE if you want to override this directory." }
                modelDirectory.mkdir()
            }
            WritingMode.OVERRIDE -> {
                if (modelDirectory.exists()) {
                    modelDirectory.deleteRecursively()
                }
                modelDirectory.mkdir()
            }
            WritingMode.APPEND -> {
                if (!modelDirectory.exists()) {
                    modelDirectory.mkdir()
                }
            }
        }

        when (savingFormat) {
            SavingFormat.TF_GRAPH_CUSTOM_VARIABLES -> saveInSimpleFormat(pathToModelDirectory, saveOptimizerState)
            SavingFormat.TF_GRAPH -> saveInSavedModelFormat(pathToModelDirectory)
            SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES -> saveInKerasFormat(pathToModelDirectory, saveOptimizerState)
        }
    }

    private fun saveInKerasFormat(pathToModelDirectory: String, saveOptimizerState: Boolean) {
        saveModel(pathToModelDirectory)
        saveVariables(pathToModelDirectory, saveOptimizerState)
    }

    private fun saveModel(pathToModelDirectory: String) {
        val jsonConfig = File("$pathToModelDirectory/modelConfig.json")
        this.saveModelConfiguration(
            jsonConfig,
            isKerasFullyCompatible = true
        ) // TODO: propogate or remove this parameter
    }

    private fun saveInSavedModelFormat(pathToModelDirectory: String) {
        saveGraphDef(pathToModelDirectory)
    }

    private fun saveInSimpleFormat(pathToModelDirectory: String, saveOptimizerState: Boolean) {
        saveGraphDef(pathToModelDirectory)
        saveVariables(pathToModelDirectory, saveOptimizerState)
    }

    private fun saveGraphDef(pathToModelDirectory: String) {
        val file = File("$pathToModelDirectory/graph.pb")
        file.writeBytes(kGraph.tfGraph.toGraphDef())
    }

    public override fun summary(stringLayerNameTypeSize: Int, stringOutputShapeSize: Int): List<String> {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }

        logger.info("=================================================================")
        logger.info("Model: Functional")
        logger.info("_________________________________________________________________")
        logger.info("Layer (type)                 Output Shape              Param #   ")
        logger.info("=================================================================")

        var totalTrainableParams = 0
        var totalFrozenParams = 0

        val layerDescriptions = mutableListOf<String>()

        for (l in layers) {
            if (l.isTrainable) totalTrainableParams += l.paramCount else totalFrozenParams += l.paramCount
            val layerDescription = createLayerDescription(l, stringLayerNameTypeSize, stringOutputShapeSize)
            layerDescriptions.add(layerDescription)
            logger.info(layerDescription)
            logger.info("_________________________________________________________________")
        }

        logger.info("=================================================================")
        logger.info("Total trainable params: $totalTrainableParams")
        logger.info("Total frozen params: $totalFrozenParams")
        logger.info("Total params: ${totalTrainableParams + totalFrozenParams}")
        logger.info("=================================================================")

        return layerDescriptions
    }

    private fun createLayerDescription(
        l: Layer,
        stringLayerNameTypeSize: Int,
        stringOutputShapeSize: Int
    ): String {
        val firstPart = "${l.name}(${l::class.simpleName})"

        val stringBuilder = StringBuilder(firstPart)
        for (i in 1 until stringLayerNameTypeSize - firstPart.length) {
            stringBuilder.append(" ")
        }

        val secondPart = l.outputShape.toString()

        stringBuilder.append(secondPart)

        for (i in 0 until stringOutputShapeSize - secondPart.length) {
            stringBuilder.append(" ")
        }

        stringBuilder.append(l.paramCount)

        return stringBuilder.toString()
    }
}
