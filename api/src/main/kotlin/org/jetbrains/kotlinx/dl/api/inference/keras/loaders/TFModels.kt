/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.loaders

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing

/**
 * Supported models for inference and transfer learning, trained on ImageNet dataset.
 *
 * All weights are imported from the `Keras.applications` or `ONNX.models` project and preprocessed with the KotlinDL project.
 */
public object TFModels {
    /** Image recognition models and preprocessing. */
    public sealed class CV<T : GraphTrainableModel>(
        override val modelRelativePath: String,
        override val channelsFirst: Boolean = false,
        public var inputShape: IntArray? = null,
        internal var noTop: Boolean = false
    ) :
        ModelType<T, ImageRecognitionModel> {

        init {
            if (inputShape != null) {
                require(inputShape!!.size == 3) { "Input shape for the model ${this.javaClass.kotlin.simpleName} should contain 3 number: height, weight and number of channels." }
                require(inputShape!![0] >= 32 && inputShape!![1] >= 32) { "Width and height should be no smaller than 32 for the model ${this.javaClass.kotlin.simpleName}." }
            }
        }

        override fun pretrainedModel(modelHub: ModelHub): ImageRecognitionModel {
            return buildImageRecognitionModel(modelHub, this)
        }

        /** */
        public class VGG16(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Sequential>("models/tensorflow/cv/vgg16", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public class VGG19(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Sequential>("models/tensorflow/cv/vgg19", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public class ResNet18(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet18", inputShape = inputShape) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public class ResNet34(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet34", inputShape = inputShape) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public class ResNet50(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet50", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public class ResNet101(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet101", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public class ResNet152(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet152", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public class ResNet50v2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet50v2", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public class ResNet101v2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet101v2", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public class ResNet152v2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet152v2", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public class MobileNet(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/mobilenet", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public class MobileNetV2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/mobilenetv2", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public class Inception(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/inception", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public class Xception(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/xception", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public class DenseNet121(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/densenet121", inputShape = inputShape, noTop = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        }

        /** */
        public class DenseNet169(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/densenet169", inputShape = inputShape, noTop = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        }

        /** */
        public class DenseNet201(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/densenet201", inputShape = inputShape, noTop = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        }

        /** */
        public class NASNetMobile(noTop: Boolean = false) :
            CV<Functional>("models/tensorflow/cv/nasnetmobile", inputShape = intArrayOf(224, 224, 3), noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public class NASNetLarge(noTop: Boolean = false, inputShape: IntArray? = intArrayOf(331, 331, 3)) :
            CV<Functional>("models/tensorflow/cv/nasnetlarge", inputShape = inputShape, noTop = noTop) {
            init {
                require(inputShape!![0] >= 331 && inputShape[1] >= 331) { "Width and height should be no smaller than 331 for the model ${this.javaClass.kotlin.simpleName}." }
            }

            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }
    }

    private fun buildImageRecognitionModel(
        modelHub: ModelHub,
        modelType: ModelType<out GraphTrainableModel, ImageRecognitionModel>
    ): ImageRecognitionModel {
        modelHub as TFModelHub
        val model = modelHub.loadModel(modelType)

        model.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        val hdfFile = modelHub.loadWeights(modelType)

        model.loadWeights(hdfFile)

        return ImageRecognitionModel(model, modelType)
    }
}

/** Basic interface for models loaded from S3. */
public interface ModelType<T : InferenceModel, U : InferenceModel> {
    /** Relative path to model for local and S3 buckets storages. */
    public val modelRelativePath: String

    public val channelsFirst: Boolean

    /**
     * Common preprocessing function for the Neural Networks trained on ImageNet and whose weights are available with the keras.application.
     *
     * It takes [data] as input with shape [tensorShape] and applied the specific preprocessing according chosen modelType.
     *
     * @param [tensorShape] Should be 3 dimensional array (HWC or CHW format)
     */
    public fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray

    /**
     * Common preprocessing function for the Neural Networks trained on ImageNet and whose weights are available with the keras.application.
     *
     * It takes preprocessing pipeline, invoke it and applied the specific preprocessing to the given data.
     */
    public fun preprocessInput(preprocessing: Preprocessing): FloatArray {
        val (data, shape) = preprocessing()
        return preprocessInput(
            data,
            longArrayOf(shape.width!!, shape.height!!, shape.channels!!)
        ) // TODO: need to be 4 or 3 in all cases
    }

    /** Returns the specially prepared pre-trained model of the type U. */
    public fun pretrainedModel(modelHub: ModelHub): U

    /** Loads the model, identified by this name, from the [modelHub]. */
    public fun model(modelHub: ModelHub): T {
        return modelHub.loadModel(this)
    }
}
