/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
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
import java.io.File

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

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * Instantiates the VGG16 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1409.1556">
         *     Very Deep Convolutional Networks for Large-Scale Image Recognition</a>
         * @see <a href="https://keras.io/api/applications/vgg/#vgg16-function">
         *    Official VGG16 model from Keras.applications.</a>
         */
        public class VGG16(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Sequential>("models/tensorflow/cv/vgg16", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * Instantiates the VGG19 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1409.1556">
         *     Very Deep Convolutional Networks for Large-Scale Image Recognition</a>
         * @see <a href="https://keras.io/api/applications/vgg/#vgg19-function">
         *    Official VGG19 model from Keras.applications.</a>
         */
        public class VGG19(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Sequential>("models/tensorflow/cv/vgg19", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * This model has 18 layers with ResNetv1 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         */
        public class ResNet18(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet18", inputShape = inputShape) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * This model has 34 layers with ResNetv1 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         */
        public class ResNet34(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet34", inputShape = inputShape) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * This model has 50 layers with ResNetv1 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet50-function">
         *    Official ResNet50 model from Keras.applications.</a>
         */
        public class ResNet50(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet50", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * This model has 101 layers with ResNetv1 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet101-function">
         *    Official ResNet101 model from Keras.applications.</a>
         */
        public class ResNet101(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet101", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * This model has 152 layers with ResNetv1 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet152-function">
         *    Official ResNet152 model from Keras.applications.</a>
         */
        public class ResNet152(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet152", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * This model has 50 layers with ResNetv2 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet50v2-function">
         *    Official ResNet50v2 model from Keras.applications.</a>
         */
        public class ResNet50v2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet50v2", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * This model has 101 layers with ResNetv2 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet101v2-function">
         *    Official ResNet101v2 model from Keras.applications.</a>
         */
        public class ResNet101v2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet101v2", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * This model has 152 layers with ResNetv2 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet152v2-function">
         *    Official ResNet152v2 model from Keras.applications.</a>
         */
        public class ResNet152v2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet152v2", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * Instantiates the MobileNet architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1704.04861">
         *     MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications/a>
         * @see <a href="https://keras.io/api/applications/mobilenet/">
         *    Official MobileNet model from Keras.applications.</a>
         */
        public class MobileNet(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/mobilenet", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * Instantiates the MobileNetV2 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1801.04381">
         *     MobileNetV2: Inverted Residuals and Linear Bottlenecks/a>
         * @see <a href="https://keras.io/api/applications/mobilenet/#mobilenetv2-function">
         *    Official MobileNetV2 model from Keras.applications.</a>
         */
        public class MobileNetV2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/mobilenetv2", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * Instantiates the InceptionV3 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.00567">
         *     Rethinking the Inception Architecture for Computer Vision/a>
         * @see <a href="https://keras.io/api/applications/inceptionv3/">
         *    Official InceptionV3 model from Keras.applications.</a>
         */
        public class Inception(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/inception", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * Instantiates the Xception architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1610.02357">
         *     Xception: Deep Learning with Depthwise Separable Convolutions/a>
         * @see <a href="https://keras.io/api/applications/xception/">
         *    Official Xception model from Keras.applications.</a>
         */
        public class Xception(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/xception", inputShape = inputShape, noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * Instantiates the DenseNet121 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1608.06993">
         *     Densely Connected Convolutional Networks/a>
         * @see <a href="https://keras.io/api/applications/densenet/#densenet121-function">
         *    Official DenseNet121 model from Keras.applications.</a>
         */
        public class DenseNet121(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/densenet121", inputShape = inputShape, noTop = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * Instantiates the DenseNet169 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1608.06993">
         *     Densely Connected Convolutional Networks/a>
         * @see <a href="https://keras.io/api/applications/densenet/#densenet169-function">
         *    Official DenseNet169 model from Keras.applications.</a>
         */
        public class DenseNet169(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/densenet169", inputShape = inputShape, noTop = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * Instantiates the DenseNet201 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1608.06993">
         *     Densely Connected Convolutional Networks/a>
         * @see <a href="https://keras.io/api/applications/densenet/#densenet201-function">
         *    Official DenseNet201 model from Keras.applications.</a>
         */
        public class DenseNet201(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/densenet201", inputShape = inputShape, noTop = false) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * Instantiates the NASNetMobile architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1707.07012">
         *     Learning Transferable Architectures for Scalable Image Recognition/a>
         * @see <a href="https://keras.io/api/applications/nasnet/#nasnetmobile-function">
         *    Official NASNetMobile model from Keras.applications.</a>
         */
        public class NASNetMobile(noTop: Boolean = false) :
            CV<Functional>("models/tensorflow/cv/nasnetmobile", inputShape = intArrayOf(224, 224, 3), noTop = noTop) {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels] method).
         *
         * Instantiates the NASNetLarge architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for his specific task.
         *
         * @see <a href="https://arxiv.org/abs/1707.07012">
         *     Learning Transferable Architectures for Scalable Image Recognition/a>
         * @see <a href="https://keras.io/api/applications/nasnet/#nasnetlarge-function">
         *    Official NASNetLarge model from Keras.applications.</a>
         */
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
        // TODO: this part is not needed for inference (if we could add manually Softmax at the end of the graph)
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

/**
 * Basic interface for models loaded from S3.
 * @param T the type of the basic model for common functionality.
 * @param U the type of the pre-trained model for usage in Easy API.
 */
public interface ModelType<T : InferenceModel, U : InferenceModel> {
    /** Relative path to model for local and S3 buckets storages. */
    public val modelRelativePath: String

    /**
     * If true it means that the second dimension is related to number of channels in image has short notation as `NCWH`,
     * otherwise, channels are at the last position and has a short notation as `NHWC`.
     */
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
    public fun preprocessInput(imageFile: File, preprocessing: Preprocessing): FloatArray {
        val (data, shape) = preprocessing(imageFile)
        return preprocessInput(
            data,
            longArrayOf(shape.width!!, shape.height!!, shape.channels!!)
        )
    }

    /** Returns the specially prepared pre-trained model of the type U. */
    public fun pretrainedModel(modelHub: ModelHub): U

    /** Loads the model, identified by this name, from the [modelHub]. */
    public fun model(modelHub: ModelHub): T {
        return modelHub.loadModel(this)
    }

    public fun preInit(): InferenceModel {
        TODO()
    }
}
