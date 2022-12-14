/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.loaders

import org.jetbrains.kotlinx.dl.api.core.*
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.ImageRecognitionModel.Companion.createPreprocessing
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.InputType
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import java.awt.image.BufferedImage
import java.io.File

/**
 * Supported models for inference and transfer learning, trained on ImageNet dataset.
 *
 * All weights are imported from the `Keras.applications` or `ONNX.models` project and preprocessed with the KotlinDL project.
 *
 * @see TFModelType
 * @see TFModelHub
 */
public object TFModels {
    /** Image recognition models and preprocessing.
     *
     * @property [channelsFirst] If true it means that the second dimension is related to number of channels in image
     *                           has short notation as `NCWH`,
     *                           otherwise, channels are at the last position and has a short notation as `NHWC`.
     * @property [inputColorMode] An expected channels order for the input image.
     *                            Note: the wrong choice of this parameter can significantly impact the model's performance.
     * */
    public sealed class CV<T : GraphTrainableModel>(
        relativePath: String,
        private val channelsFirst: Boolean = false,
        private val inputColorMode: ColorMode = ColorMode.RGB,
        public var inputShape: IntArray? = null,
        noTop: Boolean = false
    ) : TFModelType<T, ImageRecognitionModel> {

        override val modelRelativePath: String = if (noTop) "$relativePath/notop" else relativePath

        init {
            if (inputShape != null) {
                require(inputShape!!.size == 3) { "Input shape for the model ${this.javaClass.kotlin.simpleName} should contain 3 number: height, weight and number of channels." }
                require(inputShape!![0] >= 32 && inputShape!![1] >= 32) { "Width and height should be no smaller than 32 for the model ${this.javaClass.kotlin.simpleName}." }
            }
        }

        override fun pretrainedModel(modelHub: ModelHub): ImageRecognitionModel {
            val model = loadModel(modelHub, this)
            return ImageRecognitionModel(model, inputColorMode, channelsFirst, preprocessor, this::class.simpleName)
        }

        @Suppress("UNCHECKED_CAST")
        override fun loadModelConfiguration(jsonFile: File): T {
            return (Functional.loadModelConfiguration(jsonFile, inputShape) as T).apply { freeze() }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * Instantiates the VGG16 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1409.1556">
         *     Very Deep Convolutional Networks for Large-Scale Image Recognition</a>
         * @see <a href="https://keras.io/api/applications/vgg/#vgg16-function">
         *    Official VGG16 model from Keras.applications.</a>
         */
        public class VGG16(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Sequential>(
                "models/tensorflow/cv/vgg16",
                inputShape = inputShape,
                noTop = noTop,
                inputColorMode = ColorMode.BGR
            ) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.CAFFE.preprocessing()

            override fun loadModelConfiguration(jsonFile: File): Sequential {
                return Sequential.loadModelConfiguration(jsonFile, inputShape).apply { freeze() }
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * Instantiates the VGG19 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1409.1556">
         *     Very Deep Convolutional Networks for Large-Scale Image Recognition</a>
         * @see <a href="https://keras.io/api/applications/vgg/#vgg19-function">
         *    Official VGG19 model from Keras.applications.</a>
         */
        public class VGG19(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Sequential>(
                "models/tensorflow/cv/vgg19",
                inputShape = inputShape,
                noTop = noTop,
                inputColorMode = ColorMode.BGR
            ) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.CAFFE.preprocessing()

            override fun loadModelConfiguration(jsonFile: File): Sequential {
                return Sequential.loadModelConfiguration(jsonFile, inputShape).apply { freeze() }
            }
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
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
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.CAFFE.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
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
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.CAFFE.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * This model has 50 layers with ResNetv1 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet50-function">
         *    Official ResNet50 model from Keras.applications.</a>
         */
        public class ResNet50(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>(
                "models/tensorflow/cv/resnet50",
                inputShape = inputShape,
                noTop = noTop,
                inputColorMode = ColorMode.BGR
            ) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.CAFFE.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * This model has 101 layers with ResNetv1 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet101-function">
         *    Official ResNet101 model from Keras.applications.</a>
         */
        public class ResNet101(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>(
                "models/tensorflow/cv/resnet101",
                inputShape = inputShape,
                noTop = noTop,
                inputColorMode = ColorMode.BGR
            ) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.CAFFE.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * This model has 152 layers with ResNetv1 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet152-function">
         *    Official ResNet152 model from Keras.applications.</a>
         */
        public class ResNet152(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>(
                "models/tensorflow/cv/resnet152",
                inputShape = inputShape,
                noTop = noTop,
                inputColorMode = ColorMode.BGR
            ) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.CAFFE.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * This model has 50 layers with ResNetv2 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet50v2-function">
         *    Official ResNet50v2 model from Keras.applications.</a>
         */
        public class ResNet50v2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet50v2", inputShape = inputShape, noTop = noTop) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TF.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * This model has 101 layers with ResNetv2 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet101v2-function">
         *    Official ResNet101v2 model from Keras.applications.</a>
         */
        public class ResNet101v2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet101v2", inputShape = inputShape, noTop = noTop) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TF.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * This model has 152 layers with ResNetv2 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: ResNet v2 uses pre-activation function whereas ResNet v1 uses post-activation for the residual blocks.
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.03385">
         *     Deep Residual Learning for Image Recognition (CVPR 2015)</a>
         * @see <a href="https://keras.io/api/applications/resnet/#resnet152v2-function">
         *    Official ResNet152v2 model from Keras.applications.</a>
         */
        public class ResNet152v2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/resnet152v2", inputShape = inputShape, noTop = noTop) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TF.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * Instantiates the MobileNet architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1704.04861">
         *     MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications/a>
         * @see <a href="https://keras.io/api/applications/mobilenet/">
         *    Official MobileNet model from Keras.applications.</a>
         */
        public class MobileNet(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/mobilenet", inputShape = inputShape, noTop = noTop) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TF.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * Instantiates the MobileNetV2 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1801.04381">
         *     MobileNetV2: Inverted Residuals and Linear Bottlenecks/a>
         * @see <a href="https://keras.io/api/applications/mobilenet/#mobilenetv2-function">
         *    Official MobileNetV2 model from Keras.applications.</a>
         */
        public class MobileNetV2(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/mobilenetv2", inputShape = inputShape, noTop = noTop) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TF.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * Instantiates the InceptionV3 architecture.
         *
         * The model have
         * - an input with the shape (1x299x299x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1512.00567">
         *     Rethinking the Inception Architecture for Computer Vision/a>
         * @see <a href="https://keras.io/api/applications/inceptionv3/">
         *    Official InceptionV3 model from Keras.applications.</a>
         */
        public class Inception(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/inception", inputShape = inputShape, noTop = noTop) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TF.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * Instantiates the Xception architecture.
         *
         * The model have
         * - an input with the shape (1x299x299x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1610.02357">
         *     Xception: Deep Learning with Depthwise Separable Convolutions/a>
         * @see <a href="https://keras.io/api/applications/xception/">
         *    Official Xception model from Keras.applications.</a>
         */
        public class Xception(noTop: Boolean = false, inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/xception", inputShape = inputShape, noTop = noTop) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TF.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * Instantiates the DenseNet121 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1608.06993">
         *     Densely Connected Convolutional Networks/a>
         * @see <a href="https://keras.io/api/applications/densenet/#densenet121-function">
         *    Official DenseNet121 model from Keras.applications.</a>
         */
        public class DenseNet121(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/densenet121", inputShape = inputShape, noTop = false) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TORCH.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * Instantiates the DenseNet169 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1608.06993">
         *     Densely Connected Convolutional Networks/a>
         * @see <a href="https://keras.io/api/applications/densenet/#densenet169-function">
         *    Official DenseNet169 model from Keras.applications.</a>
         */
        public class DenseNet169(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/densenet169", inputShape = inputShape, noTop = false) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TORCH.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * Instantiates the DenseNet201 architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1608.06993">
         *     Densely Connected Convolutional Networks/a>
         * @see <a href="https://keras.io/api/applications/densenet/#densenet201-function">
         *    Official DenseNet201 model from Keras.applications.</a>
         */
        public class DenseNet201(inputShape: IntArray? = null) :
            CV<Functional>("models/tensorflow/cv/densenet201", inputShape = inputShape, noTop = false) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TORCH.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * Instantiates the NASNetMobile architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
         *
         * @see <a href="https://arxiv.org/abs/1707.07012">
         *     Learning Transferable Architectures for Scalable Image Recognition/a>
         * @see <a href="https://keras.io/api/applications/nasnet/#nasnetmobile-function">
         *    Official NASNetMobile model from Keras.applications.</a>
         */
        public class NASNetMobile(noTop: Boolean = false) :
            CV<Functional>("models/tensorflow/cv/nasnetmobile", inputShape = intArrayOf(224, 224, 3), noTop = noTop) {
            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TF.preprocessing()
        }

        /**
         * This model is a neural network for image classification that take images as input and classify the major object in the image into a set of 1000 different classes
         * (labels are available via [org.jetbrains.kotlinx.dl.impl.dataset.Imagenet.V1k.labels] method).
         *
         * Instantiates the NASNetLarge architecture.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * NOTE: This model is converted from Keras.applications, the last few layers in the noTop model have been removed so that the user can fine-tune the model for their specific task.
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

            override val preprocessor: Operation<FloatData, FloatData>
                get() = InputType.TF.preprocessing()
        }

        public companion object {
            /**
             * Creates a preprocessing [Operation] which converts given [BufferedImage] to [FloatData] suitable for this [model].
             */
            public fun CV<*>.createPreprocessing(model: InferenceModel): Operation<BufferedImage, FloatData> {
                return createPreprocessing(model, channelsFirst, inputColorMode, preprocessor)
            }
        }
    }

    private fun loadModel(modelHub: ModelHub, modelType: CV<out GraphTrainableModel>): GraphTrainableModel {
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
        return model
    }
}