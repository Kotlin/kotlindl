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
        override val channelsFirst: Boolean = false
    ) :
        ModelType<T, ImageRecognitionModel> {
        override fun pretrainedModel(modelHub: ModelHub): ImageRecognitionModel {
            return buildImageRecognitionModel(modelHub, this)
        }

        /** */
        public object VGG16 : CV<Sequential>("models/tensorflow/cv/vgg16") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public object VGG19 : CV<Sequential>("models/tensorflow/cv/vgg19") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public object ResNet18 : CV<Functional>("models/tensorflow/cv/resnet18") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public object ResNet34 : CV<Functional>("models/tensorflow/cv/resnet34") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public object ResNet50 : CV<Functional>("models/tensorflow/cv/resnet50") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public object ResNet101 : CV<Functional>("models/tensorflow/cv/resnet101") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public object ResNet152 : CV<Functional>("models/tensorflow/cv/resnet152") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
            }
        }

        /** */
        public object ResNet50v2 : CV<Functional>("models/tensorflow/cv/resnet50v2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public object ResNet101v2 : CV<Functional>("models/tensorflow/cv/resnet101v2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public object ResNet152v2 : CV<Functional>("models/tensorflow/cv/resnet152v2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public object MobileNet : CV<Functional>("models/tensorflow/cv/mobilenet") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public object MobileNetV2 : CV<Functional>("models/tensorflow/cv/mobilenetv2") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public object Inception : CV<Functional>("models/tensorflow/cv/inception") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public object Xception : CV<Functional>("models/tensorflow/cv/xception") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public object DenseNet121 : CV<Functional>("models/tensorflow/cv/densenet121") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        }

        /** */
        public object DenseNet169 : CV<Functional>("models/tensorflow/cv/densenet169") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        }

        /** */
        public object DenseNet201 : CV<Functional>("models/tensorflow/cv/densenet201") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TORCH)
            }
        }

        /** */
        public object NASNetMobile : CV<Functional>("models/tensorflow/cv/nasnetmobile") {
            override fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
                return preprocessInput(data, tensorShape, inputType = InputType.TF)
            }
        }

        /** */
        public object NASNetLarge : CV<Functional>("models/tensorflow/cv/nasnetlarge") {
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
