package org.jetbrains.kotlinx.dl.api.inference.onnx;

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.InputType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.classification.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.dataset.Imagenet
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessing.*
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape

/**
 * Set of pretrained mobile-friendly ONNX models
 */
public object OnnxModels {
    /** Image classification models */
    public sealed class CV<T : InferenceModel>(
        override val modelRelativePath: String,
        override val channelsFirst: Boolean,
        override val inputColorMode: ColorMode = ColorMode.RGB,
    ) : OnnxModelType<T, ImageRecognitionModel> {
        override fun pretrainedModel(modelHub: ModelHub): ImageRecognitionModel {
            return ImageRecognitionModel(modelHub.loadModel(this) as OnnxInferenceModel, this)
        }

        /**
         * Image classification model based on EfficientNet-Lite architecture.
         * Trained on ImageNet 1k dataset.
         * (labels are available via [org.jetbrains.kotlinx.dl.dataset.Imagenet.labels] method).
         *
         * EfficientNet-Lite 4 is the largest variant and most accurate of the set of EfficientNet-Lite model.
         * It is an integer-only quantized model that produces the highest accuracy of all the EfficientNet models.
         * It achieves 80.4% ImageNet top-1 accuracy, while still running in real-time (e.g. 30ms/image) on a Pixel 4 CPU.
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1000)
         *
         * @see <a href="https://arxiv.org/abs/1905.11946">
         *     EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a>
         * @see <a href="https://github.com/onnx/models/tree/main/vision/classification/efficientnet-lite4">
         *    Official EfficientNet4Lite model from ONNX Github.</a>
         */
        public class EfficientNet4Lite : CV<OnnxInferenceModel>("efficientnet_lite4", channelsFirst = false) {
            override val preprocessor: Operation<Pair<FloatArray, TensorShape>, Pair<FloatArray, TensorShape>>
                get() = InputType.TF.preprocessing(channelsLast = !channelsFirst)
        }

        /**
         * Image classification model based on MobileNetV1 architecture.
         * Trained on ImageNet 1k dataset.
         * (labels are available via [org.jetbrains.kotlinx.dl.dataset.Imagenet.labels] method).
         *
         * MobileNetV1 is small, low-latency, low-power model and can be run efficiently on mobile devices
         *
         * The model have
         * - an input with the shape (1x224x224x3)
         * - an output with the shape (1x1001)
         *
         * @see <a href="https://arxiv.org/abs/1905.11946">
         *     EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a>
         * @see <a href="https://github.com/onnx/models/tree/main/vision/classification/efficientnet-lite4">
         *    Official EfficientNet4Lite model from ONNX Github.</a>
         */
        public class MobilenetV1 : CV<OnnxInferenceModel>("mobilenet_v1", channelsFirst = false) {
            override val preprocessor: Operation<Pair<FloatArray, TensorShape>, Pair<FloatArray, TensorShape>>
                get() = pipeline<Pair<FloatArray, TensorShape>>()
                    .rescale { scalingCoefficient = 255f }
                    .normalize {
                        mean = floatArrayOf(0.5f, 0.5f, 0.5f)
                        std = floatArrayOf(0.5f, 0.5f, 0.5f)
                        channelsLast = !channelsFirst
                    }

            override fun pretrainedModel(modelHub: ModelHub): ImageRecognitionModel {
                return ImageRecognitionModel(
                    modelHub.loadModel(this),
                    this,
                    Imagenet.V1001.labels()
                )
            }
        }
    }
}
