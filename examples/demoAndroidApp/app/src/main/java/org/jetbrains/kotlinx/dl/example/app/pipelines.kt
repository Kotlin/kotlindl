package org.jetbrains.kotlinx.dl.example.app

import android.content.Context
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.Rect
import androidx.camera.core.ImageProxy
import org.jetbrains.kotlinx.dl.api.inference.FlatShape
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.dataset.Imagenet
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.InputType
import org.jetbrains.kotlinx.dl.impl.preprocessing.*
import org.jetbrains.kotlinx.dl.impl.preprocessing.camerax.toBitmap
import org.jetbrains.kotlinx.dl.impl.util.argmax
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxHighLevelModel
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.classification.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.onnx.inference.classification.predictTopKObjects
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.NNAPI
import org.jetbrains.kotlinx.dl.onnx.inference.facealignment.FaceDetectionModel
import org.jetbrains.kotlinx.dl.onnx.inference.facealignment.Fan2D106FaceAlignmentModel
import org.jetbrains.kotlinx.dl.onnx.inference.inferUsing
import org.jetbrains.kotlinx.dl.onnx.inference.objectdetection.SSDLikeModel
import org.jetbrains.kotlinx.dl.onnx.inference.objectdetection.detectObjects
import org.jetbrains.kotlinx.dl.onnx.inference.posedetection.SinglePoseDetectionModel
import org.jetbrains.kotlinx.dl.onnx.inference.posedetection.detectPose


interface InferencePipeline {
    fun analyze(image: ImageProxy, confidenceThreshold: Float): Prediction?
    fun close()
}

enum class Tasks(val descriptionId: Int) {
    Classification(R.string.model_type_classification),
    ObjectDetection(R.string.model_type_object_detection),
    PoseDetection(R.string.model_type_pose_detection),
    FaceAlignment(R.string.model_type_face_alignment)
}

enum class Pipelines(val task: Tasks, val descriptionId: Int) {
    SSDMobilenetV1(Tasks.ObjectDetection, R.string.pipeline_ssd_mobilenet_v1) {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return DetectionPipeline(ONNXModels.ObjectDetection.SSDMobileNetV1.pretrainedModel(hub))
        }
    },
    EfficientNetLite4(Tasks.Classification, R.string.pipeline_efficient_net_lite_4) {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return ClassificationPipeline(ONNXModels.CV.EfficientNet4Lite().pretrainedModel(hub))
        }
    },
    MobilenetV1(Tasks.Classification, R.string.pipeline_mobilenet_v1) {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return ClassificationPipeline(ONNXModels.CV.MobilenetV1().pretrainedModel(hub))
        }
    },
    Shufflenet(Tasks.Classification, R.string.pipeline_shufflenet) {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return ShufflenetPipeline(
                OnnxInferenceModel {
                    resources.openRawResource(R.raw.shufflenet).use { it.readBytes() }
                }
            )
        }
    },
    EfficientDetLite0(Tasks.ObjectDetection, R.string.pipeline_efficient_det_lite_0) {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return DetectionPipeline(ONNXModels.ObjectDetection.EfficientDetLite0.pretrainedModel(hub))
        }
    },
    MoveNetSinglePoseLighting(Tasks.PoseDetection, R.string.pipeline_move_net_single_pose_lighting) {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            return PoseDetectionPipeline(ONNXModels.PoseDetection.MoveNetSinglePoseLighting.pretrainedModel(hub))
        }
    },
    FaceAlignment(Tasks.FaceAlignment, R.string.pipeline_face_alignment) {
        override fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline {
            val detectionModel = ONNXModels.FaceDetection.UltraFace320.pretrainedModel(hub)
            val alignmentModel = ONNXModels.FaceAlignment.Fan2d106.pretrainedModel(hub)
            return FaceAlignmentPipeline(detectionModel, alignmentModel)
        }
    };

    abstract fun createPipeline(hub: ONNXModelHub, resources: Resources): InferencePipeline
}

internal class DetectionPipeline(private val model: SSDLikeModel) : InferencePipeline {
    override fun analyze(image: ImageProxy, confidenceThreshold: Float): Prediction? {
        val detections = model.inferUsing(CPU()) {
            it.detectObjects(image, -1)
        }.filter { it.probability >= confidenceThreshold }
        if (detections.isEmpty()) return null

        return PredictedObject(detections)
    }

    override fun close() {
        model.close()
    }

    class PredictedObject(private val detections: List<DetectedObject>) : Prediction {
        override val shapes: List<FlatShape<*>> get() = detections
        override val confidence: Float get() = detections.first().probability
        override fun getText(context: Context): String {
            val singleObject = detections.singleOrNull()
            if (singleObject != null) return singleObject.label ?: ""
            return context.getString(R.string.label_objects, detections.size)
        }
    }
}

internal class ClassificationPipeline(private val model: ImageRecognitionModel) :
    InferencePipeline {

    override fun analyze(image: ImageProxy, confidenceThreshold: Float): Prediction? {
        val predictions = model.inferUsing(NNAPI()) {
            it.predictTopKObjects(image, 1)
        }
        if (predictions.isEmpty()) return null
        val (label, confidence) = predictions.single()
        return PredictedClass(label, confidence)
    }

    override fun close() {
        model.close()
    }

    class PredictedClass(private val label: String, override val confidence: Float) : Prediction {
        override val shapes: List<FlatShape<*>> get() = emptyList()
        override fun getText(context: Context): String = label
    }
}

internal class ShufflenetPipeline(
    private val model: OnnxInferenceModel
) : InferencePipeline {
    private val labels = Imagenet.V1k.labels()

    override fun analyze(image: ImageProxy, confidenceThreshold: Float): ClassificationPipeline.PredictedClass {
        val bitmap = image.toBitmap()
        val rotation = image.imageInfo.rotationDegrees.toFloat()

        val preprocessing = pipeline<Bitmap>()
            .resize {
                outputHeight = 224
                outputWidth = 224
            }
            .rotate { degrees = rotation }
            .toFloatArray { layout = TensorLayout.NCHW }
            .call(InputType.TORCH.preprocessing(channelsLast = false))

        val (label, confidence) = model.inferUsing(CPU()) {
            val (tensor, shape) = preprocessing.apply(bitmap)
            val logits = model.predictSoftly(tensor)
            val (confidence, _) = Softmax().apply(logits to shape)
            val labelId = confidence.argmax()
            labels[labelId]!! to confidence[labelId]
        }

        return ClassificationPipeline.PredictedClass(label, confidence)
    }

    override fun close() {
        model.close()
    }
}

class PoseDetectionPipeline(private val model: SinglePoseDetectionModel) : InferencePipeline {
    override fun analyze(image: ImageProxy, confidenceThreshold: Float): Prediction? {
        val detectedPose = model.inferUsing(CPU()) {
            it.detectPose(image)
        }

        if (detectedPose.landmarks.isEmpty()) return null

        return PredictedPose(detectedPose)
    }

    override fun close() = model.close()

    class PredictedPose(private val pose: DetectedPose) : Prediction {
        override val shapes: List<FlatShape<*>> get() = listOf(pose)
        override val confidence: Float get() = pose.landmarks.maxOf { it.probability }
        override fun getText(context: Context): String = context.getString(R.string.label_pose)
    }
}

class FaceAlignmentPipeline(
    private val detectionModel: FaceDetectionModel,
    private val alignmentModel: Fan2D106FaceAlignmentModel
) : InferencePipeline {
    override fun analyze(image: ImageProxy, confidenceThreshold: Float): Prediction? {
        val bitmap = image.toBitmap(applyRotation = true)

        val detectedObjects = detectionModel.detectFaces(bitmap, 1)
        if (detectedObjects.isEmpty()) {
            return null
        }

        val face = detectedObjects.first()
        if (face.probability < confidenceThreshold) return null

        val faceRect = Rect(
            (face.xMin * 0.9f * bitmap.width).toInt().coerceAtLeast(0),
            (face.yMin * 0.9f * bitmap.height).toInt().coerceAtLeast(0),
            (face.xMax * 1.1f * bitmap.width).toInt().coerceAtMost(bitmap.width),
            (face.yMax * 1.1f * bitmap.height).toInt().coerceAtMost(bitmap.height)
        )

        val landmarks = alignmentModel.predictOnCrop(bitmap, faceRect)
        return FaceAlignmentPrediction(face, landmarks)
    }

    override fun close() {
        detectionModel.close()
        alignmentModel.close()
    }

    data class FaceAlignmentPrediction(val face: DetectedObject, val landmarks: List<Landmark>): Prediction {
        override val shapes: List<FlatShape<*>> get() = landmarks + face
        override val confidence: Float get() = face.probability
        override fun getText(context: Context): String = context.getString(R.string.label_face)
    }
}

private fun <I> Operation<I, Bitmap>.cropRect(rect: Rect): Operation<I, Bitmap> {
    return crop {
        x = rect.left
        y = rect.top
        width = rect.width()
        height = rect.height()
    }
}

private fun <T : FlatShape<T>> OnnxHighLevelModel<Bitmap, List<T>>.predictOnCrop(
    bitmap: Bitmap,
    crop: Rect
): List<T> {
    val cropBitmap = pipeline<Bitmap>().cropRect(crop).apply(bitmap)
    return predict(cropBitmap).map { shape ->
        shape.map { x, y ->
            (crop.left + x * crop.width()) / bitmap.width to
                    (crop.top + y * crop.height()) / bitmap.height
        }
    }
}
