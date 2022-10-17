In this tutorial, we will learn how to use KotlinDL to infer the ONNX models on Desktop and Android platforms.

## Desktop JVM
### Inference of the model included in KotlinDL Model Zoo
KotlinDL ONNX provides a set of pre-trained models through `OnnxModels` API.
You can find the list of models here. [here](./onnx_model_zoo.md).
In this section, we will use MoveNet model for human pose estimation. 

```kotlin
val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
val model = ONNXModels.PoseDetection.MoveNetMultiPoseLighting.pretrainedModel(modelHub)

val image = ImageConverter.toBufferedImage(File("path/to/image"))

val detectedPoses = model.inferAndCloseUsing(CPU()) {
    model.detectPoses(image = image, confidence = 0.05f)
}
```

### Inference of custom ONNX model
There are a lot of models that are not included in the ONNX Models Zoo API.
In this case, you can use a low-level API to load custom models from the file.
In this section, we will load and infer the [SSDMobilenetV1](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx) model using `OnnxInferenceModel` API.

```kotlin
val model = OnnxInferenceModel("path/to/model.onnx")

val preprocessing = pipeline<BufferedImage>()
    .resize {
        outputHeight = 300
        outputWidth = 300
    }
    .convert { colorMode = ColorMode.RGB }
    .toFloatArray { }

val image = ImageConverter.toBufferedImage(File("path/to/image"))

val detections = model.inferAndCloseUsing(CPU()) {
    val (inputData, shape) = preprocessing.apply(image)

    it.predictRaw(inputData) { output -> {
        val boxes = output.get2DFloatArray("outputBoxesName")
        val classIndices = output.getFloatArray("outputClassesName")
        val probabilities = output.getFloatArray("outputScoresName")
        val numberOfFoundObjects = boxes.size

        val foundObjects = mutableListOf<DetectedObject>()
        for (i in 0 until numberOfFoundObjects) {
            val detectedObject = DetectedObject(
                // left, top, right, bottom
                xMin = boxes[i][0],
                yMin = boxes[i][1],
                xMax = boxes[i][2],
                yMax = boxes[i][3],
                probability = probabilities[i],
                label = Coco.V2017.labels()[classIndices[i].toInt()]
            )
            foundObjects.add(detectedObject)
        }
        foundObjects
    }}
}
```

## Android
The inference on Android is almost identical to the Desktop JVM counterpart.
Slight differences appear due to the different image representations supported on a specific platform.
On Android, the primary input data type is `Bitmap` the common image representation on the Android platform.

### Inference of the model included into KotlinDL Model Zoo
In this section, the same pose detection model will be used.
One more thing to mention is that KotlinDL expects the model files to be located in the application resources. You can download the required models using Gradle plugin or manually. *TODO update*


Also note that input data type is `Bitmap` instead of `BufferedImage`.
```kotlin
val modelHub = ONNXModelHub(context) // Android context is required to access the application resources
val model = ONNXModels.PoseDetection.MoveNetMultiPoseLighting.pretrainedModel(modelHub)

val bitmap = BitmapFactory.decodeStream(imageResource)

val detectedPoses = model.inferAndCloseUsing(CPU()) {
    model.detectPoses(image = bitmap, confidence = 0.05f)
}
```
### Inference of custom ONNX model
In this section we will use the same model as in the corresponding Desktop JVM section.
Note that model instance is created from the byte representation of the model file loaded from the application resources.
You can potentially load the model from external storage or the internet.
```kotlin
val modelBytes = resources.openRawResource(modelResource).readBytes()

val model = OnnxInferenceModel(modelBytes)

val preprocessing = pipeline<Bitmap>()
    .resize {
        outputHeight = 300
        outputWidth = 300
    }
    .toFloatArray { layout = TensorLayout.NHWC }

val bitmap = BitmapFactory.decodeStream(imageResource)

val detections = model.inferAndCloseUsing(CPU()) {
    val (inputData, shape) = preprocessing.apply(image)

    it.predictRaw(inputData) { output -> {
        val boxes = output.get2DFloatArray("outputBoxesName")
        val classIndices = output.getFloatArray("outputClassesName")
        val probabilities = output.getFloatArray("outputScoresName")
        val numberOfFoundObjects = boxes.size

        val foundObjects = mutableListOf<DetectedObject>()
        for (i in 0 until numberOfFoundObjects) {
            val detectedObject = DetectedObject(
                // left, top, right, bottom
                xMin = boxes[i][0],
                yMin = boxes[i][1],
                xMax = boxes[i][2],
                yMax = boxes[i][3],
                probability = probabilities[i],
                label = Coco.V2017.labels()[classIndices[i].toInt()]
            )
            foundObjects.add(detectedObject)
        }
        foundObjects
    }}
}
```

For more information about the KotlinDL ONNX API, please refer to the [Documentation](https://kotlin.github.io/kotlindl/) and [examples](https://github.com/Kotlin/kotlindl/tree/master/examples/src/main/kotlin/examples/onnx).
Please, also check out the [Sample Android App](https://github.com/ermolenkodev/ort_mobile_demo) for more details.
