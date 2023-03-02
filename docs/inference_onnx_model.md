In this tutorial, we will learn how to use KotlinDL to infer the ONNX models on Desktop and Android platforms.

## Desktop JVM
### Inference of the model included in KotlinDL Model Zoo
KotlinDL ONNX provides a set of pre-trained models through `OnnxModels` API.
You can find the list of models [here](./onnx_model_zoo.md).
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
Another difference is that model files need to be downloaded separately.

### Inference of the model included into KotlinDL Model Zoo
In this section, the single pose detection model will be used. Note that input data type is `Bitmap` instead of `BufferedImage`.
```kotlin
val modelHub = ONNXModelHub(context) // Android context is required to access the application resources
val model = ONNXModels.PoseDetection.MoveNetSinglePoseLighting.pretrainedModel(modelHub)

val bitmap = BitmapFactory.decodeStream(imageResource)

val detectedPose = model.inferAndCloseUsing(CPU()) {
    model.detectPose(image = bitmap, confidence = 0.05f)
}
```

### Downloading pre-trained KotlinDL Model Zoo models
KotlinDL expects the model files to be located in the application resources.
You can download the required models manually or use a Gradle plugins which downloads them automatically before the build.

To use the Gradle plugin, ensure that `google` and `gradlePluginPortal` repositories are listed in the `settings.gradle` file:
```groovy
pluginManagement {
    repositories {
        google()
        gradlePluginPortal()
    }
}
```

Then apply the plugin in the build script:
```groovy
plugins {
  id "org.jetbrains.kotlinx.kotlin-deeplearning-gradle-plugin" version "[KOTLIN-DL-VERSION]"
}
```

Configure plugin in the `downloadKotlinDLModels` section.
```groovy
downloadKotlinDLModels {
    models = ["MoveNetSinglePoseLighting"] // list of model type names to download
    sourceSet = "main"                     // optional name of the target source set ("main" by default)
    overwrite = false                      // optional parameter to overwrite existing files ("true" by default)
}
```

The plugin creates a task named `downloadKotlinDLModels` which is executed automatically before project is build 
or can be executed manually if needed.

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
Please, also check out the [Sample Android App](https://github.com/Kotlin/kotlindl-app-sample) for more details.

## ONNX Execution providers support
KotlinDL currently supports the following EPs:
* CPU (default)
* CUDA (for the devices with GPU and CUDA support)
* NNAPI (for Android devices with API 27+)

It is required to have the CUDA configured on your machine to use the CUDA EP.
Please, also check how to configure dependencies for the execution on a GPU in the [README.md](../README.md#running-kotlindl-on-gpu).

There are a few options for specifying the EP to use.
The models loaded using the ONNXModelHub API are instantiated with the default CPU EP.
```kotlin
val modelHub = ONNXModelHub(...)
val model = ONNXModels.PoseDetection.MoveNetMultiPoseLighting.pretrainedModel(modelHub) // default CPU EP is used
```
You can also specify the EP explicitly using the following syntax:
```kotlin
val model = modelHub.loadModel(ONNXModels.CV.EfficientNet4Lite, NNAPI())
```

Please note that when using a low-level API ONNXInferenceModel, you need to specify the EP explicitly.
You can do it using the functions `inferUsing` and `inferAndCloseUsing`.
Those functions explicitly declare the EPs to be used for inference in their scope.
Although these two functions have the same goal to initialize the model with the given
execution providers explicitly, they have a little different behavior.
`inferAndCloseUsing` has Kotlin's 'use' scope function semantics, i.e., it closes the model at the end of the block;
meanwhile, `inferUsing` is designed for repeated use and has Kotlin's 'run' scope function semantics.
```kotlin
val model = OnnxInferenceModel(...)

model.inferAndCloseUsing(CPU()) {
    val result = it.predictRaw(image) { output -> ... }
}
```
<em>Usage of inferUsingAndClose for one-time inference with CPU execution provider</em>
```kotlin
val model = ONNXModels.PoseDetection.MoveNetMultiPoseLighting.pretrainedModel(...)

model.inferUsing(CUDA()) { poseDetectionModel ->
    for (image in images) {
        val result = it.predictRaw(image) { output -> ... }
        ...
    }
}

model.close()
```
<em>Usage of inferUsing for recurring inference with CUDA execution provider</em>

Another option is to use the initializeWith function to configure EPs for the model instance.

```kotlin
val model = OnnxInferenceModel(...)
model.initializeWith(NNAPI())
```
<em>Loading and initialization of the model with NNAPI execution provider</em>


