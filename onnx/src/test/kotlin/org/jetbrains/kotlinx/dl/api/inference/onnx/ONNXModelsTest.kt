package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.io.TempDir
import java.io.File

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ONNXModelsTest {

    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))

    @BeforeEach
    fun setUp() {

    }

    @Test
    fun createResNet18ONNXModel(@TempDir tempDir: File) {
        val resnet18v1File = tempDir.resolve(ONNXModels.CV.ResNet18.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet18)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet18v1File.isFile)
    }

    @Test
    fun loadResNet18ONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet18.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createResNet34ONNXModel(@TempDir tempDir: File) {
        val resnet34v1File = tempDir.resolve(ONNXModels.CV.ResNet34.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet34)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet34v1File.isFile)
    }

    @Test
    fun loadResNet34ONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet34.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createResNet50ONNXModel(@TempDir tempDir: File) {
        val resnet50v1File = tempDir.resolve(ONNXModels.CV.ResNet50.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet50)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet50v1File.isFile)
    }

    @Test
    fun loadResNet50ONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet50.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createResNet101ONNXModel(@TempDir tempDir: File) {
        val resnet101v1File = tempDir.resolve(ONNXModels.CV.ResNet101.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet101)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet101v1File.isFile)
    }

    @Test
    fun loadResNet101ONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet101.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createResNet152ONNXModel(@TempDir tempDir: File) {
        val resnet152v1File = tempDir.resolve(ONNXModels.CV.ResNet152.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet152)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet152v1File.isFile)
    }

    @Test
    fun loadResNet152ONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet152.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createResNet18v2ONNXModel(@TempDir tempDir: File) {
        val resnet18v2v1File = tempDir.resolve(ONNXModels.CV.ResNet18v2.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet18v2)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet18v2v1File.isFile)
    }

    @Test
    fun loadResNet18v2ONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet18v2.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createResNet34v2ONNXModel(@TempDir tempDir: File) {
        val resnet34v2v1File = tempDir.resolve(ONNXModels.CV.ResNet34v2.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet34v2)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet34v2v1File.isFile)
    }

    @Test
    fun loadResNet34v2ONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet34v2.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createResNet50v2ONNXModel(@TempDir tempDir: File) {
        val resnet50v2v1File = tempDir.resolve(ONNXModels.CV.ResNet50v2.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet50v2)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet50v2v1File.isFile)
    }

    @Test
    fun loadResNet50v2ONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet50v2.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createResNet101v2ONNXModel(@TempDir tempDir: File) {
        val resnet101v2v1File = tempDir.resolve(ONNXModels.CV.ResNet101v2.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet101v2)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet101v2v1File.isFile)
    }

    @Test
    fun loadResNet101v2ONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet101v2.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createResNet152v2ONNXModel(@TempDir tempDir: File) {
        val resnet152v2File = tempDir.resolve(ONNXModels.CV.ResNet152v2.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet152v2)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet152v2File.isFile)
    }

    @Test
    fun loadResNet152v2ONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet152v2.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createResNet50CustomONNXModel(@TempDir tempDir: File) {
        val resnet50Customv1File = tempDir.resolve(ONNXModels.CV.ResNet50custom.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet50custom)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet50Customv1File.isFile)
    }

    @Test
    fun loadResNet50CustomONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet50custom.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createResNet50TopCustomONNXModel(@TempDir tempDir: File) {
        val resnet50TopCustomv1File = tempDir.resolve(ONNXModels.CV.ResNet50noTopCustom.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.ResNet50noTopCustom)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(resnet50TopCustomv1File.isFile)
    }

    @Test
    fun loadResNet50TopCustomONNXPretrainedModel() {
        val model = ONNXModels.CV.ResNet50noTopCustom.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createEfficientNet4LiteONNXModel(@TempDir tempDir: File) {
        val EfficientNet4Litev1File = tempDir.resolve(ONNXModels.CV.EfficientNet4Lite.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.EfficientNet4Lite)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(EfficientNet4Litev1File.isFile)
    }

    @Test
    fun loadEfficientNet4LiteONNXPretrainedModel() {
        val model = ONNXModels.CV.EfficientNet4Lite.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createDenseNet121ONNXModel(@TempDir tempDir: File) {
        val DenseNet121v1File = tempDir.resolve(ONNXModels.CV.DenseNet121.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.CV.DenseNet121)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(DenseNet121v1File.isFile)
    }

    @Test
    fun loadDenseNet121ONNXPretrainedModel() {
        val model = ONNXModels.CV.DenseNet121.pretrainedModel(modelHub)
        assertNotNull(model)
    }

    @Test
    fun createSSDONNXModel(@TempDir tempDir: File) {
        val SSDv1File = tempDir.resolve(ONNXModels.ObjectDetection.SSD.modelRelativePath + ".onnx")
        val onnxModel = ONNXModelHub(tempDir).loadModel(ONNXModels.ObjectDetection.SSD)

        assertNotNull(onnxModel)
        assertTrue(tempDir.isDirectory)
        assertTrue(SSDv1File.isFile)
    }

    @Test
    fun loadSSDONNXPretrainedModel() {
        val model = ONNXModels.ObjectDetection.SSD.pretrainedModel(modelHub)
        assertNotNull(model)
    }
}