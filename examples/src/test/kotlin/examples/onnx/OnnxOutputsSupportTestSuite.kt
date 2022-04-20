package examples.onnx

import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertDoesNotThrow
import org.junit.jupiter.api.assertThrows
import kotlin.random.Random

class OnnxOutputsSupportTestSuite {
    private val pathToModel: String = OnnxInferenceModel::class.java.classLoader
        .getResource("models/onnx/lgbmSequenceOutput.onnx")!!.path
    private val model = OnnxInferenceModel.load(pathToModel)
    private val features = (1..27).map { Random.nextFloat() }.toFloatArray()

    init {
        model.reshape(27)
    }

    @Test
    fun predictSoftlyLgbmSequenceOutputTest() {
        assertThrows<IllegalArgumentException> {
            model.predictSoftly(features, "probabilities")
        }
    }

    @Test
    fun predictRawLgbmSequenceOutputTest() {
        assertDoesNotThrow {
            model.predictRaw(features)
        }
    }
}
