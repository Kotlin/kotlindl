package examples.onnx

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertDoesNotThrow
import org.junit.jupiter.api.assertThrows
import kotlin.random.Random

class OnnxOutputsSupportTestSuite {
    private val pathToModel: String = getFileFromResource("models/onnx/lgbmSequenceOutput.onnx").absolutePath
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
            model.predictRaw(features to TensorShape(27))
        }
    }
}
