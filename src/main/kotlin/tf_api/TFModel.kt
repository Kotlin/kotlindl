package tf_api

import org.tensorflow.SavedModelBundle
import util.MnistUtils
import java.io.File


class TFModel(val graph: KGraph) {
    fun predict(image: MnistUtils.Image): Double {
        return 0.0
    }

    fun predict(images: List<MnistUtils.Image>): List<Double> {
        return listOf()
    }

    companion object {
        fun loadModelFromFile(pathToModel: String): TFModel {
            val graphDef = File(pathToModel).readBytes()

            return TFModel(KGraph(graphDef))
        }

        fun loadModelFromDirectory(pathToModel: String): TFModel {
            var graphDef: ByteArray

            SavedModelBundle.load(pathToModel, "serve").use { bundle ->
                val session = bundle.session()
                val graph = bundle.graph()
                graphDef = graph.toGraphDef()
                session.close()
                return TFModel(KGraph(graphDef))
            }
        }
    }

    override fun toString(): String {
        return "Model contains $graph"
    }
}