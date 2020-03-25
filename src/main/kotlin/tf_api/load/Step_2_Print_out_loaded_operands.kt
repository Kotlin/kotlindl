package tf_api.load

import org.tensorflow.Graph
import org.tensorflow.GraphOperation
import org.tensorflow.SavedModelBundle

const val PATH_TO_MODEL = "src/main/resources/model4"

fun main() {
    SavedModelBundle.load(PATH_TO_MODEL, "serve").use { bundle ->
        val session = bundle.session()
        val graph = bundle.graph()

        printTFGraph(graph)

        session.close()
    }
}

private fun printTFGraph(graph: Graph) {
    val operations = graph.operations()

    while (operations.hasNext()) {
        val operation = operations.next() as GraphOperation
        println("Name: " + operation.name() + "; Type: " + operation.type() + "; Out #tensors:  " + operation.numOutputs())
    }
}