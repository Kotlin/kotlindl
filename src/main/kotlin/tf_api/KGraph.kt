package tf_api

import org.tensorflow.Graph
import org.tensorflow.GraphOperation

class KGraph(private val graphDef: ByteArray) {
    override fun toString(): String {
        Graph().use { g ->
            g.importGraphDef(graphDef)
            return convertGraphDef(g)
        }
    }

    private fun convertGraphDef(graph: Graph): String {
        val operations = graph.operations()

        var s = ""
        while (operations.hasNext()) {
            val operation = operations.next() as GraphOperation
            s += "Name: " + operation.name() + "; Type: " + operation.type() + "; Out #tensors:  " + operation.numOutputs() + "\n"
        }
        return s
    }
}