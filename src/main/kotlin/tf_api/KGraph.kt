package tf_api

import org.tensorflow.Graph
import org.tensorflow.GraphOperation

class KGraph(graphDef: ByteArray) : AutoCloseable {
    var tfGraph: Graph = Graph()

    init {
        tfGraph.importGraphDef(graphDef)
    }

    override fun close() {
        tfGraph.close()
    }

    override fun toString(): String {
        return convertGraphDefToString()
    }

    private fun convertGraphDefToString(): String {
        val operations = tfGraph.operations()

        var s = ""
        while (operations.hasNext()) {
            val operation = operations.next() as GraphOperation
            s += "Name: " + operation.name() + "; Type: " + operation.type() + "; Out #tensors:  " + operation.numOutputs() + "\n"
        }
        return s
    }
}
