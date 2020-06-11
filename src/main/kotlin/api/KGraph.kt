package api

import org.tensorflow.Graph
import org.tensorflow.GraphOperation
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.AssignAdd

class KGraph(graphDef: ByteArray, prefix: String) : AutoCloseable {
    constructor(graphDef: ByteArray) : this(graphDef, "")

    var tfGraph: Graph = Graph()

    /** A list of initializer to initialize the trainableVariables. */
    var optimizerInitializers: List<Assign<*>> = listOf()

    /** A list of initializer to initialize the trainableVariables. */
    var optimizerAssignAddInitializers: List<AssignAdd<*>> = listOf()


    init {
        if (prefix.isEmpty()) {
            tfGraph.importGraphDef(graphDef)
        } else {
            tfGraph.importGraphDef(graphDef, prefix)
        }
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
