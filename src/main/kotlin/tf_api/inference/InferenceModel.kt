package tf_api.inference

import org.tensorflow.Session
import org.tensorflow.Shape
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import tf_api.KGraph
import tf_api.keras.shape.TensorShape
import java.io.File
import java.nio.file.NotDirectoryException
import java.util.*

open class InferenceModel() : AutoCloseable {
    protected lateinit var session: Session
    lateinit var kGraph: KGraph

    /** The namespace wrapper for all TensorFlow graph operations. */
    protected lateinit var tf: Ops

    open fun predict(image: FloatArray): Int {
        fun reshape2(image: FloatArray): Tensor<*>? {
            val reshaped = Array(
                1
            ) { Array(28) { Array(28) { FloatArray(1) } } }
            for (i in image.indices) reshaped[0][i / 28][i % 28][0] = image[i]
            return Tensor.create(reshaped)
        }

        val runner1 = session.runner()
        val result1 = runner1.feed("x", reshape2(image))
            .fetch("output")
            .run()[0]

        val arr = Array(1) { FloatArray(10) { 0.0f } }
        result1.copyTo(arr)

        println(arr.contentDeepToString())

        val runner = session.runner()
        val result = runner.feed("x", reshape2(image))
            .fetch("ArgMax")
            .run()[0]

        return result.copyTo(LongArray(1))[0].toInt()
    }

    fun predict(inputData: DoubleArray): Double {
        return 0.0
    }

    fun predict(inputData: List<DoubleArray>): List<Double> {
        return listOf()
    }


    override fun toString(): String {
        return "Model contains $kGraph"
    }

    override fun close() {
        session.close()
        kGraph.close()
    }

    /**
     * Loads model as graph and weights.
     */
    fun load(pathToModelDirectory: String) {
        // Load graph
        val directory = File(pathToModelDirectory);
        if (!directory.exists()) {
            throw NotDirectoryException(pathToModelDirectory)
        } else {
            kGraph = KGraph(File("$pathToModelDirectory/graph.pb").readBytes())
            tf = Ops.create(kGraph.tfGraph)
            session = Session(kGraph.tfGraph)

            // Load variables names
            val variableNames = File("$pathToModelDirectory/variableNames.txt").readLines()
            if (variableNames.isNotEmpty()) {
                for (variableName in variableNames) {
                    val shape = kGraph.tfGraph.operation(variableName).output<Float>(0).shape()
                    val tensorShape = TensorShape(shape)


                    println(variableName)
                    println(tensorShape.dims().contentToString())
                    println(tensorShape.numElements())

                    /* // limited by 2GB files
                     val tensorData = File("$pathToModelDirectory/$variableName.txt").readText()*/
                    val scanner = Scanner(File("$pathToModelDirectory/$variableName.txt").inputStream())
                    scanner.useLocale(Locale.US)

                    when (variableName) {
                        "Variable_2" -> {
                            val initializerName = "Xavier"
                            val assignOpName = "Assign"

                            val source = create4DimFloatArray(shape, scanner)
                            populateVariable(initializerName, source, assignOpName)

                            // Extract variable
                            /* val variableExtractor = session.runner()
                                        val variableTensors = variableExtractor
                                            .fetch(variableName)
                                            .run();

                                        val dst = create4DimFloatArray(shape, 0.0f)

                                        variableTensors[0].copyTo(dst)
                                        println(dst[0][0][0][0])*/


                        }
                        "Variable_3" -> {
                            val initializerName = "Xavier_1"
                            val assignOpName = "Assign_1"

                            val source = create1DimFloatArray(shape, scanner)
                            populateVariable(initializerName, source, assignOpName)
                        }
                        "Variable_6" -> {
                            val initializerName = "Xavier_2"
                            val assignOpName = "Assign_2"

                            val source = create4DimFloatArray(shape, scanner)

                            // populate variable
                            populateVariable(initializerName, source, assignOpName)
                        }
                        "Variable_7" -> {
                            val initializerName = "Xavier_3"
                            val assignOpName = "Assign_3"

                            val org = create1DimFloatArray(shape, scanner)

                            // populate variable
                            populateVariable(initializerName, org, assignOpName)
                        }
                        "Variable_8" -> {
                            val initializerName = "Xavier_4"
                            val assignOpName = "Assign_4"

                            val org = create2DimFloatArray(shape, scanner)

                            // populate variable
                            populateVariable(initializerName, org, assignOpName)
                        }
                        "Variable_9" -> {
                            val initializerName = "Xavier_5"
                            val assignOpName = "Assign_5"

                            val org = create1DimFloatArray(shape, scanner)

                            // populate variable
                            populateVariable(initializerName, org, assignOpName)
                        }
                        "Variable_10" -> {
                            val initializerName = "Xavier_6"
                            val assignOpName = "Assign_6"

                            val org = create2DimFloatArray(shape, scanner)

                            // populate variable
                            populateVariable(initializerName, org, assignOpName)
                        }
                        "Variable_11" -> {
                            val initializerName = "Xavier_7"
                            val assignOpName = "Assign_7"

                            val org = create1DimFloatArray(shape, scanner)

                            // populate variable
                            populateVariable(initializerName, org, assignOpName)
                        }
                        "Variable_12" -> {
                            val initializerName = "Xavier_8"
                            val assignOpName = "Assign_8"

                            val org = create2DimFloatArray(shape, scanner)

                            // populate variable
                            populateVariable(initializerName, org, assignOpName)
                        }
                        "Variable_13" -> {
                            val initializerName = "Xavier_9"
                            val assignOpName = "Assign_9"

                            val org = create1DimFloatArray(shape, scanner)

                            // populate variable
                            populateVariable(initializerName, org, assignOpName)
                        }
                    }
                }
            }
        }
    }

    private fun populateVariable(
        initializerName: String,
        org: Any,
        assignOpName: String
    ) {
        session.runner()
            .feed(initializerName, Tensor.create(org))
            .addTarget(assignOpName)
            .run();
    }

    private fun create4DimFloatArray(
        shape: Shape,
        scanner: Scanner
    ): Array<Array<Array<FloatArray>>> {
        val result = Array(shape.size(0).toInt()) {
            Array(shape.size(1).toInt()) {
                Array(shape.size(2).toInt()) {
                    FloatArray(shape.size(3).toInt()) { 0.0f }
                }
            }
        }

        var cnt = 0

        for (i in result.indices) {
            for (j in result[i].indices) {
                for (k in result[i][j].indices) {
                    for (m in result[i][j][k].indices) {
                        if (scanner.hasNextFloat()) {
                            val weight = scanner.nextFloat()
                            result[i][j][k][m] = weight
                            cnt++
                        } else {
                            println(cnt)
                        }
                    }
                }
            }
        }

        return result
    }

    private fun create3DimFloatArray(
        shape: Shape,
        scanner: Scanner
    ): Array<Array<FloatArray>> {
        val result = Array(shape.size(0).toInt()) {
            Array(shape.size(1).toInt()) {
                FloatArray(shape.size(2).toInt()) { 0.0f }
            }
        }

        for (i in result.indices) {
            for (j in result[i].indices) {
                for (k in result[i][j].indices) {

                    val weight = scanner.nextFloat()
                    result[i][j][k] = weight

                }
            }
        }

        return result
    }

    private fun create2DimFloatArray(
        shape: Shape,
        scanner: Scanner
    ): Array<FloatArray> {
        val result = Array(shape.size(0).toInt()) {
            FloatArray(shape.size(1).toInt()) { 0.0f }
        }

        for (i in result.indices) {
            for (j in result[i].indices) {
                val weight = scanner.nextFloat()
                result[i][j] = weight


            }
        }

        return result
    }

    private fun create1DimFloatArray(
        shape: Shape,
        scanner: Scanner
    ): FloatArray {
        val result = FloatArray(shape.size(0).toInt()) { 0.0f }

        for (i in result.indices) {

            val weight = scanner.nextFloat()
            result[i] = weight

        }

        return result
    }
}
