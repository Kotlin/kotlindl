package api.inference

import api.KGraph
import api.keras.shape.TensorShape
import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import mu.KotlinLogging
import org.tensorflow.Session
import org.tensorflow.Shape
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import java.io.File
import java.nio.file.NotDirectoryException
import java.util.*

private val logger = KotlinLogging.logger {}

open class InferenceModel() : AutoCloseable {
    /** The namespace wrapper for all TensorFlow graph operations. */
    protected lateinit var tf: Ops

    /** TensorFlow session. */
    protected lateinit var session: Session

    /** TensorFlow wrapped computational graph. */
    lateinit var kGraph: KGraph

    protected var mu.KLogger.level: Level
        get() = (logger.underlyingLogger as Logger).level
        set(value) {
            (underlyingLogger as Logger).level = value
        }

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

        logger.debug { arr.contentDeepToString() }

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
            logger.debug { "The model loading is started." }

            kGraph = KGraph(File("$pathToModelDirectory/graph.pb").readBytes())
            tf = Ops.create(kGraph.tfGraph)
            session = Session(kGraph.tfGraph)

            // Load variables names
            val variableNames = File("$pathToModelDirectory/variableNames.txt").readLines()
            if (variableNames.isNotEmpty()) {
                for (variableName in variableNames) {
                    val shape = kGraph.tfGraph.operation(variableName).output<Float>(0).shape()
                    val tensorShape = TensorShape(shape)

                    logger.debug { "Loading the variable $variableName data" }
                    logger.debug { "Variable dimensions are: ${tensorShape.dims().contentToString()}"}
                    logger.debug { "Amount of elements: ${tensorShape.numElements()}" }

                    /* // limited by 2GB files
                     val tensorData = File("$pathToModelDirectory/$variableName.txt").readText()*/
                    val scanner = Scanner(File("$pathToModelDirectory/$variableName.txt").inputStream())
                    scanner.useLocale(Locale.US)

                    // TODO: describe this convention about naming
                    val initializerName = "Init_$variableName"
                    val assignOpName = "Assign_$variableName"

                    val source = createFloatArrayFromScanner(shape, scanner)
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
            }

            logger.debug { "The model loading is ended." }
        }


    }


    private fun createFloatArrayFromScanner(shape: Shape, scanner: Scanner): Any {
        when (shape.numDimensions()) {
            1 -> {
                return create1DimFloatArray(shape, scanner)
            }
            2 -> {
                return create2DimFloatArray(shape, scanner)
            }
            3 -> {
                return create3DimFloatArray(shape, scanner)
            }
            4 -> {
                return create4DimFloatArray(shape, scanner)
            }
            else -> {
                throw RuntimeException("The loading of tensors with 5 and more dimenstions is not supported yet")
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
