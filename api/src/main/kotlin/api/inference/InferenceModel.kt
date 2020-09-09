package api.inference

import api.core.KGraph
import api.inference.savedmodel.Input
import api.inference.savedmodel.Output
import api.keras.ModelFormat
import api.keras.shape.TensorShape
import api.keras.util.DATA_PLACEHOLDER
import api.keras.util.OUTPUT_NAME
import api.keras.util.defaultAssignOpName
import api.keras.util.defaultInitializerOpName
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

open class InferenceModel() : AutoCloseable {
    /** The namespace wrapper for all TensorFlow graph operations. */
    protected lateinit var tf: Ops

    /** TensorFlow session. */
    lateinit var session: Session

    /** TensorFlow wrapped computational graph. */
    lateinit var kGraph: KGraph

    private val logger = KotlinLogging.logger {}

    protected var input: Input = Input.PLACEHOLDER

    protected var output: Output = Output.ARGMAX

    private lateinit var reshape: (FloatArray) -> Tensor<*>?

    protected var mu.KLogger.level: Level
        get() = (logger.underlyingLogger as Logger).level
        set(value) {
            (underlyingLogger as Logger).level = value
        }

    open fun predict(image: FloatArray): Int {
        reshape(image).use { tensor ->
            val runner = session.runner()

            val result = runner.feed(DATA_PLACEHOLDER, tensor)
                .fetch(output.tfName)
                .run()[0]

            return result.copyTo(LongArray(1))[0].toInt()

        }
    }

    open fun predictSoftly(image: FloatArray, predictionTensorName: String = OUTPUT_NAME): FloatArray {
        reshape(image).use { tensor ->
            val runner1 = session.runner()
            val result1 = runner1.feed(DATA_PLACEHOLDER, tensor)
                .fetch(OUTPUT_NAME)
                .run()[0]

            val arr = Array(1) { FloatArray(10) { 0.0f } }
            result1.copyTo(arr)

            return arr[0]
        }
    }

    fun input(inputOp: Input) {
        input = inputOp
    }

    fun output(outputOp: Output) {
        output = outputOp
    }

    fun reshape(reshapeFunction: (FloatArray) -> Tensor<*>?) {
        reshape = reshapeFunction
    }

    override fun toString(): String {
        return "Model contains $kGraph"
    }

    override fun close() {
        session.close()
        kGraph.close()
    }

    fun runAssignOpByVarName(
        variableName: String
    ) {
        val assignOpName = defaultAssignOpName(variableName)

        session.runner()
            .addTarget(assignOpName)
            .run()
    }

    fun addInitOpsToGraph(
        variableName: String,
        kernelData: Any
    ) {
        val initializerName = defaultInitializerOpName(variableName)
        val assignOpName = defaultAssignOpName(variableName)

        populateVariable(initializerName, kernelData, assignOpName)
    }

    /**
     * Loads model as graph and weights.
     */
    fun load(
        pathToModelDirectory: String,
        modelFormat: ModelFormat = ModelFormat.TF_GRAPH_CUSTOM_VARIABLES,
        loadOptimizerState: Boolean = false
    ) {
        // Load graph
        val directory = File(pathToModelDirectory)
        if (!directory.exists()) {
            throw NotDirectoryException(pathToModelDirectory)
        } else {
            logger.debug { "The model loading is started." }

            when (modelFormat) {
                ModelFormat.TF_GRAPH_CUSTOM_VARIABLES -> loadModelFromSimpleFormat(
                    pathToModelDirectory,
                    loadOptimizerState
                )
                ModelFormat.TF_GRAPH -> loadModelFromSavedModelFormat(pathToModelDirectory)
                ModelFormat.KERAS_CONFIG_CUSTOM_VARIABLES -> throw UnsupportedOperationException("The inference model requires the graph in .pb format to load. To load Sequential model in Keras format use Sequential.load(..) instead. ")
            }

            logger.debug { "The model loading is finished." }
        }
    }

    open fun loadVariablesFromTxtFiles(pathToModelDirectory: String, loadOptimizerState: Boolean = false) {
        loadVariablesFromTxt(pathToModelDirectory, loadOptimizerState)
    }

    protected fun loadVariablesFromTxt(pathToModelDirectory: String, loadOptimizerState: Boolean) {
        // Load variables names
        val variableNames = File("$pathToModelDirectory/variableNames.txt").readLines()
        if (variableNames.isNotEmpty()) {
            for (variableName in variableNames) {
                if (!loadOptimizerState && variableName.startsWith("optimizer")) // skip loading optimizers' variables
                    continue
                loadVariable(variableName, pathToModelDirectory)
            }
        }
    }

    protected fun loadVariable(variableName: String, pathToModelDirectory: String) {
        val operation = kGraph.tfGraph.operation(variableName)
        check(operation != null) { "Operation $variableName is not found in static graph." }
        val scanner = Scanner(File("$pathToModelDirectory/$variableName.txt").inputStream())
        try {
            scanner.useLocale(Locale.US)
            val initializerName = defaultInitializerOpName(variableName)
            val assignOpName = defaultAssignOpName(variableName)

            val shape = operation.output<Float>(0).shape()
            val tensorShape = TensorShape(shape)

            val source = createFloatArrayFromScanner(shape, scanner)
            populateVariable(initializerName, source, assignOpName)

            logger.debug { "Loading the variable $variableName data" }
            logger.debug { "Variable dimensions are: ${tensorShape.dims().contentToString()}" }
            logger.debug { "Amount of elements: ${tensorShape.numElements()}" }
        } catch (ex: Exception) {
            logger.error { ex.message }
        } finally {
            scanner.close()
        }
    }

    private fun loadModelFromSavedModelFormat(pathToModelDirectory: String) {
        throw UnsupportedOperationException("Model loading from saved ")
    }

    private fun loadModelFromSimpleFormat(pathToModelDirectory: String, loadOptimizerState: Boolean) {
        inferenceGraphInitialization(pathToModelDirectory)
        loadVariablesFromTxtFiles(pathToModelDirectory, loadOptimizerState)
    }

    private fun inferenceGraphInitialization(pathToModelDirectory: String) {
        kGraph = KGraph(File("$pathToModelDirectory/graph.pb").readBytes())
        tf = Ops.create(kGraph.tfGraph)
        session = Session(kGraph.tfGraph)
    }

    private fun createFloatArrayFromScanner(shape: Shape, scanner: Scanner): Any {
        when (shape.numDimensions()) {
            0 -> {
                return scanner.nextFloat()
            }
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
                throw RuntimeException("The loading of tensors with 5 and more dimensions is not supported yet")
            }
        }
    }

    private fun populateVariable(
        initializerName: String,
        data: Any,
        assignOpName: String
    ) {
        Tensor.create(data).use { tensor ->
            session.runner()
                .feed(initializerName, tensor)
                .addTarget(assignOpName)
                .run()

        }
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
                            logger.debug { cnt }
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
