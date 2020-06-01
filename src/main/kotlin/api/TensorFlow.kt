package api

class TensorFlow {
    companion object {
        fun version(): String {
            return org.tensorflow.TensorFlow.version()
        }
    }
}