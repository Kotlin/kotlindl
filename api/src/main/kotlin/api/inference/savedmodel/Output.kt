package api.inference.savedmodel

/**
 * Possible outputs for static TensorFlow graph in [SavedModel].
 *
 * @property [tfName] Maps TensorFlow operand name to enum value.
 */
enum class Output(val tfName: String) {
    /** ArgMax. */
    ARGMAX("ArgMax")
}
