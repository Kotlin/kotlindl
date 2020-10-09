package api.inference.savedmodel

/**
 * Possible outputs for static TensorFlow graph in [SavedModel].
 *
 * @property [tfName] Maps TensorFlow operand name to enum value.
 */
public enum class Output(public val tfName: String) {
    /** ArgMax. */
    ARGMAX("ArgMax")
}
