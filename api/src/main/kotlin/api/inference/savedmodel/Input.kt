package api.inference.savedmodel

/**
 * Possible inputs for static TensorFlow graph in [SavedModel].
 *
 * @property [tfName] Maps TensorFlow operand name to enum value.
 */
public enum class Input(public val tfName: String) {
    /** Placeholder. */
    PLACEHOLDER("Placeholder")
}
