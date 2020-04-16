package tf_api.keras.initializers

enum class Initializers {
    ZEROS,
    ONES,
    TRUNCATED_NORMAL,
    RANDOM_NORMAL,
    CONSTANT;

    /* companion object {
         fun <T : Number> convert(initializer: Initializers): Initializer<T> {
             return when (initializer) {
                 ZEROS -> Zeros()
                 ONES -> Ones()
                 TRUNCATED_NORMAL -> TruncatedNormal()
                 RANDOM_NORMAL -> RandomNormal()
                 CONSTANT -> Constant()
             }
         }
     }*/
}