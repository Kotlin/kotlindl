package org.jetbrains.kotlinx.dl.api.core.exception

/**
 * Thrown if it's not 2D matrix created.
 *
 * @param [dimensions] The cardinality of created matrix.
 */
public class IdentityDimensionalityException(dimensions: Long) :
    Exception("Identity matrix is not defined for order $dimensions tensors.")
