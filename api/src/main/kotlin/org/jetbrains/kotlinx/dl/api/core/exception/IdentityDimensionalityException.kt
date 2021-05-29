package org.jetbrains.kotlinx.dl.api.core.exception

public class IdentityDimensionalityException(dimensions: Long):
        Exception("Identity matrix is not defined for order $dimensions tensors.")
