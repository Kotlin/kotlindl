plugins {
    kotlin("jvm")
}

dependencies {
    api(project(":api"))
    api(project(":impl"))
    api(project(":tensorflow"))
    api(project(":dataset"))
    api(project(":visualization"))
    api(project(":onnx"))

    implementation("org.apache.logging.log4j:log4j-api:2.17.2")
    implementation("org.apache.logging.log4j:log4j-core:2.17.2")
    implementation("org.apache.logging.log4j:log4j-slf4j-impl:2.17.2")

    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.2")
    testImplementation("org.junit.jupiter:junit-jupiter-engine:5.8.2")
    testImplementation("org.junit.jupiter:junit-jupiter-params:5.8.2")
    // to run on GPU (if CUDA is updated and machine with NVIDIA onboard)
    /*implementation("org.tensorflow:libtensorflow:1.15.0")
    implementation("org.tensorflow:libtensorflow_jni_gpu:1.15.0")
    api("com.microsoft.onnxruntime:onnxruntime_gpu:1.12.1")
    */
}

val publishedArtifactsVersion = System.getenv("KOTLIN_DL_RELEASE_VERSION")
if (!publishedArtifactsVersion.isNullOrBlank()) {
    configurations.all {
        resolutionStrategy.dependencySubstitution {
            substitute(project(":api")).with(module("org.jetbrains.kotlinx:kotlin-deeplearning-api:$publishedArtifactsVersion"))
            substitute(project(":impl")).with(module("org.jetbrains.kotlinx:kotlin-deeplearning-impl:$publishedArtifactsVersion"))
            substitute(project(":tensorflow")).with(module("org.jetbrains.kotlinx:kotlin-deeplearning-tensorflow:$publishedArtifactsVersion"))
            substitute(project(":dataset")).with(module("org.jetbrains.kotlinx:kotlin-deeplearning-dataset:$publishedArtifactsVersion"))
            substitute(project(":visualization")).with(module("org.jetbrains.kotlinx:kotlin-deeplearning-visualization:$publishedArtifactsVersion"))
            substitute(project(":onnx")).with(module("org.jetbrains.kotlinx:kotlin-deeplearning-onnx:$publishedArtifactsVersion"))
        }
    }
}

tasks.compileKotlin {
    kotlinOptions.jvmTarget = "1.8"
}

tasks.compileTestKotlin {
    kotlinOptions.jvmTarget = "1.8"
}

tasks.named<Test>("test") {
    useJUnitPlatform()
    // set heap size for the test JVM(s)
    minHeapSize = "1024m"
    maxHeapSize = "8g"
}