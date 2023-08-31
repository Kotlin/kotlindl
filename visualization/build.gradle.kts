project.description = "This module contains the Kotlin API for visualization of the Deep Learning models built with the KotlinDL."

plugins {
    kotlin("multiplatform")
    id("com.android.library")
}

kotlin {
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "1.8"
        }
    }
    android {
        publishLibraryVariants("release")
    }
    sourceSets {
        val jvmMain by getting {
            dependencies {
                api(project(":api"))
                api(project(":tensorflow"))

                api("org.jetbrains.lets-plot:lets-plot-batik:2.3.0") {
                    exclude(group = "org.slf4j", module = "slf4j-simple")
                }
                api("org.jetbrains.lets-plot:lets-plot-common:2.3.0") {
                    exclude(group = "org.slf4j", module = "slf4j-simple")
                }
                api("org.jetbrains.lets-plot:lets-plot-kotlin-api:2.0.1") {
                    exclude(group = "org.slf4j", module = "slf4j-simple")
                }
            }
        }
        val androidMain by getting {
            dependencies {
                api(project(":api"))
                api("androidx.camera:camera-view:1.0.0-alpha22")
            }
        }
    }
}

android {
    compileSdkVersion(31)
    namespace = "org.jetbrains.kotlinx.dl.visualization"
    defaultConfig {
        minSdkVersion(24)
        targetSdkVersion(31)
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
}