project.description = "This module contains the Kotlin API for loading and executing the ONNX models."

plugins {
    kotlin("multiplatform")
    id("com.android.library")
}

kotlin {
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "1.8"
        }
        testRuns["test"].executionTask.configure {
            useJUnitPlatform()
        }
    }
    android {
        publishLibraryVariants("release")
    }
    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":api"))
                api(project(":impl"))
            }
        }
        val jvmMain by getting {
            dependencies {
                api("org.jetbrains.kotlinx:multik-core:0.2.0")
                api("org.jetbrains.kotlinx:multik-default:0.2.0")
                api("com.microsoft.onnxruntime:onnxruntime:1.14.0")
                api("io.github.microutils:kotlin-logging:2.1.21")
            }
        }
        val jvmTest by getting {
            dependencies {
                implementation("org.junit.jupiter:junit-jupiter-api:5.8.2")
                implementation("org.junit.jupiter:junit-jupiter-engine:5.8.2")
                implementation("org.junit.jupiter:junit-jupiter-params:5.8.2")
                implementation(project(":api"))
                implementation(project(":dataset"))
            }
        }
        val androidMain by getting {
            dependencies {
                api("com.microsoft.onnxruntime:onnxruntime-android:1.14.0")
                api("androidx.camera:camera-core:1.0.0-rc03")
            }
        }
    }
    explicitApiWarning()
}

android {
    compileSdkVersion(31)
    namespace = "org.jetbrains.kotlinx.dl.onnx"
    defaultConfig {
        minSdkVersion(24)
        targetSdkVersion(31)
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
}