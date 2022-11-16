project.setDescription("A KotlinDL library provides Dataset API for better Kotlin programming for Deep Learning.")

apply plugin: 'kotlin-multiplatform'

kotlin {
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = '1.8'
        }
    }
    sourceSets {
        jvmMain {
            dependencies {
                api project(':api')
                api project(':impl')
            }
        }
    }
    explicitApiWarning()
}