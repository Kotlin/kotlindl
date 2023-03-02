project.setDescription("This module contains the Kotlin API for building, training, and evaluating TensorFlow models.")

apply plugin: 'kotlin'

dependencies {
    api project(":dataset")
    api project(":api")
    api project(":impl")
    api group: 'org.tensorflow', name: 'tensorflow', version: '1.15.0'
    api 'com.beust:klaxon:5.5'
    api 'io.github.microutils:kotlin-logging:2.1.21' // for logging
    testImplementation 'ch.qos.logback:logback-classic:1.2.11'
    testImplementation 'org.junit.jupiter:junit-jupiter-api:5.8.2'
    testImplementation 'org.junit.jupiter:junit-jupiter-engine:5.8.2'
    testImplementation 'org.junit.jupiter:junit-jupiter-params:5.8.2'
}

compileKotlin {
    kotlinOptions.jvmTarget = "1.8"
}

compileTestKotlin {
    kotlinOptions.jvmTarget = "1.8"
}

test {
    useJUnitPlatform()
}

kotlin {
    explicitApiWarning()
}

task sourcesJar(type: Jar) {
    classifier 'sources'
    from sourceSets.main.allSource
}

artifacts {
    archives sourcesJar
}