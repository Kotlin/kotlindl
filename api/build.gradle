project.setDescription("This module contains the Kotlin API for building, training, and evaluating the Deep Learning models.")

apply plugin: 'kotlin'

compileKotlin {
    kotlinOptions.jvmTarget = "1.8"
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