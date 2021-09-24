**Release activities check-list for minor releases:**

0. Run code inspections (fix typos, Kotlin issue, fix code formatting)
1. Write missed KDocs
2. Generate new documentation using Dokka and deploy it with gh-pages branch
3. Add new release section to the CHANGELOG.md
4. Update tutorials according last code changes
5. Update README.MD according last code changes
6. Make release branch
7. Make last commit with release tag (_v0.1.1_ for example) to the release branch
8. Run tests and build artifacts on TC for the commit with the release tag
9. Deploy artifacts on MavenCentral based on the commit with the release tag
10. Check artifacts' availability on MavenCentral
11. Update project TFHelloWorld with new dependency to the released artifact
12. Move examples from the KotlinDL project to the separate branch of TFHelloWorld and run them as tests 
13. Run TFHelloWorld on different OS (Windows, Ubuntu, macOS)
14. Make fatJar from TFHelloWorld project and run on independent Amazon instance

