package org.jetbrains.kotlinx.dl.example.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.AdapterView.OnItemSelectedListener
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA
import androidx.camera.core.CameraSelector.DEFAULT_FRONT_CAMERA
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.android.synthetic.main.activity_main.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException


class MainActivity : AppCompatActivity() {
    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }

    @Volatile
    private lateinit var cameraProcessor: CameraProcessor
    private var currentPipeline: Int = 0
    private var isBackCamera: Boolean = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        savedInstanceState?.apply {
            currentPipeline = getInt(CURRENT_PIPELINE, 0)
            isBackCamera = getBoolean(IS_BACK_CAMERA, true)
        }

        if (allPermissionsGranted()) {
            startCamera(currentPipeline, isBackCamera)
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        detector_view.scaleType = viewFinder.scaleType
    }

    private fun startCamera(currentPipelineIndex: Int, isBackCamera: Boolean) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val imageAnalyzer = ImageAnalyzer(
                applicationContext, resources, ::updateUI,
                currentPipelineIndex
            )
            runOnUiThread {
                cameraProcessor = CameraProcessor(
                    imageAnalyzer,
                    cameraProviderFuture.get(),
                    viewFinder.surfaceProvider,
                    backgroundExecutor,
                    isBackCamera
                )
                if (!cameraProcessor.bindCameraUseCases(this)) {
                    showError("Could not initialize camera.")
                }

                val modelsSpinnerAdapter = PipelineSelectorAdapter(
                    this,
                    R.layout.pipelines_selector,
                    imageAnalyzer.pipelinesList
                )
                models.adapter = modelsSpinnerAdapter
                models.onItemSelectedListener = ModelItemSelectedListener()
                models.setSelection(imageAnalyzer.currentPipelineIndex, false)

                backCameraSwitch.isChecked = cameraProcessor.isBackCamera
                backCameraSwitch.setOnCheckedChangeListener { _, isChecked ->
                    if (!cameraProcessor.setBackCamera(isChecked, this)) {
                        showError("Could not switch to the lens facing ${if (cameraProcessor.isBackCamera) "back" else "front"}.")
                    }
                }
            }
        }, backgroundExecutor)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera(currentPipeline, isBackCamera)
            } else {
                showError("Permissions not granted by the user.")
            }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        if (::cameraProcessor.isInitialized) {
            outState.putInt(CURRENT_PIPELINE, cameraProcessor.imageAnalyzer.currentPipelineIndex)
            outState.putBoolean(IS_BACK_CAMERA, cameraProcessor.isBackCamera)
        }
    }

    private fun showError(text: String) {
        Toast.makeText(this, text, Toast.LENGTH_SHORT).show()
        finish()
    }

    private fun updateUI(result: AnalysisResult?) {
        runOnUiThread {
            clearUi()
            if (result == null) {
                detector_view.setDetection(null)
                return@runOnUiThread
            }

            if (result is AnalysisResult.WithPrediction) {
                detector_view.setDetection(result)
                detected_item_text.text = result.prediction.getText(this)
                val confidencePercent = result.prediction.confidence * 100
                percentMeter.progress = confidencePercent.toInt()
                detected_item_confidence.text = "%.2f%%".format(confidencePercent)
            } else {
                detector_view.setDetection(null)
            }
            inference_time_value.text = getString(R.string.inference_time_placeholder, result.processTimeMs)
        }
    }

    private fun clearUi() {
        detected_item_text.text = ""
        detected_item_confidence.text = ""
        inference_time_value.text = ""
        percentMeter.progress = 0
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::cameraProcessor.isInitialized) cameraProcessor.close()
        backgroundExecutor.shutdown()
    }

    companion object {
        const val TAG = "KotlinDL demo app"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val CURRENT_PIPELINE = "current_pipeline"
        private const val IS_BACK_CAMERA = "is_back_camera"
    }

    private inner class ModelItemSelectedListener : OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
            if (::cameraProcessor.isInitialized) cameraProcessor.imageAnalyzer.setPipeline(position)
        }

        override fun onNothingSelected(p0: AdapterView<*>?) {
            if (::cameraProcessor.isInitialized) cameraProcessor.imageAnalyzer.clear()
        }
    }
}

private class CameraProcessor(
    val imageAnalyzer: ImageAnalyzer,
    private val cameraProvider: ProcessCameraProvider,
    private val surfaceProvider: Preview.SurfaceProvider,
    private val executor: ExecutorService,
    isInitialBackCamera: Boolean
) {
    @Volatile
    var isBackCamera: Boolean = isInitialBackCamera
        private set
    private val cameraSelector get() = if (isBackCamera) DEFAULT_BACK_CAMERA else DEFAULT_FRONT_CAMERA

    fun bindCameraUseCases(lifecycleOwner: LifecycleOwner): Boolean {
        try {
            cameraProvider.unbindAll()

            val imagePreview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also {
                    it.setSurfaceProvider(surfaceProvider)
                }
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(executor, ImageAnalyzerProxy(imageAnalyzer, isBackCamera))
                }

            if (cameraProvider.hasCamera(cameraSelector)) {
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    imagePreview,
                    imageAnalysis
                )
                return true
            }
        } catch (exc: RuntimeException) {
            Log.e(MainActivity.TAG, "Use case binding failed", exc)
        }
        return false
    }

    fun setBackCamera(backCamera: Boolean, lifecycleOwner: LifecycleOwner): Boolean {
        if (backCamera == isBackCamera) return true

        isBackCamera = backCamera
        return bindCameraUseCases(lifecycleOwner)
    }

    fun close() {
        cameraProvider.unbindAll()
        try {
            executor.submit { imageAnalyzer.close() }.get(500, TimeUnit.MILLISECONDS)
        } catch (_: InterruptedException) {
        } catch (_: TimeoutException) {
        }
    }
}

private class ImageAnalyzerProxy(private val delegate: ImageAnalyzer, private val isBackCamera: Boolean): ImageAnalysis.Analyzer {
    override fun analyze(image: ImageProxy) {
        delegate.analyze(image, !isBackCamera)
    }
}