package com.digitalhuman

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File

class MainActivity : AppCompatActivity() {

    private var job: Thread? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), 1)
            }
        }

        findViewById<Button>(R.id.btnRun).setOnClickListener { runInference() }
    }

    private fun runInference() {
        if (job != null) return

        val progress = findViewById<ProgressBar>(R.id.progress)
        val status = findViewById<TextView>(R.id.status)
        progress.progress = 0
        status.text = "加载中..."

        val outputPath = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM), "digital_human_output.mp4").absolutePath

        job = Thread {
            try {
                val rootList = assets.list("") ?: emptyArray()
                if (!rootList.contains("unet.onnx")) {
                    throw IllegalStateException("请将 unet.onnx 放入 assets。见 android/README.md")
                }
                if (!rootList.contains("audio_feat.bin")) {
                    throw IllegalStateException("请将 audio_feat.bin 放入 assets。见 android/README.md")
                }
                val inference = DigitalHumanInference(this@MainActivity)
                assets.open("unet.onnx").use { inference.loadModels(it) }
                assets.open("audio_feat.bin").use { inference.loadAudioFeaturesFromBin(it) }
                inference.loadAvatarAssets("full_body_img", "landmarks")

                val w = inference.outputWidth
                val h = inference.outputHeight

                val encoder = VideoEncoder(w, h, 20)
                encoder.start(outputPath)

                inference.run { index, total, bitmap ->
                    encoder.encodeFrame(bitmap)
                    bitmap.recycle()
                    runOnUiThread {
                        progress.progress = (index + 1) * 100 / total
                        status.text = "推理 ${index + 1}/$total"
                    }
                }

                encoder.stop()
                inference.release()

                runOnUiThread {
                    status.text = "完成: $outputPath"
                    Toast.makeText(this@MainActivity, "已保存至 $outputPath", Toast.LENGTH_LONG).show()
                    job = null
                }
            } catch (e: Exception) {
                runOnUiThread {
                    status.text = "错误: ${e.message}"
                    Toast.makeText(this@MainActivity, "错误: ${e.message}", Toast.LENGTH_LONG).show()
                    job = null
                }
            }
        }.apply { start() }
    }

    override fun onDestroy() {
        super.onDestroy()
    }
}
