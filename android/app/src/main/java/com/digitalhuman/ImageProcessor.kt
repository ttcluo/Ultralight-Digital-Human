package com.digitalhuman

import android.graphics.Bitmap
import android.graphics.Rect
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * 图像处理：裁剪、拼接，与 inference_onnx.py 逻辑一致。
 */
object ImageProcessor {

    data class LandmarkBbox(val xmin: Int, val ymin: Int, val xmax: Int, val ymax: Int)

    fun loadLandmarks(input: java.io.InputStream): Array<FloatArray> {
        BufferedReader(InputStreamReader(input)).use { reader ->
            return reader.readLines().map { line ->
                line.trim().split("\\s+".toRegex()).map { it.toFloat() }.toFloatArray()
            }.toTypedArray()
        }
    }

    fun getBbox(lms: Array<FloatArray>): LandmarkBbox {
        val xmin = lms[1][0].toInt()
        val ymin = lms[52][1].toInt()
        val xmax = lms[31][0].toInt()
        val width = xmax - xmin
        val ymax = ymin + width
        return LandmarkBbox(xmin, ymin, xmax, ymax)
    }

    /**
     * 从 Bitmap 裁剪人脸区域，resize 到 168x168，取 [4:164,4:164] 作为 img_real，
     * 涂黑 [5:150,5:145] 作为 img_masked，concat 为 [6, 160, 160] float32，范围 0~1。
     */
    fun prepareModelInput(bitmap: Bitmap, bbox: LandmarkBbox): FloatArray {
        val crop = Bitmap.createBitmap(bitmap, bbox.xmin, bbox.ymin, bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin)
        val resized = Bitmap.createScaledBitmap(crop, 168, 168, true)
        if (crop != resized && crop != bitmap) crop.recycle()

        val imgReal = FloatArray(3 * 160 * 160)
        val imgMasked = FloatArray(3 * 160 * 160)

        for (y in 0 until 160) {
            for (x in 0 until 160) {
                val px = x + 4
                val py = y + 4
                val pixel = resized.getPixel(px, py)
                val r = (pixel shr 16 and 0xFF) / 255f
                val g = (pixel shr 8 and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f
                val idx = (y * 160 + x) * 3
                imgReal[idx] = b
                imgReal[idx + 1] = g
                imgReal[idx + 2] = r

                val mask = if (x in 5..149 && y in 5..144) 0f else 1f
                imgMasked[idx] = b * mask
                imgMasked[idx + 1] = g * mask
                imgMasked[idx + 2] = r * mask
            }
        }
        resized.recycle()

        val result = FloatArray(6 * 160 * 160)
        for (c in 0 until 3) {
            for (y in 0 until 160) {
                for (x in 0 until 160) {
                    result[c * 160 * 160 + y * 160 + x] = imgReal[(y * 160 + x) * 3 + c]
                    result[(3 + c) * 160 * 160 + y * 160 + x] = imgMasked[(y * 160 + x) * 3 + c]
                }
            }
        }
        return result
    }

    /**
     * 将模型输出 [3,160,160] BGR 贴回原图。与 inference_onnx 一致：crop 168x168，pred 填入 [4:164,4:164]，再 resize 到 crop 尺寸。
     */
    fun overlayPrediction(bitmap: Bitmap, bbox: LandmarkBbox, pred: FloatArray): Bitmap {
        val out = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val cropW = bbox.xmax - bbox.xmin
        val cropH = bbox.ymax - bbox.ymin

        val crop = Bitmap.createBitmap(bitmap, bbox.xmin, bbox.ymin, cropW, cropH)
        val full168 = Bitmap.createScaledBitmap(crop, 168, 168, true)
        crop.recycle()

        val inner = Bitmap.createBitmap(160, 160, Bitmap.Config.ARGB_8888)
        for (y in 0 until 160) {
            for (x in 0 until 160) {
                val b = (pred[y * 160 + x].coerceIn(0f, 1f) * 255).toInt()
                val g = (pred[160 * 160 + y * 160 + x].coerceIn(0f, 1f) * 255).toInt()
                val r = (pred[2 * 160 * 160 + y * 160 + x].coerceIn(0f, 1f) * 255).toInt()
                inner.setPixel(x, y, (255 shl 24) or (r shl 16) or (g shl 8) or b)
            }
        }
        android.graphics.Canvas(full168).drawBitmap(inner, 4f, 4f, null)
        inner.recycle()

        val scaled = Bitmap.createScaledBitmap(full168, cropW, cropH, true)
        full168.recycle()

        android.graphics.Canvas(out).drawBitmap(scaled, bbox.xmin.toFloat(), bbox.ymin.toFloat(), null)
        scaled.recycle()
        return out
    }
}
