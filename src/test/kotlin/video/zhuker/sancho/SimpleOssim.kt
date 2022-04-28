package video.zhuker.sancho

data class Params(
    val latitude: Double,
    val longitude: Double,
    val elevation: Double,
    val roll: Double,
    val pitch: Double,
    val heading: Double,
    val px: Double,
    val py: Double,
    val focal_length: Double,
    val pix_size_x: Double,
    val pix_size_y: Double,
    val img_width: Int,
    val img_height: Int,
    val u: Double,
    val v: Double
)

object SimpleOssim {
    fun makeParams(lat: Double, lon: Double): Params {
        val params = Params(
            latitude = lat,
            longitude = lon,
            elevation = 108.8,
            roll = 0.0,
            pitch = 75.4,
            heading = 53.7,
            px = 0.0,
            py = 0.0,
            focal_length = 24.0,
            pix_size_x = 0.05693451728060701,
            pix_size_y = 0.05693451728060701,
            img_width = 1920,
            img_height = 1080,
            u = 960.0,
            v = 540.0
        )
        return params
    }

    fun findTarget(params: Params, elevManager: ossimElevManager? = null): ossimGpt {
        val model = ossimApplanixEcefModel(
            ossimDrect(ossimDpt(0.0, 0.0), ossimDpt(params.img_width.toDouble(), params.img_height.toDouble())),
            ossimGpt(params.latitude, params.longitude, params.elevation),
            params.roll,
            params.pitch,
            params.heading,
            ossimDpt(params.px, params.py),
            params.focal_length,
            ossimDpt(params.pix_size_x, params.pix_size_y),
            elevManager ?: ossimElevManager.instance()
        )
        model.setPrincipalPoint(ossimDpt(params.px, params.py))
        val res = ossimGpt()
        model.lineSampleToWorld(ossimDpt(params.u, params.v), res)
        return res
    }

}