package video.zhuker.sancho

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import java.io.File

class SrtmTest {
    @Test
    fun readSrtm() {
        val file = File("/Users/azhukov/personal/uav_pixel_to_coord/data/elevation/srtm_nasa/N37W123.hgt")
        val ef = SrtmElevationFile.loadHgt(file)
        assertEquals(3601, ef.square_side)
        assertEquals(0.0002777777777777778, ef.resolution, 0.00000001)
        assertEquals(39.0, ef.get_elevation(37.551116, -122.087915))
        println(ef.get_elevation(37.24909210805589, -122.95113440773974))
        println(ef.get_elevation(37.2490921080559, -122.087915))
        assertEquals(560.905048961587, ef.get_precise_elevation(37.2490921080559, -122.087915))
    }
}