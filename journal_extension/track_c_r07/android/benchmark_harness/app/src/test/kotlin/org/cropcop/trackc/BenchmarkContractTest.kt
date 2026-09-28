package org.cropcop.trackc

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class BenchmarkContractTest {
    @Test fun `frozen benchmark modes are accepted`() {
        assertEquals("load", BenchmarkContract.requireMode("load"))
        assertEquals("end_to_end", BenchmarkContract.requireMode("end_to_end"))
        assertEquals(7, BenchmarkContract.modes.size)
    }

    @Test fun `unknown mode fails closed`() {
        assertThrows(IllegalArgumentException::class.java) {
            BenchmarkContract.requireMode("interactive_camera")
        }
    }
}
