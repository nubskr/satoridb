#[derive(Clone, Debug)]
pub struct Quantizer {
    pub min: f32,
    pub max: f32,
    scale: f32,
}

impl Quantizer {
    pub fn new(min: f32, max: f32) -> Self {
        let range = max - min;
        let scale = if range.is_finite() && range.abs() >= f32::EPSILON {
            255.0 / range
        } else {
            0.0
        };
        Self { min, max, scale }
    }

    /// Returns (min, max). If input is empty, returns None.
    pub fn compute_bounds(vectors: &[Vec<f32>]) -> Option<(f32, f32)> {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for vec in vectors {
            for &val in vec {
                if val < min {
                    min = val;
                }
                if val > max {
                    max = val;
                }
            }
        }

        Self::compute_bounds_from_minmax(min, max)
    }

    /// Returns (min, max) after applying the same padding policy as `compute_bounds`.
    pub fn compute_bounds_from_minmax(min: f32, max: f32) -> Option<(f32, f32)> {
        if !min.is_finite() || !max.is_finite() || min > max {
            return None;
        }

        let range = (max - min).abs();
        let pad = (range * 0.001).max(1e-3);
        Some((min - pad, max + pad))
    }

    #[inline(always)]
    pub fn quantize_into(&self, src: &[f32], dst: &mut Vec<u8>) {
        dst.clear();
        dst.resize(src.len(), 0);

        let min = self.min;
        let scale = self.scale;

        let mut i = 0usize;
        let len = src.len();

        while i + 4 <= len {
            dst[i] = quant_one(src[i], min, scale);
            dst[i + 1] = quant_one(src[i + 1], min, scale);
            dst[i + 2] = quant_one(src[i + 2], min, scale);
            dst[i + 3] = quant_one(src[i + 3], min, scale);
            i += 4;
        }
        while i < len {
            dst[i] = quant_one(src[i], min, scale);
            i += 1;
        }
    }

    #[inline(always)]
    pub fn quantize(&self, src: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(src.len());
        self.quantize_into(src, &mut out);
        out
    }
}

#[inline(always)]
fn quant_one(x: f32, min: f32, scale: f32) -> u8 {
    let v = x.mul_add(scale, -min * scale);
    v.clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_bounds() {
        assert!(
            Quantizer::compute_bounds(&[vec![0.0, 1.0, 2.0], vec![-1.0, 3.0]]).unwrap()
                == (-1.004, 3.004)
        );
    }

    #[test]
    fn empty_bounds_returns_none() {
        assert!(Quantizer::compute_bounds(&[]).is_none());
        assert!(Quantizer::compute_bounds(&[vec![]]).is_none());
    }

    #[test]
    fn quantizes_into_byte_range() {
        let q = Quantizer::new(0.0, 10.0);
        let bytes = q.quantize(&[0.0, 5.0, 10.0, 20.0]);
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 127);
        assert_eq!(bytes[2], 255);
        assert_eq!(bytes[3], 255);
    }

    /// When min == max (zero range), scale should be 0 and all values quantize to 0.
    #[test]
    fn zero_range_quantizes_to_zero() {
        let q = Quantizer::new(5.0, 5.0);
        assert_eq!(q.scale, 0.0, "zero range should have scale=0");
        let bytes = q.quantize(&[5.0, 5.0, 5.0]);
        assert!(
            bytes.iter().all(|&b| b == 0),
            "all values should quantize to 0"
        );
    }

    /// Very small range (< f32::EPSILON) triggers scale=0 to avoid overflow.
    #[test]
    fn tiny_range_uses_zero_scale() {
        let q = Quantizer::new(0.0, 1e-10);
        assert_eq!(q.scale, 0.0, "tiny range should use scale=0");
        let bytes = q.quantize(&[0.0, 1e-10]);
        assert!(
            bytes.iter().all(|&b| b == 0),
            "scale=0 means all values -> 0"
        );
    }

    /// Range just above f32::EPSILON should work normally.
    #[test]
    fn range_above_epsilon_works() {
        let q = Quantizer::new(0.0, 1e-6);
        assert!(
            q.scale > 0.0,
            "range above epsilon should have positive scale"
        );
        let bytes = q.quantize(&[0.0, 1e-6]);
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 255);
    }

    /// Negative values should quantize correctly.
    #[test]
    fn negative_values_quantize() {
        let q = Quantizer::new(-10.0, 10.0);
        let bytes = q.quantize(&[-10.0, 0.0, 10.0]);
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 127);
        assert_eq!(bytes[2], 255);
    }

    /// Values outside range should clamp to 0 or 255.
    #[test]
    fn out_of_range_clamps() {
        let q = Quantizer::new(0.0, 100.0);
        let bytes = q.quantize(&[-50.0, 150.0]);
        assert_eq!(bytes[0], 0, "below min should clamp to 0");
        assert_eq!(bytes[1], 255, "above max should clamp to 255");
    }

    /// compute_bounds_from_minmax with invalid inputs returns None.
    #[test]
    fn invalid_bounds_returns_none() {
        assert!(Quantizer::compute_bounds_from_minmax(10.0, 5.0).is_none());
        assert!(Quantizer::compute_bounds_from_minmax(f32::NAN, 5.0).is_none());
        assert!(Quantizer::compute_bounds_from_minmax(0.0, f32::NAN).is_none());
        assert!(Quantizer::compute_bounds_from_minmax(f32::NEG_INFINITY, 5.0).is_none());
        assert!(Quantizer::compute_bounds_from_minmax(0.0, f32::INFINITY).is_none());
    }

    /// Vectors with all same values should still produce valid bounds.
    #[test]
    fn constant_vectors_produce_bounds() {
        let result = Quantizer::compute_bounds(&[vec![5.0, 5.0, 5.0]]);
        assert!(result.is_some());
        let (min, max) = result.unwrap();
        assert!(min < max, "padding should ensure min < max");
    }

    /// quantize_into reuses the destination buffer.
    #[test]
    fn quantize_into_reuses_buffer() {
        let q = Quantizer::new(0.0, 10.0);
        let mut dst = Vec::with_capacity(100);
        q.quantize_into(&[0.0, 5.0, 10.0], &mut dst);
        assert_eq!(dst.len(), 3);
        assert_eq!(dst[0], 0);
        assert_eq!(dst[1], 127);
        assert_eq!(dst[2], 255);

        q.quantize_into(&[2.5], &mut dst);
        assert_eq!(dst.len(), 1);
        assert_eq!(dst[0], 63);
    }
}
