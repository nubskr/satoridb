#[derive(Clone, Debug)]
pub struct Quantizer {
    pub min: f32,
    pub max: f32,
    scale: f32, // 255.0 / (max - min)
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

        // Relative padding: 0.1% of range, with a small floor.
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
    // uses mul_add where possible; avoids clamp() overhead
    let mut v = x.mul_add(scale, -min * scale);
    if v < 0.0 {
        v = 0.0;
    } else if v > 255.0 {
        v = 255.0;
    }
    v as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_bounds_with_padding() {
        let (min, max) =
            Quantizer::compute_bounds(&vec![vec![0.0, 1.0, 2.0], vec![-1.0, 3.0]]).unwrap();
        assert!(min < -1.0);
        assert!(max > 3.0);
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
}
