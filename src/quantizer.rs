#[derive(Clone, Debug)]
pub struct Quantizer {
    pub min: f32,
    pub max: f32,
}

impl Quantizer {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    pub fn compute_bounds(vectors: &[Vec<f32>]) -> (f32, f32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

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

        // Add padding to prevent boundary issues
        (min - 0.01, max + 0.01)
    }

    pub fn quantize(&self, vec: &[f32]) -> Vec<u8> {
        let range = self.max - self.min;
        let scale = if range.abs() < f32::EPSILON {
            0.0
        } else {
            255.0 / range
        };

        vec.iter()
            .map(|&v| {
                let normalized = (v - self.min) * scale;
                normalized.clamp(0.0, 255.0) as u8
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_bounds_with_padding() {
        let (min, max) = Quantizer::compute_bounds(&vec![vec![0.0, 1.0, 2.0], vec![-1.0, 3.0]]);
        assert!(min < -1.0);
        assert!(max > 3.0);
    }

    #[test]
    fn quantizes_into_byte_range() {
        let q = Quantizer::new(0.0, 10.0);
        let bytes = q.quantize(&[0.0, 5.0, 10.0, 20.0]);
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 127);
        assert_eq!(bytes[2], 255);
        assert_eq!(bytes[3], 255); // clamped
    }
}
