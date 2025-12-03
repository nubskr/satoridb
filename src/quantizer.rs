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
                if val < min { min = val; }
                if val > max { max = val; }
            }
        }
        
        // Add padding to prevent boundary issues
        (min - 0.01, max + 0.01)
    }

    pub fn quantize(&self, vec: &[f32]) -> Vec<u8> {
        let range = self.max - self.min;
        let scale = if range.abs() < f32::EPSILON { 0.0 } else { 255.0 / range };

        vec.iter().map(|&v| {
            let normalized = (v - self.min) * scale;
            normalized.clamp(0.0, 255.0) as u8
        }).collect()
    }
}
