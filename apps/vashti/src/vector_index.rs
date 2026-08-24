#[derive(Debug)]
pub struct VectorIndex<M> {
    dimensions: usize,
    vectors: Vec<f32>,
    metadata: Vec<M>,
}

impl<M> VectorIndex<M> {
    pub fn build(entries: Vec<(M, Vec<f32>)>) -> Result<Self, String> {
        let dimensions = entries
            .first()
            .map(|(_, vector)| vector.len())
            .unwrap_or_default();
        if dimensions == 0 && !entries.is_empty() {
            return Err("embedding vector cannot be empty".to_string());
        }

        let mut vectors = Vec::with_capacity(entries.len().saturating_mul(dimensions));
        let mut metadata = Vec::with_capacity(entries.len());
        for (item, mut vector) in entries {
            if vector.len() != dimensions {
                return Err("embedding vectors have inconsistent dimensions".to_string());
            }
            normalize(&mut vector)?;
            vectors.extend(vector);
            metadata.push(item);
        }

        Ok(Self {
            dimensions,
            vectors,
            metadata,
        })
    }

    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    pub fn allocated_vector_bytes(&self) -> usize {
        self.vectors.capacity() * size_of::<f32>()
    }

    pub fn for_each_score(
        &self,
        query: &[f32],
        mut visit: impl FnMut(usize, &M, f32),
    ) -> Result<(), String> {
        if self.is_empty() {
            return Ok(());
        }
        if query.len() != self.dimensions {
            return Err("query and index dimensions do not match".to_string());
        }

        let mut normalized_query = query.to_vec();
        normalize(&mut normalized_query)?;
        for (index, (vector, metadata)) in self
            .vectors
            .chunks_exact(self.dimensions)
            .zip(&self.metadata)
            .enumerate()
        {
            let score = dot_product(&normalized_query, vector);
            visit(index, metadata, score);
        }
        Ok(())
    }
}

pub fn normalize(vector: &mut [f32]) -> Result<(), String> {
    if vector.is_empty() {
        return Err("embedding vector cannot be empty".to_string());
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err("embedding vector contains a non-finite value".to_string());
    }

    let norm_squared = vector.iter().map(|value| value * value).sum::<f32>();
    if !norm_squared.is_finite() || norm_squared <= f32::EPSILON {
        return Err("embedding vector has zero magnitude".to_string());
    }
    let inverse_norm = norm_squared.sqrt().recip();
    for value in vector {
        *value *= inverse_norm;
    }
    Ok(())
}

#[inline]
fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    let mut sum_0 = 0.0;
    let mut sum_1 = 0.0;
    let mut sum_2 = 0.0;
    let mut sum_3 = 0.0;
    let mut left_chunks = left.chunks_exact(4);
    let mut right_chunks = right.chunks_exact(4);
    while let (Some(left), Some(right)) = (left_chunks.next(), right_chunks.next()) {
        sum_0 += left[0] * right[0];
        sum_1 += left[1] * right[1];
        sum_2 += left[2] * right[2];
        sum_3 += left[3] * right[3];
    }
    let left_remainder = left_chunks.remainder();
    let right_remainder = right_chunks.remainder();
    sum_0
        + sum_1
        + sum_2
        + sum_3
        + left_remainder
            .iter()
            .zip(right_remainder)
            .map(|(left, right)| left * right)
            .sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_search_normalizes_vectors_once() {
        let index = VectorIndex::build(vec![
            ("east", vec![10.0, 0.0]),
            ("north", vec![0.0, 4.0]),
            ("northeast", vec![1.0, 1.0]),
        ])
        .unwrap();

        let mut hits = Vec::new();
        index
            .for_each_score(&[2.0, 0.0], |_, item, score| hits.push((*item, score)))
            .unwrap();
        hits.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        assert_eq!(hits[0].0, "east");
        assert_eq!(hits[1].0, "northeast");
        assert!((hits[0].1 - 1.0).abs() < 0.0001);
    }

    #[test]
    fn search_can_filter_without_copying_metadata() {
        let index = VectorIndex::build(vec![
            (1, vec![1.0, 0.0]),
            (2, vec![0.9, 0.1]),
            (3, vec![0.0, 1.0]),
        ])
        .unwrap();

        let mut hits = Vec::new();
        index
            .for_each_score(&[1.0, 0.0], |_, item, score| {
                if *item != 1 {
                    hits.push((*item, score));
                }
            })
            .unwrap();
        hits.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        assert_eq!(hits[0].0, 2);
        assert_eq!(hits[1].0, 3);
    }

    #[test]
    fn invalid_vectors_are_rejected() {
        assert!(VectorIndex::<()>::build(vec![((), Vec::new())]).is_err());
        assert!(VectorIndex::build(vec![((), vec![1.0]), ((), vec![1.0, 2.0])]).is_err());
        assert!(normalize(&mut [f32::NAN]).is_err());
        assert!(normalize(&mut [0.0, 0.0]).is_err());
    }
}
