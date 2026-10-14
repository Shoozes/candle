//! Immutable KV pages. Transactions share complete pages and replace only tails.

use candle::{bail, DType, Result, Tensor};
use std::collections::VecDeque;

use super::memory::PAGE_ROWS;

#[derive(Debug, Clone)]
struct Page {
    keys: Tensor,
    values: Tensor,
    rows: usize,
    first: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PagedKvCache {
    pages: VecDeque<Page>,
    rows: usize,
}

impl PagedKvCache {
    pub(crate) fn append(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        window: Option<usize>,
    ) -> Result<()> {
        let (rows, width) = key.dims2()?;
        if rows != 1
            || width == 0
            || value.dims2()? != (1, width)
            || key.dtype() != DType::F32
            || value.dtype() != DType::F32
            || !key.device().same_device(value.device())
            || window == Some(0)
        {
            bail!("GPT-OSS KV append requires matching nonempty F32 rows and a positive window");
        }
        let next_rows = self
            .rows
            .checked_add(1)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS KV page row count overflowed".into()))?;
        if let Some(tail) = self.pages.back().filter(|page| page.rows < PAGE_ROWS) {
            // A new allocation never changes a page held by a committed root.
            let next = Page {
                keys: Tensor::cat(&[&tail.keys, key], 0)?,
                values: Tensor::cat(&[&tail.values, value], 0)?,
                rows: tail.rows + 1,
                first: tail.first,
            };
            *self
                .pages
                .back_mut()
                .ok_or_else(|| candle::Error::Msg("missing KV tail".into()))? = next;
        } else {
            self.pages.try_reserve(1).map_err(candle::Error::wrap)?;
            // Own exact row allocations; a QKV view must not retain its parent.
            self.pages.push_back(Page {
                keys: key.force_contiguous()?,
                values: value.force_contiguous()?,
                rows: 1,
                first: 0,
            });
        }
        self.rows = next_rows;
        if let Some(window) = window {
            let mut excess = self.rows.saturating_sub(window);
            while excess > 0 {
                let first = self
                    .pages
                    .front_mut()
                    .ok_or_else(|| candle::Error::Msg("missing KV front".into()))?;
                let count = (first.rows - first.first).min(excess);
                first.first += count;
                excess -= count;
                self.rows -= count;
                if first.first == first.rows {
                    self.pages.pop_front();
                }
            }
        }
        Ok(())
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn storage_bytes(&self) -> Result<usize> {
        self.pages.iter().try_fold(0usize, |sum, page| {
            let width = page.keys.dims2()?.1;
            page.rows
                .checked_mul(width)
                .and_then(|n| n.checked_mul(8))
                .and_then(|n| sum.checked_add(n))
                .ok_or_else(|| candle::Error::Msg("GPT-OSS KV page storage overflowed".into()))
        })
    }

    pub(crate) fn materialize(&self) -> Result<(Tensor, Tensor)> {
        if self.pages.is_empty() {
            bail!("GPT-OSS attention requires cached KV rows");
        }
        let mut keys = Vec::new();
        let mut values = Vec::new();
        keys.try_reserve_exact(self.pages.len())
            .map_err(candle::Error::wrap)?;
        values
            .try_reserve_exact(self.pages.len())
            .map_err(candle::Error::wrap)?;
        for page in &self.pages {
            let rows = page.rows - page.first;
            keys.push(page.keys.narrow(0, page.first, rows)?);
            values.push(page.values.narrow(0, page.first, rows)?);
        }
        Ok((Tensor::cat(&keys, 0)?, Tensor::cat(&values, 0)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn page_boundaries_snapshot_rollback_and_window_retention() -> Result<()> {
        let mut cache = PagedKvCache::default();
        let mut sliding = PagedKvCache::default();
        for i in 0..(2 * PAGE_ROWS + 3) {
            let row = Tensor::from_slice(&[i as f32, -(i as f32)], (1, 2), &Device::Cpu)?;
            cache.append(&row, &row, None)?;
            sliding.append(&row, &row, Some(3))?;
            assert_eq!(cache.rows(), i + 1);
            assert_eq!(cache.storage_bytes()?, (i + 1) * 16);
            assert!(sliding.rows() <= 3);
            assert!(sliding.storage_bytes()? <= (3 + PAGE_ROWS - 1) * 16);
            if [PAGE_ROWS - 1, PAGE_ROWS, PAGE_ROWS + 1, 2 * PAGE_ROWS].contains(&(i + 1)) {
                let before = cache.materialize()?.0.to_vec2::<f32>()?;
                let mut staged = cache.clone();
                staged.append(&row, &row, None)?;
                assert_eq!(cache.materialize()?.0.to_vec2::<f32>()?, before);
                assert_eq!(staged.rows(), cache.rows() + 1);
                if cache.pages.len() > 1 || cache.pages.back().unwrap().rows == PAGE_ROWS {
                    assert_eq!(
                        cache.pages.front().unwrap().keys.id(),
                        staged.pages.front().unwrap().keys.id()
                    );
                }
            }
        }
        let (keys, values) = sliding.materialize()?;
        assert_eq!(keys.to_vec2::<f32>()?, values.to_vec2::<f32>()?);
        assert_eq!(
            keys.to_vec2::<f32>()?
                .iter()
                .map(|r| r[0])
                .collect::<Vec<_>>(),
            vec![128., 129., 130.]
        );
        let before = sliding.storage_bytes()?;
        let bad = Tensor::zeros((1, 0), DType::F32, &Device::Cpu)?;
        assert!(sliding.append(&bad, &bad, None).is_err());
        let row = Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?;
        assert!(sliding.append(&row, &row, Some(0)).is_err());
        assert_eq!(sliding.storage_bytes()?, before);
        assert!(PagedKvCache::default().materialize().is_err());
        Ok(())
    }
}
