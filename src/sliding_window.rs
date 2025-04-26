/// A special kind of circular buffer, where a write pointer moves forward and a read pointer lags.
/// Data is read past that pointer, in circular fashion, until reaching the write pointer again.
/// It is done so to imitate a window that slides over a string of data—something which it is
/// eponymous for.
pub(crate) struct SlidingWindow<T, const N: usize> 
{
    buffer: [T; N],
    cursor: usize,
}

impl<T, const N: usize> SlidingWindow<T, N> {
    pub(crate) fn new(buffer: [T; N], cursor: usize) -> Self {
        Self {
            buffer,
            cursor
        }
    }

    pub(crate) fn iter(&self) -> SlidingWindowIter<T, N> {
        SlidingWindowIter { window: self, index: self.cursor, capacity: N }
    }

    pub(crate) fn put(&mut self, elem: T) {
        self.buffer[self.cursor] = elem;
        self.cursor = (self.cursor + 1) % self.buffer.len();
    }
}

pub(crate) struct SlidingWindowIter<'a, T, const N: usize> {
    window: &'a SlidingWindow<T, N>,
    index: usize,
    capacity: usize,
}

impl<'a, T, const N: usize> Iterator for SlidingWindowIter<'a, T, N>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.capacity == 0 {
            None
        } else {
            let elem = &self.window.buffer[self.index];

            self.index = (self.index + 1) % self.window.buffer.len();
            self.capacity -= 1;
            
            Some(elem)
        }
    }
}
