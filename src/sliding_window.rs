/// An adaptor for special kind of circular buffer, where only a write pointer moves forward.
/// Data is read past that pointer, in circular fashion, until reaching the write pointer again.
/// It is done so to imitate a window that slides over a string of data—something which it is
/// eponymous for.
#[derive(Debug)]
pub(crate) struct SlidingWindowAdapter<'a, T> {
    buffer: &'a mut [T],
    cursor: usize,
}

impl<'a, T> SlidingWindowAdapter<'a, T> {
    pub(crate) fn new(buffer: &'a mut [T], cursor: usize) -> Self {
        Self { buffer, cursor }
    }

    pub(crate) fn iter(&'a self) -> SlidingWindowIter<'a, T> {
        SlidingWindowIter {
            window: self,
            index: self.cursor,
            capacity: self.buffer.len(),
        }
    }

    pub(crate) fn put(&mut self, elem: T) {
        self.buffer[self.cursor] = elem;
        self.cursor = (self.cursor + 1) % self.buffer.len();
    }
}

pub(crate) struct SlidingWindowIter<'a, T> {
    window: &'a SlidingWindowAdapter<'a, T>,
    index: usize,
    capacity: usize,
}

impl<'a, T> Iterator for SlidingWindowIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
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
