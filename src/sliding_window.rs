use std::marker::PhantomData;

#[derive(Debug)]
pub(crate) struct SlidingWindow<T, U>
where
    U: AsRef<[T]> + AsMut<[T]>,
{
    buffer: U,
    cursor: usize,
    _phantom: PhantomData<T>,
}

impl<T, U> SlidingWindow<T, U>
where
    U: AsRef<[T]> + AsMut<[T]>,
{
    pub(crate) fn new(buffer: U, cursor: usize) -> Self {
        Self {
            buffer,
            cursor,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn iter(&self) -> SlidingWindowIter<T, U> {
        let capacity = self.buffer.as_ref().len();

        SlidingWindowIter {
            window: self,
            index: self.cursor,
            capacity,
        }
    }

    pub(crate) fn put(&mut self, elem: T) {
        let buffer = self.buffer.as_mut();

        buffer[self.cursor] = elem;
        self.cursor = (self.cursor + 1) % buffer.len();
    }

    // pub(crate) fn into_inner(self) -> U {
    //     self.buffer
    // }
}

pub(crate) struct SlidingWindowIter<'a, T, U>
where
    U: AsRef<[T]> + AsMut<[T]>,
{
    window: &'a SlidingWindow<T, U>,
    index: usize,
    capacity: usize,
}

impl<'a, T, U: AsMut<[T]> + 'a> Iterator for SlidingWindowIter<'a, T, U>
where
    U: AsRef<[T]> + AsMut<[T]>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.capacity == 0 {
            None
        } else {
            let buffer = self.window.buffer.as_ref();
            let elem = &buffer[self.index];

            self.index = (self.index + 1) % buffer.len();
            self.capacity -= 1;

            Some(&elem)
        }
    }
}
