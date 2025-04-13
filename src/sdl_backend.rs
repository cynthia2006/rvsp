use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::rc::Rc;

use glium::backend::{Backend, Context, Facade};

use glium::debug::DebugCallbackBehavior;
use glium::{Frame, IncompatibleOpenGl};

use sdl3;
use sdl3::video::{GLContext, Window};

pub(crate) struct SDLBackend {
    window: Rc<RefCell<Window>>,
    context: GLContext,
}

impl SDLBackend {
    pub(crate) fn new(window: Rc<RefCell<Window>>) -> Result<Self, sdl3::Error> {
        let context = window.borrow().gl_create_context()?;

        Ok(Self { window, context })
    }
}

unsafe impl Backend for SDLBackend {
    fn swap_buffers(&self) -> Result<(), glium::SwapBuffersError> {
        self.window.borrow().gl_swap_window();

        Ok(())
    }

    unsafe fn get_proc_address(&self, symbol: &str) -> *const std::os::raw::c_void {
        sdl3::sys::video::SDL_GL_GetProcAddress(
            CString::new(symbol).unwrap().as_ptr() as *const c_char
        )
        .unwrap() as *const _
    }

    fn get_framebuffer_dimensions(&self) -> (u32, u32) {
        self.window.borrow().size()
    }

    fn resize(&self, _new_size: (u32, u32)) {
        unimplemented!()
    }

    fn is_current(&self) -> bool {
        unimplemented!()
    }

    unsafe fn make_current(&self) {
        self.window
            .borrow_mut()
            .gl_make_current(&self.context)
            .unwrap()
    }
}

pub(crate) struct Display {
    context: Rc<Context>,
}

impl Display {
    pub(crate) fn new(backend: SDLBackend) -> Result<Self, IncompatibleOpenGl> {
        let context = unsafe { Context::new(backend, false, DebugCallbackBehavior::default())? };

        Ok(Self { context })
    }

    pub(crate) fn draw(&self) -> Frame {
        let dimensions = self.context.get_framebuffer_dimensions();
        Frame::new(self.context.clone(), dimensions)
    }
}

impl Facade for Display {
    fn get_context(&self) -> &std::rc::Rc<Context> {
        &self.context
    }
}
