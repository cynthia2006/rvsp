use std::ffi::CString;

use gl::types::GLuint;

pub(crate) fn ensure_shader_compilation(shader: GLuint) -> Result<(), CString> {
    let mut success = 0;

    unsafe {
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
    }

    if success != 0 {
        Ok(())
    } else {
        let mut log = [0u8; 1024];
        let mut log_length = 0;

        unsafe {
            gl::GetShaderInfoLog(
                shader,
                log.len() as i32,
                &mut log_length,
                log.as_mut_ptr() as *mut _,
            );

            Err(CString::from_vec_with_nul_unchecked(
                log[..log_length as usize + 1].into(),
            ))
        }
    }
}

pub(crate) fn ensure_shader_linking(program: GLuint) -> Result<(), CString> {
    let mut success = 0;

    unsafe {
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
    }

    if success != 0 {
        Ok(())
    } else {
        let mut log = [0u8; 1024];
        let mut log_length = 0;

        unsafe {
            gl::GetProgramInfoLog(
                program,
                log.len() as i32,
                &mut log_length,
                log.as_mut_ptr() as *mut _,
            );

            Err(CString::from_vec_with_nul_unchecked(
                log[..log_length as usize + 1].into(),
            ))
        }
    }
}
