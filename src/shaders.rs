use std::ffi::CString;

use lazy_static::lazy_static;

lazy_static! {
    pub(crate) static ref VERTEX_SHADER_SRC: CString = CString::new(
        r"#version 330 core
    layout (location = 0) in float x;
    layout (location = 1) in float y;

    void main() {
        gl_Position = vec4(x, y, 0.0f, 1.0f);
    }
    "
    )
    .unwrap();
    
    pub(crate) static ref FRAGMENT_SHADER_SRC: CString = CString::new(
        r"#version 330 core
    out vec4 FragColor;

    void main() {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }"
    )
    .unwrap();
}