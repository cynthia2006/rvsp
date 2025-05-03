use std::{mem::MaybeUninit, ptr};

use gl::types::{GLsizeiptr, GLuint};

const VERTEX_SHADER_SRC: &[u8] = b"#version 100
    precision mediump float;
    attribute vec2 coords;
    
    void main() {
        gl_Position = vec4(coords, 0.0, 1.0);
    }\0";

const FRAGMENT_SHADER_SRC: &[u8] = b"#version 100
    precision mediump float;
    void main() {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }\0";

pub(crate) struct PolygonRenderer {
    vao: GLuint,
    program: GLuint,
    line_width: f32,
}

impl PolygonRenderer {
    pub(crate) unsafe fn new(line_width: f32) -> Self {
        let mut vbo = MaybeUninit::uninit();
        let mut vao = MaybeUninit::uninit();

        let program = gl::CreateProgram();
        let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
        let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);

        gl::ShaderSource(
            vertex_shader,
            1,
            &VERTEX_SHADER_SRC.as_ptr().cast(),
            ptr::null(),
        );
        gl::ShaderSource(
            fragment_shader,
            1,
            &FRAGMENT_SHADER_SRC.as_ptr().cast(),
            ptr::null(),
        );
        gl::CompileShader(vertex_shader);
        gl::CompileShader(fragment_shader);
        gl::AttachShader(program, vertex_shader);
        gl::AttachShader(program, fragment_shader);
        gl::LinkProgram(program);
        gl::DeleteShader(vertex_shader);
        gl::DeleteShader(fragment_shader);

        gl::GenBuffers(1, vbo.as_mut_ptr());
        gl::GenVertexArrays(1, vao.as_mut_ptr());

        let vao = vao.assume_init();
        let vbo = vbo.assume_init();

        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::VertexAttribPointer(
            0,
            2,
            gl::FLOAT,
            gl::FALSE,
            2 * size_of::<f32>() as i32,
            0 as *const _,
        );
        gl::EnableVertexAttribArray(0);

        Self {
            vao,
            program,
            line_width,
        }
    }

    pub(crate) unsafe fn clear(&self, color: (f32, f32, f32)) {
        gl::ClearColor(color.0, color.1, color.2, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);
    }

    pub(crate) unsafe fn upload(&self, plot: &[f32]) {
        gl::UseProgram(self.program);
        gl::BindVertexArray(self.vao);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (plot.len() * size_of::<f32>()) as GLsizeiptr,
            plot.as_ptr() as *const _,
            gl::STREAM_DRAW,
        );
        gl::LineWidth(self.line_width);

        // Length of this array must be aligned to the number of vertices.
        gl::DrawArrays(gl::LINE_STRIP, 0, plot.len() as i32 / 2);
    }

    pub(crate) fn set_line_width(&mut self, line_width: f32) {
        self.line_width = line_width;
    }
}
