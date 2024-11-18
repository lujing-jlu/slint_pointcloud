// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: MIT

use rand::Rng;
use slint::platform::Platform;
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
use std::num::NonZeroU32;
use std::rc::Rc;

slint::include_modules!();

use glow::HasContext;

macro_rules! define_scoped_binding {
    (struct $binding_ty_name:ident => $obj_name:path, $param_name:path, $binding_fn:ident, $target_name:path) => {
        struct $binding_ty_name {
            saved_value: Option<$obj_name>,
            gl: Rc<glow::Context>,
        }

        impl $binding_ty_name {
            unsafe fn new(gl: &Rc<glow::Context>, new_binding: Option<$obj_name>) -> Self {
                let saved_value =
                    NonZeroU32::new(gl.get_parameter_i32($param_name) as u32).map($obj_name);

                gl.$binding_fn($target_name, new_binding);
                Self {
                    saved_value,
                    gl: gl.clone(),
                }
            }
        }

        impl Drop for $binding_ty_name {
            fn drop(&mut self) {
                unsafe {
                    self.gl.$binding_fn($target_name, self.saved_value);
                }
            }
        }
    };
    (struct $binding_ty_name:ident => $obj_name:path, $param_name:path, $binding_fn:ident) => {
        struct $binding_ty_name {
            saved_value: Option<$obj_name>,
            gl: Rc<glow::Context>,
        }

        impl $binding_ty_name {
            unsafe fn new(gl: &Rc<glow::Context>, new_binding: Option<$obj_name>) -> Self {
                let saved_value =
                    NonZeroU32::new(gl.get_parameter_i32($param_name) as u32).map($obj_name);

                gl.$binding_fn(new_binding);
                Self {
                    saved_value,
                    gl: gl.clone(),
                }
            }
        }

        impl Drop for $binding_ty_name {
            fn drop(&mut self) {
                unsafe {
                    self.gl.$binding_fn(self.saved_value);
                }
            }
        }
    };
}

define_scoped_binding!(struct ScopedTextureBinding => glow::NativeTexture, glow::TEXTURE_BINDING_2D, bind_texture, glow::TEXTURE_2D);
define_scoped_binding!(struct ScopedFrameBufferBinding => glow::NativeFramebuffer, glow::DRAW_FRAMEBUFFER_BINDING, bind_framebuffer, glow::DRAW_FRAMEBUFFER);
define_scoped_binding!(struct ScopedVBOBinding => glow::NativeBuffer, glow::ARRAY_BUFFER_BINDING, bind_buffer, glow::ARRAY_BUFFER);
define_scoped_binding!(struct ScopedVAOBinding => glow::NativeVertexArray, glow::VERTEX_ARRAY_BINDING, bind_vertex_array);

struct DemoTexture {
    texture: glow::Texture,
    width: u32,
    height: u32,
    fbo: glow::Framebuffer,
    gl: Rc<glow::Context>,
}

impl DemoTexture {
    unsafe fn new(gl: &Rc<glow::Context>, width: u32, height: u32) -> Self {
        let fbo = gl
            .create_framebuffer()
            .expect("Unable to create framebuffer");

        let texture = gl.create_texture().expect("Unable to allocate texture");

        let _saved_texture_binding = ScopedTextureBinding::new(gl, Some(texture));

        let old_unpack_alignment = gl.get_parameter_i32(glow::UNPACK_ALIGNMENT);
        let old_unpack_row_length = gl.get_parameter_i32(glow::UNPACK_ROW_LENGTH);
        let old_unpack_skip_pixels = gl.get_parameter_i32(glow::UNPACK_SKIP_PIXELS);
        let old_unpack_skip_rows = gl.get_parameter_i32(glow::UNPACK_SKIP_ROWS);

        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, width as i32);
        gl.pixel_store_i32(glow::UNPACK_SKIP_PIXELS, 0);
        gl.pixel_store_i32(glow::UNPACK_SKIP_ROWS, 0);

        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA as _,
            width as _,
            height as _,
            0,
            glow::RGBA as _,
            glow::UNSIGNED_BYTE as _,
            None,
        );

        let _saved_fbo_binding = ScopedFrameBufferBinding::new(gl, Some(fbo));

        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(texture),
            0,
        );

        debug_assert_eq!(
            gl.check_framebuffer_status(glow::FRAMEBUFFER),
            glow::FRAMEBUFFER_COMPLETE
        );

        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, old_unpack_alignment);
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, old_unpack_row_length);
        gl.pixel_store_i32(glow::UNPACK_SKIP_PIXELS, old_unpack_skip_pixels);
        gl.pixel_store_i32(glow::UNPACK_SKIP_ROWS, old_unpack_skip_rows);

        Self {
            texture,
            width,
            height,
            fbo,
            gl: gl.clone(),
        }
    }

    unsafe fn with_texture_as_active_fbo<R>(&self, callback: impl FnOnce() -> R) -> R {
        let _saved_fbo = ScopedFrameBufferBinding::new(&self.gl, Some(self.fbo));
        callback()
    }
}

impl Drop for DemoTexture {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_framebuffer(self.fbo);
            self.gl.delete_texture(self.texture);
        }
    }
}

struct PointCloud {
    gl: Rc<glow::Context>,
    positions: Vec<f32>,
    colors: Vec<f32>,
    vbo_pos: glow::Buffer,
    vbo_color: glow::Buffer,
    num_points: i32,
}

impl PointCloud {
    unsafe fn new(gl: &Rc<glow::Context>) -> Self {
        let mut rng = rand::thread_rng();
        let mut positions = Vec::new();
        let mut colors = Vec::new();

        // 生成10000个随机点
        const NUM_POINTS: usize = 10000;

        for _ in 0..NUM_POINTS {
            // 生成球形分布的随机点
            let theta = rng.gen_range(0.0f32..std::f32::consts::PI * 2.0);
            let phi = rng.gen_range(0.0f32..std::f32::consts::PI);
            let r = rng.gen_range(0.0f32..1.0).powf(1.0 / 3.0); // 使用立方根使点分布更均匀

            // 球坐标转笛卡尔坐标
            let x = r * phi.sin() * theta.cos();
            let y = r * phi.sin() * theta.sin();
            let z = r * phi.cos();

            positions.extend_from_slice(&[x, y, z]);

            // 生成随机颜色
            colors.extend_from_slice(&[
                rng.gen_range(0.0f32..1.0), // r
                rng.gen_range(0.0f32..1.0), // g
                rng.gen_range(0.0f32..1.0), // b
            ]);
        }

        let num_points = NUM_POINTS as i32;

        // 创建并上传位置数据
        let vbo_pos = gl.create_buffer().expect("Cannot create buffer");
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_pos));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            positions.align_to::<u8>().1,
            glow::STATIC_DRAW,
        );

        // 创建并上传颜色数据
        let vbo_color = gl.create_buffer().expect("Cannot create buffer");
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_color));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            colors.align_to::<u8>().1,
            glow::STATIC_DRAW,
        );

        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        Self {
            gl: gl.clone(),
            positions,
            colors,
            vbo_pos,
            vbo_color,
            num_points,
        }
    }
}

impl Drop for PointCloud {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_buffer(self.vbo_pos);
            self.gl.delete_buffer(self.vbo_color);
        }
    }
}

struct DemoRenderer {
    gl: Rc<glow::Context>,
    program: glow::Program,
    vao: glow::VertexArray,
    point_cloud: PointCloud,
    rotation_x_location: glow::UniformLocation,
    rotation_y_location: glow::UniformLocation,
    displayed_texture: DemoTexture,
    next_texture: DemoTexture,
    start_time: web_time::Instant,
    scale_location: glow::UniformLocation,
}

impl DemoRenderer {
    unsafe fn new(gl: glow::Context) -> Self {
        let gl = Rc::new(gl);
        unsafe {
            let program = gl.create_program().expect("Cannot create program");

            let vertex_shader_source = r#"#version 100
            attribute vec3 position;
            attribute vec3 color;
            varying vec3 v_color;
            
            uniform float rotation_x;
            uniform float rotation_y;
            uniform float scale;
            
            void main() {
                float rx = radians(rotation_x);
                float ry = radians(rotation_y);
                
                mat4 rotateY = mat4(
                    cos(ry), 0.0, sin(ry), 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    -sin(ry), 0.0, cos(ry), 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
                
                mat4 rotateX = mat4(
                    1.0, 0.0, 0.0, 0.0,
                    0.0, cos(rx), -sin(rx), 0.0,
                    0.0, sin(rx), cos(rx), 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
                
                vec4 rotated_pos = rotateX * rotateY * vec4(position, 1.0);
                
                rotated_pos.xyz *= scale;
                
                gl_Position = vec4(rotated_pos.xyz, 1.0);
                gl_PointSize = 2.0;
                v_color = color;
            }"#;

            let fragment_shader_source = r#"#version 100
            precision mediump float;
            varying vec3 v_color;
            
            void main() {
                gl_FragColor = vec4(v_color, 1.0);
            }"#;

            let shader_sources = [
                (glow::VERTEX_SHADER, vertex_shader_source),
                (glow::FRAGMENT_SHADER, fragment_shader_source),
            ];

            let mut shaders = Vec::with_capacity(shader_sources.len());

            for (shader_type, shader_source) in shader_sources.iter() {
                let shader = gl
                    .create_shader(*shader_type)
                    .expect("Cannot create shader");
                gl.shader_source(shader, shader_source);
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    panic!("{}", gl.get_shader_info_log(shader));
                }
                gl.attach_shader(program, shader);
                shaders.push(shader);
            }

            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                panic!("{}", gl.get_program_info_log(program));
            }

            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }

            let rotation_x_location = gl
                .get_uniform_location(program, "rotation_x")
                .expect("Failed to get rotation_x uniform location");
            let rotation_y_location = gl
                .get_uniform_location(program, "rotation_y")
                .expect("Failed to get rotation_y uniform location");
            let scale_location = gl
                .get_uniform_location(program, "scale")
                .expect("Failed to get scale uniform location");

            let position_location =
                gl.get_attrib_location(program, "position")
                    .expect("Failed to get position attribute location") as u32;
            let color_location =
                gl.get_attrib_location(program, "color")
                    .expect("Failed to get color attribute location") as u32;

            let point_cloud = PointCloud::new(&gl);

            let vao = gl
                .create_vertex_array()
                .expect("Cannot create vertex array");
            gl.bind_vertex_array(Some(vao));

            // 设置顶点属性
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(point_cloud.vbo_pos));
            gl.enable_vertex_attrib_array(position_location);
            gl.vertex_attrib_pointer_f32(position_location, 3, glow::FLOAT, false, 0, 0);

            gl.bind_buffer(glow::ARRAY_BUFFER, Some(point_cloud.vbo_color));
            gl.enable_vertex_attrib_array(color_location);
            gl.vertex_attrib_pointer_f32(color_location, 3, glow::FLOAT, false, 0, 0);

            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_vertex_array(None);

            let displayed_texture = DemoTexture::new(&gl, 320, 200);
            let next_texture = DemoTexture::new(&gl, 320, 200);

            Self {
                gl,
                program,
                vao,
                point_cloud,
                rotation_x_location,
                rotation_y_location,
                displayed_texture,
                next_texture,
                start_time: web_time::Instant::now(),
                scale_location,
            }
        }
    }

    fn render(
        &mut self,
        width: u32,
        height: u32,
        rotation_x: f32,
        rotation_y: f32,
        scale: f32,
    ) -> slint::Image {
        unsafe {
            let gl = &self.gl;
            gl.use_program(Some(self.program));

            let _saved_vao = ScopedVAOBinding::new(gl, Some(self.vao));

            if self.next_texture.width != width || self.next_texture.height != height {
                let mut new_texture = DemoTexture::new(gl, width, height);
                std::mem::swap(&mut self.next_texture, &mut new_texture);
            }

            self.next_texture.with_texture_as_active_fbo(|| {
                let mut saved_viewport: [i32; 4] = [0, 0, 0, 0];
                gl.get_parameter_i32_slice(glow::VIEWPORT, &mut saved_viewport);

                gl.viewport(
                    0,
                    0,
                    self.next_texture.width as _,
                    self.next_texture.height as _,
                );

                gl.clear_color(0.0, 0.0, 0.0, 1.0);
                gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

                // 设置旋转角度
                gl.uniform_1_f32(Some(&self.rotation_x_location), rotation_x);
                gl.uniform_1_f32(Some(&self.rotation_y_location), rotation_y);
                gl.uniform_1_f32(Some(&self.scale_location), scale);

                // 绘制点云
                gl.draw_arrays(glow::POINTS, 0, self.point_cloud.num_points);

                gl.viewport(
                    saved_viewport[0],
                    saved_viewport[1],
                    saved_viewport[2],
                    saved_viewport[3],
                );
            });

            gl.use_program(None);
        }

        let result_texture = unsafe {
            slint::BorrowedOpenGLTextureBuilder::new_gl_2d_rgba_texture(
                self.next_texture.texture.0,
                (self.next_texture.width, self.next_texture.height).into(),
            )
            .build()
        };

        std::mem::swap(&mut self.next_texture, &mut self.displayed_texture);

        result_texture
    }
}

fn main() {
    // 在程序启动时设置环境变量
    std::env::set_var("SLINT_BACKEND", "winit");

    let app = App::new().unwrap();

    let mut underlay = None;

    let app_weak = app.as_weak();

    app.window()
        .set_rendering_notifier(move |state, graphics_api| match state {
            slint::RenderingState::RenderingSetup => {
                let context = match graphics_api {
                    slint::GraphicsAPI::NativeOpenGL { get_proc_address } => unsafe {
                        glow::Context::from_loader_function_cstr(|s| get_proc_address(s))
                    },
                    _ => return,
                };
                unsafe {
                    underlay = Some(DemoRenderer::new(context));
                }
            }
            slint::RenderingState::BeforeRendering => {
                if let (Some(underlay), Some(app)) = (underlay.as_mut(), app_weak.upgrade()) {
                    let texture = underlay.render(
                        app.get_requested_texture_width() as u32,
                        app.get_requested_texture_height() as u32,
                        app.get_rotation_x(),
                        app.get_rotation_y(),
                        app.get_point_scale(),
                    );
                    app.set_texture(slint::Image::from(texture));
                    app.window().request_redraw();
                }
            }
            slint::RenderingState::AfterRendering => {}
            slint::RenderingState::RenderingTeardown => {
                drop(underlay.take());
            }
            _ => {}
        })
        .expect("Unable to set rendering notifier");

    app.run().unwrap();
}
