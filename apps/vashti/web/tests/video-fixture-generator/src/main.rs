use kuno_image::video::{Audio,Clip,encode_mp4};
fn main() {
    let directory=std::path::PathBuf::from(std::env::args().nth(1).expect("output directory"));
    std::fs::create_dir_all(&directory).unwrap();
    let (width,height,frames)=(256,256,22);
    let mut rgb=vec![0; width*height*3*frames];
    for frame in 0..frames {for y in 0..height {for x in 0..width {
        let color=if (24+frame*4..72+frame*4).contains(&x)&&(80..144).contains(&y) {[90,220,65]}else if x%32<2||y%32<2 {[90,90,90]}else{[35,35,35]};
        rgb[((frame*height+y)*width+x)*3..((frame*height+y)*width+x)*3+3].copy_from_slice(&color);
    }}}
    let mut clip=Clip{width:width as u32,height:height as u32,fps:24,frames:frames as u32,rgb,audio:None};
    std::fs::write(directory.join("clip-silent.mp4"),encode_mp4(&clip).unwrap()).unwrap();
    clip.audio=Some(Audio{sample_rate:32000,channels:1,samples:(0..frames*32000/24).map(|i|((i as f32)*220.0*std::f32::consts::TAU/32000.0).sin()*0.05).collect()});
    std::fs::write(directory.join("clip.mp4"),encode_mp4(&clip).unwrap()).unwrap();
}
