// use std::collections::HashMap;

use indicatif::{ProgressBar, ProgressStyle};
use rusty_tube::params::Params;
use std::thread;
use std::time::Duration;
// use serde_any;

fn main() -> Result<(), String> {
    let params = Params::new().unwrap();

    // Element Pool

    // Microtubule pool

    // run loop

    // evlolve microtubule pool
    // println!("{:?}", params);
    // println!("{:?}", params.get_int_param("nb_max_steps"));

    let pb = ProgressBar::new(params.get_int_param("nb_max_steps") as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} {elapsed_precise} {bar:40} {pos}/{len} ({eta})"),
    );

    for t in 0..params.get_int_param("nb_max_steps") {
        pb.set_position(t as u64);
        thread::sleep(Duration::from_millis(10));
    }
    // run loop end
    pb.finish_with_message("Done!");
    Ok(())
}
