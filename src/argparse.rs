// Handle the arguments

use std::env;

#[derive(Debug)]
pub struct Args {
    pub path: String,
}

impl Args {
    pub fn parse() -> Result<Args, String> {
        let args: Vec<String> = env::args().collect();

        if args.len() < 2 {
            return Err("Not enough arguments".to_string());
        }

        Ok(Args {
            path: args[1].clone(),
        })
    }
}
