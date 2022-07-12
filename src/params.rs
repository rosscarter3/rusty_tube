use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    str::FromStr,
};

use crate::argparse::Args;

#[derive(Debug)]
pub enum ParType {
    String(String),
    // Integer(i64),
    Float(f32),
    // Boolean(bool),
}

impl FromStr for ParType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let b = s.chars().all(char::is_numeric);
        if b {
            Ok(ParType::Float(s.parse::<f32>().unwrap()))
        } else {
            Ok(ParType::String(s.to_string()))
        }
    }
}

#[derive(Debug)]
pub struct Params {
    pub params: HashMap<String, ParType>,
}

impl Params {
    pub fn new() -> Result<Params, String> {
        let mut p = Params {
            params: HashMap::new(),
        };
        let arguments = Args::parse().unwrap_or_else(|err| {
            println!("Error: {}", err);
            std::process::exit(1);
        });
        p.read_from_file(&arguments.path)?;
        return Ok(p);
    }

    pub fn read_from_file(&mut self, path: &str) -> Result<(), String> {
        let file = File::open(path).map_err(|err| err.to_string())?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.map_err(|err| err.to_string())?;
            let line = line.trim();
            if line.len() == 0 || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split('=');
            let key = parts.next().unwrap();
            let value = parts.next().unwrap();
            let value = value.trim();
            let value = value.to_string();
            let mut parts = value.split("#");
            let value = parts.next().unwrap().to_string();
            self.params
                .insert(key.to_string(), ParType::from_str(&value).unwrap());
        }
        Ok(())
    }

    pub fn get_string_param(&self, key: &str) -> String {
        match self.params.get(key) {
            Some(&ParType::String(ref value)) => value.clone(),
            _ => "".to_string(),
        }
    }

    pub fn get_float_param(&self, key: &str) -> f32 {
        match self.params.get(key) {
            Some(&ParType::Float(ref value)) => *value,
            _ => 0.0,
        }
    }

    pub fn get_int_param(&self, key: &str) -> i32 {
        match self.params.get(key) {
            Some(&ParType::Float(ref value)) => value.floor() as i32,
            _ => 0,
        }
    }
}
