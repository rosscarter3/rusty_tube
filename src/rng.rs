//! One owned, seedable generator for every stochastic decision.
//!
//! xoshiro256++ seeded through splitmix64. `seed=0` draws a seed from the
//! clock and the process id; either way the seed in use is reported, so any
//! run can be replayed.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    seed: u64,
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// `requested` <= 0 draws a seed from the clock and the pid.
    pub fn new(requested: i64) -> Self {
        let seed = if requested > 0 {
            requested as u64
        } else {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1);
            // Keep it small enough to type back in as `seed=`.
            ((now ^ ((std::process::id() as u64) << 16)) & 0x7FFF_FFFF).max(1)
        };
        let mut x = seed;
        let s = [
            splitmix64(&mut x),
            splitmix64(&mut x),
            splitmix64(&mut x),
            splitmix64(&mut x),
        ];
        Rng { s, seed }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [-1, 1).
    #[inline]
    pub fn signed_unit(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }

    /// Uniform integer in [0, n). 0 for n == 0.
    #[inline]
    pub fn index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((self.unit() * n as f64) as usize).min(n - 1)
    }
}
