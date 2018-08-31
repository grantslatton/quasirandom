#[cfg(test)]
extern crate rand;

/// A type that implements `FromUniform` is able to instantiate itself
/// an `f64` uniformly distributed in the range [0, 1).
pub trait FromUniform {
    fn from_uniform(uniform_value: f64) -> Self;
}

impl FromUniform for f64 {
    fn from_uniform(uniform_value: f64) -> Self {
        uniform_value
    }
}

impl FromUniform for f32 {
    fn from_uniform(uniform_value: f64) -> Self {
        uniform_value as f32
    }
}

macro_rules! unsigned {
    ($($ut:tt)*) => {
        $(
        impl FromUniform for $ut {
            fn from_uniform(uniform_value: f64) -> Self {
                (::std::$ut::MAX as f64 * uniform_value) as $ut
            }
        }
        )*
    }
}
unsigned!(u8 u16 u32 u64 u128 usize);

macro_rules! signed {
    ($($st:tt)*) => {
        $(
        impl FromUniform for $st {
            fn from_uniform(uniform_value: f64) -> Self {
                let min = ::std::$st::MIN as f64;
                let result = (::std::$st::MAX as f64 - min + 1.0) * uniform_value + min;
                result as $st
            }
        }
        )*
    }
}
signed!(i8 i16 i32 i64 i128 isize);

impl <T: FromUniform, E: FromUniform> FromUniform for Result<T, E> {
    fn from_uniform(uniform_value: f64) -> Self {
        if uniform_value < 0.5 {
            Ok(T::from_uniform(uniform_value*2.0))
        } else {
            Err(E::from_uniform(uniform_value*2.0 - 1.0))
        }
    }
}

impl <T: FromUniform> FromUniform for Option<T> {
    fn from_uniform(uniform_value: f64) -> Self {
        if uniform_value < 0.5 {
            Some(T::from_uniform(uniform_value*2.0))
        } else {
            None
        }
    }
}

impl FromUniform for () {
    fn from_uniform(_: f64) -> Self {
        ()
    }
}

impl FromUniform for bool {
    fn from_uniform(uniform_value: f64) -> Self {
        uniform_value < 0.5
    }
}

/// A `Qrng` is a Quasirandom Number Generator. It generates values in up to 16 dimensions
/// that are relatively evenly distributed over the space, as opposed to true random or
/// pseudorandom values that will form clumps and gaps.
#[derive(Debug, Clone)]
pub struct Qrng(f64);

impl Qrng {

    pub fn new(seed: u32) -> Self {
        Qrng(seed as f64)
    }

    /// Generate a quasirandom value in [0, 1)
    pub fn next(&mut self) -> f64 {
        self.next1()
    }

    /// Generate a quasirandom value in [0, 1)
    pub fn next1(&mut self) -> f64 {
        let result = (self.0 * SEQ1[0]).fract();
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^2
    pub fn next2(&mut self) -> (f64, f64) {
        let result = (
            (self.0 * SEQ2[0]).fract(),
            (self.0 * SEQ2[1]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^3
    pub fn next3(&mut self) -> (f64, f64, f64) {
        let result = (
            (self.0 * SEQ3[0]).fract(),
            (self.0 * SEQ3[1]).fract(),
            (self.0 * SEQ3[2]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^4
    pub fn next4(&mut self) -> (f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ4[0]).fract(),
            (self.0 * SEQ4[1]).fract(),
            (self.0 * SEQ4[2]).fract(),
            (self.0 * SEQ4[3]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^5
    pub fn next5(&mut self) -> (f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ5[0]).fract(),
            (self.0 * SEQ5[1]).fract(),
            (self.0 * SEQ5[2]).fract(),
            (self.0 * SEQ5[3]).fract(),
            (self.0 * SEQ5[4]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^6
    pub fn next6(&mut self) -> (f64, f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ6[0]).fract(),
            (self.0 * SEQ6[1]).fract(),
            (self.0 * SEQ6[2]).fract(),
            (self.0 * SEQ6[3]).fract(),
            (self.0 * SEQ6[4]).fract(),
            (self.0 * SEQ6[5]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^7
    pub fn next7(&mut self) -> (f64, f64, f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ7[0]).fract(),
            (self.0 * SEQ7[1]).fract(),
            (self.0 * SEQ7[2]).fract(),
            (self.0 * SEQ7[3]).fract(),
            (self.0 * SEQ7[4]).fract(),
            (self.0 * SEQ7[5]).fract(),
            (self.0 * SEQ7[6]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^8
    pub fn next8(&mut self) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ8[0]).fract(),
            (self.0 * SEQ8[1]).fract(),
            (self.0 * SEQ8[2]).fract(),
            (self.0 * SEQ8[3]).fract(),
            (self.0 * SEQ8[4]).fract(),
            (self.0 * SEQ8[5]).fract(),
            (self.0 * SEQ8[6]).fract(),
            (self.0 * SEQ8[7]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^9
    pub fn next9(&mut self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ9[0]).fract(),
            (self.0 * SEQ9[1]).fract(),
            (self.0 * SEQ9[2]).fract(),
            (self.0 * SEQ9[3]).fract(),
            (self.0 * SEQ9[4]).fract(),
            (self.0 * SEQ9[5]).fract(),
            (self.0 * SEQ9[6]).fract(),
            (self.0 * SEQ9[7]).fract(),
            (self.0 * SEQ9[8]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^10
    pub fn next10(&mut self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ10[0]).fract(),
            (self.0 * SEQ10[1]).fract(),
            (self.0 * SEQ10[2]).fract(),
            (self.0 * SEQ10[3]).fract(),
            (self.0 * SEQ10[4]).fract(),
            (self.0 * SEQ10[5]).fract(),
            (self.0 * SEQ10[6]).fract(),
            (self.0 * SEQ10[7]).fract(),
            (self.0 * SEQ10[8]).fract(),
            (self.0 * SEQ10[9]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^11
    pub fn next11(&mut self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ11[0]).fract(),
            (self.0 * SEQ11[1]).fract(),
            (self.0 * SEQ11[2]).fract(),
            (self.0 * SEQ11[3]).fract(),
            (self.0 * SEQ11[4]).fract(),
            (self.0 * SEQ11[5]).fract(),
            (self.0 * SEQ11[6]).fract(),
            (self.0 * SEQ11[7]).fract(),
            (self.0 * SEQ11[8]).fract(),
            (self.0 * SEQ11[9]).fract(),
            (self.0 * SEQ11[10]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^12
    pub fn next12(&mut self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ12[0]).fract(),
            (self.0 * SEQ12[1]).fract(),
            (self.0 * SEQ12[2]).fract(),
            (self.0 * SEQ12[3]).fract(),
            (self.0 * SEQ12[4]).fract(),
            (self.0 * SEQ12[5]).fract(),
            (self.0 * SEQ12[6]).fract(),
            (self.0 * SEQ12[7]).fract(),
            (self.0 * SEQ12[8]).fract(),
            (self.0 * SEQ12[9]).fract(),
            (self.0 * SEQ12[10]).fract(),
            (self.0 * SEQ12[11]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^13
    pub fn next13(&mut self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ13[0]).fract(),
            (self.0 * SEQ13[1]).fract(),
            (self.0 * SEQ13[2]).fract(),
            (self.0 * SEQ13[3]).fract(),
            (self.0 * SEQ13[4]).fract(),
            (self.0 * SEQ13[5]).fract(),
            (self.0 * SEQ13[6]).fract(),
            (self.0 * SEQ13[7]).fract(),
            (self.0 * SEQ13[8]).fract(),
            (self.0 * SEQ13[9]).fract(),
            (self.0 * SEQ13[10]).fract(),
            (self.0 * SEQ13[11]).fract(),
            (self.0 * SEQ13[12]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^14
    pub fn next14(&mut self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ14[0]).fract(),
            (self.0 * SEQ14[1]).fract(),
            (self.0 * SEQ14[2]).fract(),
            (self.0 * SEQ14[3]).fract(),
            (self.0 * SEQ14[4]).fract(),
            (self.0 * SEQ14[5]).fract(),
            (self.0 * SEQ14[6]).fract(),
            (self.0 * SEQ14[7]).fract(),
            (self.0 * SEQ14[8]).fract(),
            (self.0 * SEQ14[9]).fract(),
            (self.0 * SEQ14[10]).fract(),
            (self.0 * SEQ14[11]).fract(),
            (self.0 * SEQ14[12]).fract(),
            (self.0 * SEQ14[13]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^15
    pub fn next15(&mut self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ15[0]).fract(),
            (self.0 * SEQ15[1]).fract(),
            (self.0 * SEQ15[2]).fract(),
            (self.0 * SEQ15[3]).fract(),
            (self.0 * SEQ15[4]).fract(),
            (self.0 * SEQ15[5]).fract(),
            (self.0 * SEQ15[6]).fract(),
            (self.0 * SEQ15[7]).fract(),
            (self.0 * SEQ15[8]).fract(),
            (self.0 * SEQ15[9]).fract(),
            (self.0 * SEQ15[10]).fract(),
            (self.0 * SEQ15[11]).fract(),
            (self.0 * SEQ15[12]).fract(),
            (self.0 * SEQ15[13]).fract(),
            (self.0 * SEQ15[14]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value in [0, 1)^16
    pub fn next16(&mut self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let result = (
            (self.0 * SEQ16[0]).fract(),
            (self.0 * SEQ16[1]).fract(),
            (self.0 * SEQ16[2]).fract(),
            (self.0 * SEQ16[3]).fract(),
            (self.0 * SEQ16[4]).fract(),
            (self.0 * SEQ16[5]).fract(),
            (self.0 * SEQ16[6]).fract(),
            (self.0 * SEQ16[7]).fract(),
            (self.0 * SEQ16[8]).fract(),
            (self.0 * SEQ16[9]).fract(),
            (self.0 * SEQ16[10]).fract(),
            (self.0 * SEQ16[11]).fract(),
            (self.0 * SEQ16[12]).fract(),
            (self.0 * SEQ16[13]).fract(),
            (self.0 * SEQ16[14]).fract(),
            (self.0 * SEQ16[15]).fract(),
        );
        self.0 += 1.0;
        result
    }

    /// Generate a quasirandom value
    pub fn gen<T: FromUniform>(&mut self) -> T {
        self.gen1()
    }

    /// Generate a quasirandom value
    pub fn gen1<T: FromUniform>(&mut self) -> T {
        let x = self.next1();
        T::from_uniform(x)
    }

    /// Generate a quasirandom 2-tuple
    pub fn gen2<T0, T1>(&mut self) -> (T0, T1) where
        T0: FromUniform,
        T1: FromUniform,
    {
        let data = self.next2();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
        )
    }

    /// Generate a quasirandom 3-tuple
    pub fn gen3<T0, T1, T2>(&mut self) -> (T0, T1, T2) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
    {
        let data = self.next3();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
        )
    }

    /// Generate a quasirandom 4-tuple
    pub fn gen4<T0, T1, T2, T3>(&mut self) -> (T0, T1, T2, T3) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
    {
        let data = self.next4();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
        )
    }

    /// Generate a quasirandom 5-tuple
    pub fn gen5<T0, T1, T2, T3, T4>(&mut self) -> (T0, T1, T2, T3, T4) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
    {
        let data = self.next5();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
        )
    }

    /// Generate a quasirandom 6-tuple
    pub fn gen6<T0, T1, T2, T3, T4, T5>(&mut self) -> (T0, T1, T2, T3, T4, T5) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
        T5: FromUniform,
    {
        let data = self.next6();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
            T5::from_uniform(data.5),
        )
    }

    /// Generate a quasirandom 7-tuple
    pub fn gen7<T0, T1, T2, T3, T4, T5, T6>(&mut self) -> (T0, T1, T2, T3, T4, T5, T6) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
        T5: FromUniform,
        T6: FromUniform,
    {
        let data = self.next7();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
            T5::from_uniform(data.5),
            T6::from_uniform(data.6),
        )
    }

    /// Generate a quasirandom 8-tuple
    pub fn gen8<T0, T1, T2, T3, T4, T5, T6, T7>(&mut self) -> (T0, T1, T2, T3, T4, T5, T6, T7) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
        T5: FromUniform,
        T6: FromUniform,
        T7: FromUniform,
    {
        let data = self.next8();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
            T5::from_uniform(data.5),
            T6::from_uniform(data.6),
            T7::from_uniform(data.7),
        )
    }

    /// Generate a quasirandom 9-tuple
    pub fn gen9<T0, T1, T2, T3, T4, T5, T6, T7, T8>(&mut self) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
        T5: FromUniform,
        T6: FromUniform,
        T7: FromUniform,
        T8: FromUniform,
    {
        let data = self.next9();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
            T5::from_uniform(data.5),
            T6::from_uniform(data.6),
            T7::from_uniform(data.7),
            T8::from_uniform(data.8),
        )
    }

    /// Generate a quasirandom 10-tuple
    pub fn gen10<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(&mut self) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
        T5: FromUniform,
        T6: FromUniform,
        T7: FromUniform,
        T8: FromUniform,
        T9: FromUniform,
    {
        let data = self.next10();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
            T5::from_uniform(data.5),
            T6::from_uniform(data.6),
            T7::from_uniform(data.7),
            T8::from_uniform(data.8),
            T9::from_uniform(data.9),
        )
    }
   
    /// Generate a quasirandom 11-tuple
    pub fn gen11<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(&mut self) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
        T5: FromUniform,
        T6: FromUniform,
        T7: FromUniform,
        T8: FromUniform,
        T9: FromUniform,
        T10: FromUniform,
    {
        let data = self.next11();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
            T5::from_uniform(data.5),
            T6::from_uniform(data.6),
            T7::from_uniform(data.7),
            T8::from_uniform(data.8),
            T9::from_uniform(data.9),
            T10::from_uniform(data.10),
        )
    }

    /// Generate a quasirandom 12-tuple
    pub fn gen12<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(&mut self) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
        T5: FromUniform,
        T6: FromUniform,
        T7: FromUniform,
        T8: FromUniform,
        T9: FromUniform,
        T10: FromUniform,
        T11: FromUniform,
    {
        let data = self.next12();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
            T5::from_uniform(data.5),
            T6::from_uniform(data.6),
            T7::from_uniform(data.7),
            T8::from_uniform(data.8),
            T9::from_uniform(data.9),
            T10::from_uniform(data.10),
            T11::from_uniform(data.11),
        )
    }

    /// Generate a quasirandom 13-tuple
    pub fn gen13<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(&mut self) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
        T5: FromUniform,
        T6: FromUniform,
        T7: FromUniform,
        T8: FromUniform,
        T9: FromUniform,
        T10: FromUniform,
        T11: FromUniform,
        T12: FromUniform,
    {
        let data = self.next13();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
            T5::from_uniform(data.5),
            T6::from_uniform(data.6),
            T7::from_uniform(data.7),
            T8::from_uniform(data.8),
            T9::from_uniform(data.9),
            T10::from_uniform(data.10),
            T11::from_uniform(data.11),
            T12::from_uniform(data.12),
        )
    }

    /// Generate a quasirandom 14-tuple
    pub fn gen14<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>(&mut self) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
        T5: FromUniform,
        T6: FromUniform,
        T7: FromUniform,
        T8: FromUniform,
        T9: FromUniform,
        T10: FromUniform,
        T11: FromUniform,
        T12: FromUniform,
        T13: FromUniform,
    {
        let data = self.next14();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
            T5::from_uniform(data.5),
            T6::from_uniform(data.6),
            T7::from_uniform(data.7),
            T8::from_uniform(data.8),
            T9::from_uniform(data.9),
            T10::from_uniform(data.10),
            T11::from_uniform(data.11),
            T12::from_uniform(data.12),
            T13::from_uniform(data.13),
        )
    }

    /// Generate a quasirandom 15-tuple
    pub fn gen15<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>(&mut self) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
        T5: FromUniform,
        T6: FromUniform,
        T7: FromUniform,
        T8: FromUniform,
        T9: FromUniform,
        T10: FromUniform,
        T11: FromUniform,
        T12: FromUniform,
        T13: FromUniform,
        T14: FromUniform,
    {
        let data = self.next15();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
            T5::from_uniform(data.5),
            T6::from_uniform(data.6),
            T7::from_uniform(data.7),
            T8::from_uniform(data.8),
            T9::from_uniform(data.9),
            T10::from_uniform(data.10),
            T11::from_uniform(data.11),
            T12::from_uniform(data.12),
            T13::from_uniform(data.13),
            T14::from_uniform(data.14),
        )
    }

    /// Generate a quasirandom 16-tuple
    pub fn gen16<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>(&mut self) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15) where
        T0: FromUniform,
        T1: FromUniform,
        T2: FromUniform,
        T3: FromUniform,
        T4: FromUniform,
        T5: FromUniform,
        T6: FromUniform,
        T7: FromUniform,
        T8: FromUniform,
        T9: FromUniform,
        T10: FromUniform,
        T11: FromUniform,
        T12: FromUniform,
        T13: FromUniform,
        T14: FromUniform,
        T15: FromUniform,
    {
        let data = self.next16();
        (
            T0::from_uniform(data.0), 
            T1::from_uniform(data.1),
            T2::from_uniform(data.2),
            T3::from_uniform(data.3),
            T4::from_uniform(data.4),
            T5::from_uniform(data.5),
            T6::from_uniform(data.6),
            T7::from_uniform(data.7),
            T8::from_uniform(data.8),
            T9::from_uniform(data.9),
            T10::from_uniform(data.10),
            T11::from_uniform(data.11),
            T12::from_uniform(data.12),
            T13::from_uniform(data.13),
            T14::from_uniform(data.14),
            T15::from_uniform(data.15),
        )
    }


}

// Each sequence SEQD is (1/g_d^1, 1/g_d^2, 1/g_d^3, ..., 1/g_d^d) where g_d is the generalized golden ratio which
// is defined as the unique positive root of x^(d+1) = x + 1. This idea is directly taken from Martin Roberts' blog post:
// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
static SEQ1: [f64; 1] = [0.6180339887498955];
static SEQ2: [f64; 2] = [0.7548776662466942, 0.5698402909980553];
static SEQ3: [f64; 3] = [0.8191725133961674, 0.6710436067037939, 0.5497004779019762];
static SEQ4: [f64; 4] = [0.8566748838545053, 0.7338918566271301, 0.6287067210378139, 0.5385972572236161];
static SEQ5: [f64; 5] = [0.8812714616335721, 0.7766393890897725, 0.6844301295853482, 0.603168740685735, 0.5315553977157986];
static SEQ6: [f64; 6] = [0.898653712628702, 0.8075784952213495, 0.7257334129697662, 0.6521830259439794, 0.586086697577978, 0.5266889867007453];
static SEQ7: [f64; 7] = [0.9115923534820571, 0.8310006189269559, 0.7575338099526698, 0.6905620286569838, 0.6295110649287636, 0.5738574732214077, 0.5231240845771696];
static SEQ8: [f64; 8] = [0.9215993196339888, 0.849345305949831, 0.7827560560976863, 0.721387448739012, 0.6648301819503725, 0.6127070433576042, 0.5646703942933209, 0.5203998511981808];
static SEQ9: [f64; 9] = [0.9295701282320245, 0.8641006233013023, 0.8032421272075638, 0.7466698871896993, 0.69408202278192, 0.6451979149209323, 0.5997567085080858, 0.5575159204358782, 0.5182501456509744];
static SEQ10: [f64; 10] = [0.9360691110777617, 0.876225380713911, 0.820207513228644, 0.7677709178072383, 0.7186866405431788, 0.6727403647567162, 0.6297314752239486, 0.5894721822305691, 0.5517867016256371, 0.5165104872952403];
static SEQ11: [f64; 11] = [0.9414696173216355, 0.8863650403397467, 0.8344857553359374, 0.7856429847364809, 0.7396590001912821, 0.6963664758585899, 0.6556078795422025, 0.6172348994656462, 0.5811079045974801, 0.5470954365639671, 0.5150737313002911];
static SEQ12: [f64; 12] = [0.9460285282856161, 0.8949699763302488, 0.8466671295675179, 0.800971258532566, 0.7577416609086413, 0.7168452282901002, 0.67815603632785, 0.6415549569952428, 0.6069292917805512, 0.5741724246765862, 0.5431834938989747, 0.5138670813222859];
static SEQ13: [f64; 13] = [0.9499283999636238, 0.9023639650574503, 0.857181157511855, 0.8142607254342034, 0.7734893880649323, 0.7347595367933636, 0.6979689511441332, 0.6630205289846351, 0.6298220302414098, 0.5982858334490635, 0.5683287044891719, 0.5398715769087983, 0.5128393432388132];
static SEQ14: [f64; 14] = [0.9533025374016683, 0.908785727816459, 0.8663477402818522, 0.8258914990828911, 0.7873244616941877, 0.7505584070914716, 0.7155092339484541, 0.6820967682573852, 0.6502445799332429, 0.6198798079820423, 0.5909329938333397, 0.5633379224556871, 0.5370314708915909, 0.5119534638655037];
static SEQ15: [f64; 15] = [0.9562505576379922, 0.9144151289829711, 0.8744099770025826, 0.8361550281129435, 0.7995737119048133, 0.764592807881657, 0.7311422989028328, 0.6991552310385574, 0.6685675795561398, 0.6393181207692413, 0.6113483094936603, 0.5846021618643565, 0.5590261432791667, 0.5345690612449192, 0.511181962911472];
static SEQ16: [f64; 16] = [0.9588484010075664, 0.919390256114767, 0.8815558769775812, 0.8452784430387766, 0.8104938835138963, 0.7771407642337123, 0.7451601791432932, 0.7144956462660584, 0.6850930079490779, 0.6569003352134374, 0.6298678360407388, 0.6039477674337584, 0.5790943510959489, 0.5552636925808648, 0.5324137037687192, 0.5105040285331525];

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, ChaChaRng, SeedableRng};
    use std::collections::HashSet;

    // Test that the QRNG covers a variety of ranges more effectively than a CPRNG
    #[test]
    fn coverage() {
        for &n in &[100, 1_000, 100_000] {
            let mut qrng = Qrng::new(0);
            let mut qrng_set = HashSet::new();
            let mut rng = ChaChaRng::from_seed([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]);
            let mut rng_set = HashSet::new();
            for _ in 0..n {
                qrng_set.insert((qrng.next1() * n as f64) as u32);
                rng_set.insert((rng.gen::<f64>() * n as f64) as u32);
            }
            assert!(qrng_set.len() > rng_set.len())
        }
    }

    // Test that the QRNG has at least a 3x lower standard deviation of minimum distance between 3D points than a CPRNG
    #[test]
    fn distance() {
        let mut qrng = Qrng::new(0);
        let mut qrng_points = vec![];
        let mut rng = ChaChaRng::from_seed([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]);
        let mut rng_points = vec![];
        let n = 1000;
        for _ in 0..n {
            qrng_points.push(qrng.next3());
            rng_points.push((rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>()));
        }

        let d = |p1: (f64, f64, f64), p2: (f64, f64, f64)| {
            ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2) + (p1.2 - p2.2).powi(2)).sqrt()
        };

        let mut qrng_distances = vec![];
        let mut rng_distances = vec![];

        for i in 0..qrng_points.len() {
            let mut qrng_closest = ::std::f64::INFINITY;
            let mut rng_closest = ::std::f64::INFINITY;
            for j in 0..qrng_points.len() {
                if i != j {
                    qrng_closest = qrng_closest.min(d(qrng_points[i], qrng_points[j]));
                    rng_closest = rng_closest.min(d(rng_points[i], rng_points[j]));
                }
            }
            qrng_distances.push(qrng_closest);
            rng_distances.push(rng_closest);
        }

        let qrng_distance_sum = qrng_distances.iter().sum::<f64>();
        let rng_distance_sum = rng_distances.iter().sum::<f64>();
        let qrng_distance_mean = qrng_distance_sum / n as f64;
        let rng_distance_mean = rng_distance_sum / n as f64;

        let standard_deviation = |mean: f64, xs: Vec<f64>| {
            let sum_squared_error = xs
                .into_iter()
                .map(|x| (x-mean).powi(2))
                .sum::<f64>();
            (sum_squared_error / n as f64).sqrt()
        };

        let qrng_standard_deviation = standard_deviation(qrng_distance_mean, qrng_distances);
        let rng_standard_deviation = standard_deviation(rng_distance_mean, rng_distances);
        assert!(qrng_standard_deviation < rng_standard_deviation / 3.0);
    }
}
