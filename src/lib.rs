/// A type that implements `FromUniform` is able to instantiate itself
/// from an `f64` uniformly distributed in the range `[0, 1)`.
///
/// For example, `bool` maps values < 0.5 to `false` and >= 0.5 to
/// `true`, and `u32` maps values evenly in `0..=u32::MAX`.
///
/// The mapping is less clear when it comes to more complicated enums,
/// such as `Option`. In general, this is up to the judgment of the
/// implementer. Reasonable implementations are provided for many common
/// standard library types.
pub trait FromUniform {
    fn from_uniform(uniform_value: f64) -> Self;
}

/// The identity mapping
impl FromUniform for f64 {
    fn from_uniform(uniform_value: f64) -> Self {
        uniform_value
    }
}

/// The identity mapping
impl FromUniform for f32 {
    fn from_uniform(uniform_value: f64) -> Self {
        uniform_value as f32
    }
}

macro_rules! unsigned {
    ($($ut:tt)*) => {
        $(
        /// Uniform in `0 .. = MAX`
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
        /// Uniform in `MIN ..= MAX`
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

/// 50% delegate to `Ok`, 50% to `Err`
impl <T: FromUniform, E: FromUniform> FromUniform for Result<T, E> {
    fn from_uniform(uniform_value: f64) -> Self {
        if uniform_value < 0.5 {
            Ok(T::from_uniform(uniform_value*2.0))
        } else {
            Err(E::from_uniform(uniform_value*2.0 - 1.0))
        }
    }
}

/// 50% delegate to `Some`, 50% to `None`
impl <T: FromUniform> FromUniform for Option<T> {
    fn from_uniform(uniform_value: f64) -> Self {
        if uniform_value < 0.5 {
            Some(T::from_uniform(uniform_value*2.0))
        } else {
            None
        }
    }
}

/// Always returns `()`
impl FromUniform for () {
    fn from_uniform(_: f64) -> Self {}
}

/// 50% false, 50% true
impl FromUniform for bool {
    fn from_uniform(uniform_value: f64) -> Self {
        uniform_value < 0.5
    }
}

/// A helper trait implemented for all tuples up to 32. The user
/// does not need to implement this. It exists because the `Qrng`
/// needs to maintain different state for different cardinality
/// tuples.
pub trait Quasirandom {
    type State;
}

impl<T: FromUniform> Quasirandom for T {
    #[doc(hidden)]
    type State = State<1>;
}

#[doc(hidden)]
pub struct State<const N: usize>([f64; N]);

#[doc(hidden)]
impl<const N: usize> State<N> {
    fn gen(&mut self) -> &[f64; N] {
        for i in 0..N {
            self.0[i] = (self.0[i] + CONSTANTS[N-1][i]).fract();
        }
        &self.0
    }
}

/// Main driver of this library
/// 
/// # QRNG vs PRNG
/// 
/// A `Qrng` is a quasirandom value generator. Rather than generating values
/// that are truly random (or pseudorandom), it produces values that are
/// evenly distributed across the domain of all possible values.
/// 
/// # Uses
/// 
/// `Qrng`s are particularly useful for things like monte-carlo simulations,
/// where evenly covering the domain improves convergence.
/// 
/// # Features
/// 
/// A `Qrng` can be built for any tuple up to size 32 for which all elements
/// implement `FromUniform`.
/// 
/// For instance, a `Qrng<(f64, u32, bool, Option<i16>)>` will generate values of
/// the 5-tuple that, over enough samples, will uniformly cover that space.
/// 
/// # Note
/// 
/// Type inference will typically force you to specify the type at construction time, e.g.
/// `Qrng::<(f64, f64)>::new(seed)`.
/// 
/// # Example usage
/// 
/// ```
/// use quasirandom::Qrng;
/// 
/// fn compute_pi() {
///     let mut qrng = Qrng::<(f64, f64)>::new(0.123);
///     let mut hits = 0.0;
///     let mut total = 0.0;
///     for _ in 0..1_000_000 {
///         let (x, y) = qrng.gen();
///         if x.hypot(y) < 1.0 {
///             hits += 1.0;
///         }
///         total += 1.0;
///     }
///     println!("pi is approximately {}", 4.0 * hits / total);
/// }
/// ```
/// 
/// # Acknowledgments
/// The technique used in this generator is directly taken from 
/// [this blog post by Martin Roberts](http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/).
/// 
#[derive(Debug, Clone, Default)]
pub struct Qrng<T: Quasirandom> {
    state: T::State,
}

impl<T: FromUniform> Qrng<T> {
    pub fn new(seed: f64) -> Self {
        let Qrng { state } = Qrng::<(T,)>::new(seed);
        Self { state }
    }
    
    pub fn gen(&mut self) -> T {
        let [x] = self.state.gen();
        T::from_uniform(*x)
    }

}

macro_rules! define_from_uniform {
    (@inner [$n:expr] [$([$t:tt $x:ident])*]) => {
        impl<$($t: FromUniform,)*> Quasirandom for ($($t,)*) {
            #[doc(hidden)]
            type State = State<{$n}>;
        }
        impl<$($t: FromUniform,)*> Qrng<($($t,)*)> {
            pub fn new(seed: f64) -> Self {
                assert!(seed >= 0.0);
                assert!(seed < 1.0);
                let mut seeds = [0.0; $n];
                for i in 0..$n {
                    seeds[i] = (seed * i as f64).fract();
                }

                Self { state: State(seeds) }
            }
            pub fn gen(&mut self) -> ($($t,)*) {
                let [$($x,)*] = self.state.gen();
                ($($t::from_uniform(*$x),)*)
            }
        }
    };

    (@inner [$n:expr] [$($t:tt)*] $next:tt $($rem:tt)*) => {
        define_from_uniform!(@inner [$n + 1] [[$next x] $($t)*] $($rem)*);
    };

    ($next:tt $($rem:tt)*) => {
        define_from_uniform!($($rem)*);
        define_from_uniform!(@inner [1] [[$next x]] $($rem)*);
    };

    () => {}
}

define_from_uniform!(T31 T30 T29 T28 T27 T26 T25 T24 T23 T22 T21 T20 T19 T18 T17 T16 T15 T14 T13 T12 T11 T10 T9 T8 T7 T6 T5 T4 T3 T2 T1 T0);

/// The binary search finds the unique positive root of x^(d+1) = x + 1, and
/// the magic numbers emitted in the loop are that the inverse of that root
/// exponentiated by increasing integers. See the following blog post by
/// Martin Roberts for a full explanation:
/// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
///
/// Generated by the following snippet:
///
/// ```
/// for d in 1..=32 {
///     let mut lower = 1.0;
///     let mut upper = 2.0;
///     while upper - lower > 1e-14_f64 {
///         let mid = (lower + upper) / 2.0;
///         let y = mid.powi(d+1);
///         if y < mid + 1.0 {
///             lower = mid;
///         } else if y > mid + 1.0 {
///             upper = mid;
///         }
///     }
///     let mut parameters = vec![f64::NAN; 32];
///     for i in 1..=d {
///         parameters[i as usize - 1] = lower.powi(i).recip();
///     }
///     println!("    {:?},", parameters);
/// }
/// ```

use std::f64::NAN;
static CONSTANTS: [[f64; 32]; 32] = [
    [0.6180339887498955, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.7548776662466942, 0.5698402909980553, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.8191725133961674, 0.6710436067037939, 0.5497004779019761, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.8566748838545053, 0.7338918566271301, 0.6287067210378139, 0.5385972572236161, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.8812714616335721, 0.7766393890897725, 0.6844301295853483, 0.6031687406857351, 0.5315553977157988, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.898653712628702, 0.8075784952213495, 0.7257334129697662, 0.6521830259439793, 0.586086697577978, 0.5266889867007452, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9115923534820571, 0.8310006189269559, 0.7575338099526698, 0.6905620286569838, 0.6295110649287636, 0.5738574732214077, 0.5231240845771696, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9215993196339888, 0.849345305949831, 0.7827560560976864, 0.7213874487390121, 0.6648301819503726, 0.6127070433576043, 0.564670394293321, 0.5203998511981808, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9295701282320245, 0.8641006233013023, 0.8032421272075638, 0.7466698871896992, 0.6940820227819199, 0.6451979149209321, 0.5997567085080856, 0.557515920435878, 0.5182501456509743, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9360691110777617, 0.876225380713911, 0.820207513228644, 0.7677709178072383, 0.7186866405431788, 0.6727403647567163, 0.6297314752239486, 0.589472182230569, 0.5517867016256371, 0.5165104872952402, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9414696173216355, 0.8863650403397467, 0.8344857553359374, 0.7856429847364809, 0.7396590001912821, 0.6963664758585899, 0.6556078795422026, 0.6172348994656464, 0.5811079045974802, 0.5470954365639671, 0.5150737313002912, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9460285282856161, 0.8949699763302488, 0.8466671295675179, 0.8009712585325659, 0.757741660908641, 0.7168452282901001, 0.6781560363278498, 0.6415549569952425, 0.606929291780551, 0.5741724246765859, 0.5431834938989744, 0.5138670813222856, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9499283999636238, 0.9023639650574503, 0.857181157511855, 0.8142607254342034, 0.7734893880649323, 0.7347595367933636, 0.6979689511441332, 0.663020528984635, 0.6298220302414098, 0.5982858334490634, 0.5683287044891718, 0.5398715769087982, 0.5128393432388131, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9533025374016683, 0.908785727816459, 0.8663477402818522, 0.8258914990828911, 0.7873244616941877, 0.7505584070914716, 0.7155092339484541, 0.6820967682573851, 0.6502445799332428, 0.6198798079820423, 0.5909329938333397, 0.5633379224556871, 0.5370314708915908, 0.5119534638655036, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9562505576379922, 0.9144151289829711, 0.8744099770025826, 0.8361550281129437, 0.7995737119048135, 0.7645928078816573, 0.7311422989028329, 0.6991552310385577, 0.66856757955614, 0.6393181207692418, 0.6113483094936605, 0.5846021618643568, 0.5590261432791671, 0.5345690612449197, 0.5111819629114723, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9588484010075664, 0.919390256114767, 0.8815558769775812, 0.8452784430387769, 0.8104938835138964, 0.7771407642337125, 0.7451601791432934, 0.7144956462660588, 0.6850930079490782, 0.6569003352134377, 0.629867836040739, 0.6039477674337588, 0.5790943510959491, 0.5552636925808652, 0.5324137037687195, 0.5105040285331529, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9611549719965047, 0.9238188801936017, 0.8879331099223235, 0.8534413234021603, 0.8202893712952632, 0.7884252076963293, 0.7577988084247037, 0.7283620924904307, 0.7000688466109555, 0.6728746526599783, 0.6467368179345593, 0.6216143081309999, 0.5974676829242778, 0.5742590340499008, 0.5519519257909723, 0.5305113377770387, 0.5099036100049179, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9632166633389043, 0.927786340533732, 0.8936592632203137, 0.8607874936809742, 0.8291248575072461, 0.7986268787394741, 0.7692507173921999, 0.7409551092775732, 0.7137003080422573, 0.6874480293364112, 0.6621613970363233, 0.6378048914451547, 0.6143442993990339, 0.5917466662084143, 0.5699802493671892, 0.5490144739645405, 0.5288198897368884, 0.509368129699613, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9650705109167201, 0.9313610910410594, 0.8988291239789491, 0.8674334819051925, 0.8371344735685137, 0.8078937941127652, 0.7796744766508538, 0.7524408455301659, 0.7261584712304062, 0.7007941268368325, 0.6763157460338588, 0.6526923825659189, 0.6298941711143428, 0.6078922895407828, 0.5866589224494579, 0.566167226022151, 0.5463912940814994, 0.5273061253396806, 0.5088875917910817, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9667464397509411, 0.9345986787711201, 0.9035199452979138, 0.8734746903607233, 0.8444285471187851, 0.8163482915511453, 0.789201804453833, 0.7629580347007615, 0.7375869637263363, 0.7130595711891421, 0.6893478017774359, 0.6664245331184737, 0.6442635447549676, 0.6228394881531861, 0.6021278577083912, 0.5821049627144484, 0.5627479002655473, 0.5440345290590357, 0.5259434440694026, 0.5084539520644433, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.968268892614234, 0.9375446484043949, 0.9077953184869247, 0.8789899677517204, 0.8510986426939796, 0.8240923402667772, 0.7979429777219849, 0.7726233634081705, 0.7481071684951142, 0.7243688995955343, 0.7013838722555592, 0.6791281852863735, 0.6575786959103513, 0.636712995695828, 0.6165093872554908, 0.5969468616841539, 0.578005076712458, 0.559664335553777, 0.5419055664223368, 0.5247103027012452, 0.5080606637398142, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9696580306006657, 0.9402366963083615, 0.911708063240842, 0.884045045084862, 0.8572213773792641, 0.8312115925783671, 0.8059909958719824, 0.7815356417390957, 0.757822311212959, 0.7348284898360026, 0.7125323462836395, 0.6909127116366655, 0.6699490592825748, 0.6496214854267098, 0.6299106901947427, 0.6107979593085402, 0.5922651463180246, 0.5742946553721509, 0.5568694245126478, 0.5399729094746601, 0.5235890679789106, 0.5077023445004684, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9709306314075442, 0.9427062910054524, 0.915302414357788, 0.8886951511012567, 0.862861344187566, 0.8377785097291959, 0.8134248174310394, 0.7897790715908856, 0.7668206926522025, 0.7445296992931734, 0.7228866910363898, 0.7018728313640723, 0.6814698313241194, 0.66165993361272, 0.6424258971196719, 0.6237509819229611, 0.6056189347195363, 0.5880139746796036, 0.5709207797121273, 0.5543244731295832, 0.5382106107003604, 0.5225651680775409, 0.5073745285931162, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9721007705580277, 0.944979908119511, 0.9186156968448309, 0.8929870267495597, 0.868073376801569, 0.8438547984897142, 0.8203118998509404, 0.7974258299430187, 0.7751782637504832, 0.7535513875116787, 0.7325278844551737, 0.7120909209341162, 0.6922241329474299, 0.6729116130370592, 0.6541378975507706, 0.6358879542603122, 0.6181471703250173, 0.6009013405912137, 0.5841366562180706, 0.5678396936207761, 0.551997403722191, 0.5365971015043723, 0.5216264558516047, 0.5070734796767978, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9731803443915577, 0.9470799827100708, 0.9216796237401371, 0.8969604936501081, 0.8729043221160337, 0.8494933288177609, 0.8267102102971994, 0.8045381271690456, 0.7829606916745105, 0.7619619555688525, 0.7415263983337606, 0.7216389157058806, 0.7022848085129991, 0.6834497718096395, 0.6651198843040366, 0.6472815980686754, 0.6299217285267913, 0.6130274447074281, 0.5965862597618514, 0.58058602173431, 0.5650149045803201, 0.549861399425839, 0.535114306060862, 0.520762724661159, 0.5067960477320326, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.9741794761063388, 0.9490256516668206, 0.92452131215226, 0.9006496875216334, 0.8773944407451627, 0.8547396566237366, 0.8326698308970235, 0.811169859632816, 0.7902250288903491, 0.7698210046505166, 0.7499438230060956, 0.7305798806052629, 0.7117159253418667, 0.6933390472860778, 0.6754366698492191, 0.6579965411767223, 0.6410067257633224, 0.6244555962847529, 0.6083318256403519, 0.5926243792011306, 0.5773225072580017, 0.562415737664998, 0.5478938686724477, 0.5337469619452003, 0.5199653357611251, 0.5065395583852295, NAN, NAN, NAN, NAN, NAN, NAN],
    [0.975106834269357, 0.9508333382388071, 0.9271640863678078, 0.9040840371063537, 0.8815785233362363, 0.8596332430502518, 0.8382342502634316, 0.8173679461505227, 0.7970210704040824, 0.777180692807699, 0.7578342050189809, 0.7389693125570932, 0.7205740269897503, 0.7026366583146973, 0.6851458075308443, 0.6680903593943238, 0.651459475354876, 0.6352425866680693, 0.6194293876789786, 0.604009829273055, 0.5889741124900234, 0.5743126822967508, 0.5600162215151275, 0.5460756449011029, 0.5324820933711121, 0.5192269283722252, 0.5063017263924426, NAN, NAN, NAN, NAN, NAN],
    [0.9759698850464326, 0.9525172165175468, 0.929628118309378, 0.9072890477623351, 0.8854867876484934, 0.8642084383514349, 0.843441410234007, 0.8231734161894848, 0.8033924643717306, 0.7840868511000482, 0.7652451539345332, 0.7468562249178259, 0.7289091839792632, 0.7113934124975304, 0.6942985470180042, 0.6776144731210667, 0.6613313194377665, 0.6454394518092825, 0.6299294675867378, 0.614792190067989, 0.6000186630680998, 0.5856001456202875, 0.5715281068042061, 0.5577942206985063, 0.5443903614546857, 0.5313085984893154, 0.5185411917917983, 0.5060805873448815, NAN, NAN, NAN, NAN],
    [0.9767750937050804, 0.9540895836825685, 0.931930942504582, 0.9102869336915768, 0.8891456049551002, 0.8684952815974784, 0.8483245600647971, 0.8286223016496131, 0.8093776263399202, 0.790579906810971, 0.7722187625566399, 0.7542840541570831, 0.7367658776795328, 0.7196545592091314, 0.7029406495067876, 0.6866149187911024, 0.6706683516414852, 0.6550921420196435, 0.639877688406699, 0.6250165890532436, 0.6105006373397118, 0.5963218172445082, 0.5824722989173883, 0.5689444343556455, 0.5557307531807195, 0.5428239585128922, 0.5302169229417929, 0.517902684590489, 0.5058744432709877, NAN, NAN, NAN],
    [0.9775280869070946, 0.9555611606922443, 0.9340878733342124, 0.9130971318235092, 0.8925780924317901, 0.8725201551100316, 0.8529129580125907, 0.8337463721443189, 0.8150104961279666, 0.7966956510891733, 0.7787923756564016, 0.7612914210732337, 0.7441837464205016, 0.7274605139457871, 0.7111130844978774, 0.6951330130638131, 0.6795120444062337, 0.6642421087987542, 0.6493153178571804, 0.6347239604644017, 0.6204604987868609, 0.6065175643805418, 0.5928879543844617, 0.5795646277997036, 0.5665407018520666, 0.5538094484364534, 0.5413642906411595, 0.529198799350269, 0.5173066899223998, 0.5056818189440851, NAN, NAN],
    [0.9782337844131518, 0.9569413369672767, 0.9361123455228803, 0.9157367223967191, 0.8958045994762384, 0.8763063234403483, 0.8572324510842273, 0.8385737447458858, 0.8203211678322762, 0.8024658804427837, 0.7849992350879762, 0.7679127725015403, 0.7511982175433775, 0.7348474751918722, 0.7188526266233948, 0.7032059253771379, 0.68789979360343, 0.6729268183937093, 0.6582797481903799, 0.6439514892748122, 0.6299351023317845, 0.6162237990887076, 0.6028109390279961, 0.5896900261710024, 0.5768547059319501, 0.5642987620403473, 0.5520161135303857, 0.5400008117958692, 0.5282470377092473, 0.5167490988033538, 0.5055014265144905, NAN],
    [0.97889650672095, 0.9582383708704787, 0.9380161938510856, 0.9182207754085091, 0.8988431094459914, 0.8798743799268776, 0.8613059568636822, 0.8431293923918037, 0.8253364169260937, 0.8079189353985387, 0.7908690235753382, 0.7741789244517071, 0.7578410447227583, 0.7418479513288633, 0.7261923680739174, 0.7108671723149721, 0.6958653917217258, 0.6811802011044027, 0.666804919308574, 0.652733006175508, 0.638958059566669, 0.625473812451009, 0.6122741300537271, 0.599353007065202, 0.5867045649088232, 0.5743230490664818, 0.5622028264605037, 0.5503383828908314, 0.5387243205262915, 0.5273553554488041, 0.5162263152494191, 0.5053321366750843],
];

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaChaRng;
    use std::collections::HashSet;

    // Test that the QRNG covers a variety of ranges more effectively than a CPRNG
    #[test]
    fn coverage() {
        for &n in &[100, 1_000, 100_000] {
            let mut qrng = Qrng::<f64>::new(0.0);
            let mut qrng_set = HashSet::new();
            let mut rng = ChaChaRng::seed_from_u64(0x5c329d7775ca89e6);
            let mut rng_set = HashSet::new();
            for _ in 0..n {
                qrng_set.insert((qrng.gen() * n as f64) as u32);
                rng_set.insert((rng.gen::<f64>() * n as f64) as u32);
            }
            assert!(qrng_set.len() > rng_set.len())
        }
    }

    // Test that the QRNG has at least a 3x lower standard deviation of minimum distance between 3D points than a CPRNG
    #[test]
    fn distance() {
        let mut qrng = Qrng::<(f64, f64, f64)>::new(0.0);
        let mut qrng_points = vec![];
        let mut rng = ChaChaRng::from_seed([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]);
        let mut rng_points = vec![];
        let n = 1000;
        for _ in 0..n {
            qrng_points.push(qrng.gen());
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
