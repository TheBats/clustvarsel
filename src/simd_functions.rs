/*

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch="aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn dot_product_f32_x86_64(a: Vec<f32>, b: Vec<f32>) -> Result<f32, &'static str> {

    let mut index: usize = 0;
    let mut dp_mm = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    if b.len() != a.len() {
        let msg = "A and B do not have the same length: A {:?} B {:?}";
        return Err(msg);
    }

    while index + 8 <= a.len() {

        let c = _mm256_set_ps(a[index], a[index+1], a[index+2], a[index+3], a[index+4], a[index+5], a[index+6], a[index+7]);
        let d = _mm256_set_ps(b[index], b[index+1], b[index+2], b[index+3], b[index+4], b[index+5], b[index+6], b[index+7]);

        dp_mm = _mm256_fmadd_ps(c, d, dp_mm);

        index += 8
    }

    let mut dp = reduce_vec_f32(dp_mm);

    for i in index..a.len() {
        dp += a[i]*b[i];
    }

    Ok(dp)
}

#[cfg(target_arch = "x86_64")]
unsafe fn reduce_vec_f32(a: __m256) -> f32{
    // hiQuad = ( x7, x6, x5, x4 )
    let hi_quad = _mm256_extractf128_ps(a, 1);
    // loQuad = ( x3, x2, x1, x0 )
    let lo_quad = _mm256_castps256_ps128(a);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    let sum_quad = _mm_add_ps(lo_quad, hi_quad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    let lo_dual = sum_quad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    let hi_dual = _mm_movehl_ps(sum_quad, sum_quad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    let sum_dual = _mm_add_ps(lo_dual, hi_dual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
       let lo = sum_dual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    let hi = _mm_shuffle_ps(sum_dual, sum_dual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    let sum = _mm_add_ss(lo, hi);
    _mm_cvtss_f32(sum)
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn dot_product_f64x8_x86_64(a: Vec<f64>, b: Vec<f64>) -> Result<f64, &'static str> {

    let mut index: usize = 0;
    let mut dp_mm = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    if b.len() != a.len() {
        let msg = "A and B do not have the same length: A {:?} B {:?}";
        return Err(msg);
    }


    while index + 8 <= a.len() {

        println!("Looop");

        let c = _mm512_set_pd(a[index], a[index+1], a[index+2], a[index+3], a[index+4], a[index+5], a[index+6], a[index+7]);
        let d = _mm512_set_pd(b[index], b[index+1], b[index+2], b[index+3], b[index+4], b[index+5], b[index+6], b[index+7]);

        dp_mm = _mm512_fmadd_pd(c, d, dp_mm);

        index += 8
    }

    let mut dp = reduce_vec_f64x8(dp_mm);

    for i in index..a.len() {
        dp += a[i]*b[i];
    }

    Ok(dp)
}

#[cfg(target_arch = "x86_64")]
unsafe fn reduce_vec_f64x8(a: __m512d) -> f64 {

    println!("Reduction start");
    // hiQuad = ( x7, x6, x5, x4 )
    let hi_quad = _mm512_extractf64x4_pd(a, 1);
    // loQuad = ( x3, x2, x1, x0 )
    let lo_quad = _mm512_castpd512_pd256(a);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    let sum_quad = _mm256_add_pd(lo_quad, hi_quad);

    let hi_quad = _mm256_extractf128_pd(sum_quad, 1);
    let lo_quad = _mm256_castpd256_pd128(sum_quad);
    let sum_quad = _mm_add_pd(lo_quad, hi_quad);
    _mm_cvtsd_f64(sum_quad)

}

#[cfg(target_arch = "x86_64")]
pub unsafe fn dot_product_f64x4_x86_64(a: Vec<f64>, b: Vec<f64>) -> Result<f64, &'static str> {

    let mut index: usize = 0;
    let mut dp_mm = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

    if b.len() != a.len() {
        let msg = "A and B do not have the same length: A {:?} B {:?}";
        return Err(msg);
    }

    while index + 4 <= a.len() {

        let c = _mm256_set_pd(a[index], a[index+1], a[index+2], a[index+3]);
        let d = _mm256_set_pd(b[index], b[index+1], b[index+2], b[index+3]);

        dp_mm = _mm256_fmadd_pd(c, d, dp_mm);

        index += 4
    }

    let mut dp = reduce_vec_f64x4(dp_mm);

    for i in index..a.len() {
        dp += a[i]*b[i];
    }

    Ok(dp)
}

#[cfg(target_arch = "x86_64")]
unsafe fn reduce_vec_f64x4(a: __m256d) -> f64 {

    let hi_quad = _mm256_extractf128_pd(a, 1);

    let lo_quad = _mm256_castpd256_pd128(a);
    let sum_quad = _mm_add_pd(lo_quad, hi_quad);

    let hi = _mm_shuffle_pd(sum_quad, sum_quad, 0x1);
    let res = _mm_add_pd(sum_quad, hi);

     _mm_cvtsd_f64(res)
}

fn dot_product_f64x4_aarch64(a: Vec<f64>, b: Vec<f64>) {


} */
