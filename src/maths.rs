use ndarray::{Array1};

#[derive(Debug, Clone)]
pub struct ReLU;

impl Differentiable for ReLU {
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        x.map(|i| i.max(0.0))
    }
    
    fn diff(&self, x: &Array1<f64>) -> Array1<f64> {
        x.map(|&i| if i > 0.0 { 1.0 } else { 0.0 })
    }
}

#[derive(Debug, Clone)]
pub struct Mse(pub Array1<f64>); //TODO

impl Differentiable for Mse {
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        (0.5 * (x - &self.0)).map(|x| x.powi(2))
    }
    
    fn diff(&self, x: &Array1<f64>) -> Array1<f64> {
        x - &self.0
    }
}

#[derive(Debug, Clone)]
pub struct Linear;

impl Differentiable for Linear {
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        x.clone()
    }

    fn diff(&self, x: &Array1<f64>) -> Array1<f64> {
        Array1::from_vec(vec![1.0; x.shape()[0]])
    }
}

pub trait Differentiable {
    fn apply(&self, x: &Array1<f64>) -> Array1<f64>;

    fn diff(&self, x: &Array1<f64>) -> Array1<f64>;
}