use maths::{Differentiable, ReLU, Linear};
use ndarray::prelude::{*, s, NewAxis};
use rand::{random, prelude::{thread_rng}};

use crate::maths::Mse;

mod maths;

struct Network<A: Differentiable + Clone, C: Differentiable> {
    layers: Vec<Layer<A>>,
    cost: C,
}

impl<A: Differentiable + Clone, C: Differentiable> Network<A, C> {
    fn from_layout(shape: Vec<usize>, cost: C, activation: A) -> Self {
        Self {
            layers: shape[..shape.len() - 1]
                .iter()
                .zip(shape[1..].iter())
                .map(|(a, b)| {
                    Layer { 
                        weights: Array2::from_shape_vec(
                            (*b, *a),
                            vec![0.0; a * b]
                                .iter()
                                .map(|_| (rand::random::<f64>() - 0.5) * 2.0)
                                .collect::<Vec<f64>>()
                        ).unwrap(), 
                        biases: Array1::from_vec(vec![0.0; *b]), 
                        activation: activation.clone(),
                    }
                })
                .collect(),
            cost,
        }
    }

    fn train(
        &mut self, 
        training_data: (Array2<f64>, Array2<f64>), 
        learning_rate: f64,
        epochs: usize,
    ) {
        
        for _i in 1..=epochs {
            
            for (i, o) in training_data.0.columns().into_iter()
                .zip(training_data.1.columns().into_iter()) {
                    let acts = self.activations(&i.to_owned());
                    let prev = Mse(o.to_owned()).diff(acts.last().unwrap());

                    self.layers.iter_mut().enumerate().rev().fold(prev, |a, (idx, layer)| {
                        // println!("i");   
                        layer.descend(&acts[idx], &a, learning_rate)
                    });
            }
        }

        let binding = self.activations(&array![3.0, 2.0]);
        let res = binding.last().unwrap();
        println!("{:?}", res);
        
    }

    fn activations(&mut self, input: &Array1<f64>) -> Vec<Array1<f64>> {
        let mut acts = vec![input.clone()];
        self.layers.iter().fold(input.clone(), |a, x| {
            let n = x.apply(&a);
            acts.append(&mut vec![n.clone()]);
            n
        });
        acts
    }
}

#[derive(Debug, Clone)]
struct Layer<A: Differentiable + Clone> {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub activation: A,
}

impl<A: Differentiable + Clone> Layer<A> {
    pub fn apply(&self, inputs: &Array1<f64>) -> Array1<f64> {
        &self.weights.dot(inputs) + &self.biases
    }
    
    pub fn differentiate(&self, inputs: &Array1<f64>, doutput: &Array1<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let z = &self.weights.dot(inputs) + &self.biases;
        let dz = self.activation.diff(&z);
        
        // println!("dz: {:?}\ndoutput: {:?}", dz, doutput);
        let rhs = &dz * doutput; 
        
        let dw = &inputs.slice(s![NewAxis, ..]) * &rhs.slice(s![.., NewAxis]);
        // let dw = &inputs.slice(s![NewAxis, ..]).dot(&rhs.slice(s![.., NewAxis]));
        // println!("{:?}", self.weights.sum_axis(Axis(0)));
        let da = rhs.dot(&self.weights);
        let db = doutput.clone();

        (dw, db, da)
    }

    pub fn descend(&mut self, inputs: &Array1<f64>, doutput: &Array1<f64>, learning_rate: f64) -> Array1<f64> {
        let (dw, db, da) = self.differentiate(inputs, doutput);
        // dw.swap_axes(0, 1);
        self.weights -= &(dw * learning_rate);
        // println!("applied");
        self.biases -= &(db * learning_rate);

        da
    }
}



fn main() {
    let layer1 = Layer {
            weights: arr2(&[[ 0.5, 0.5 ],
                            [ 0.5, 0.5 ] ]),
            biases: arr1(&[0.0, 0.0]),
            activation: ReLU,
    };

    let layer2 = Layer {
        weights: arr2(&[[ 0.5,   0.5 ],
                        [ 0.5,   0.5 ] ]),
        biases: arr1(&[0.0, 0.0]),
        activation: Linear,
    };


    // println!("{:?}", layer1.weights);
    // println!("{:?}", layer1.biases);
    // println!("{:?}", layer2.weights);
    // println!("{:?}", layer2.biases);

    // let mut model = Network {
    //     layers: vec![ layer1, 
    //     layer2 
    //     ],
    //     cost: Linear, // TODO
    // };

    let mut model = Network::from_layout(vec![2; 2], Mse(array![]), Linear);

    // println!("{:?}", model.layers[0].weights);
    // println!("{:?}", model.layers[0].biases);
    // println!("{:?}", model.layers[1].weights);
    // println!("{:?}", model.layers[1].biases);

    
    // let inputs = Array2::from_shape_vec((2, 50), (0..100).map(
    //     |_| (random::<f64>() - 0.5) * 20.0,
    // ).collect::<Vec<f64>>()).unwrap();

    // let mut outputs = arr2(
    //     &inputs.,
    // );

    model.train((
            arr2(&[ [3.0, 1.0, 8.0],
                    [2.0, 3.0, 1.0], ]),
            arr2(&[ [5.0, 4.0, 9.0],
                    [1.0, -2.0, 7.0] ])
        ),
        0.002,
        500_000,
    );

    // model.train((
    //         inputs,
    //         outputs
    //     ),
    //     0.002,
    //     50_000,
    // );
}
