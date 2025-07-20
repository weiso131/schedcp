// Exact ML implementation from the scx_rusty fork
extern crate tensorflow;

use tensorflow::Graph;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;
use std::error::Error;

struct TensorFlowModel {
    graph: Graph,
    session: Session,
}

impl TensorFlowModel {
    fn new(model_dir: &str) -> Result<Self, Box<dyn Error>> {
        let mut graph = Graph::new();
        let bundle = tensorflow::SavedModelBundle::load(
            &SessionOptions::new(),
            &["serve"],
            &mut graph,
            model_dir,
        )?;
        let session = bundle.session;
        
        Ok(TensorFlowModel { graph, session })
    }

    fn predict(&self, input_data: Vec<f64>) -> Result<bool, Box<dyn Error>> {
        let input_tensor = Tensor::new(&[input_data.len() as u64]).with_values(&input_data)?;

        let input_op = self.graph.operation_by_name_required("serving_default_input")?;
        let output_op = self.graph.operation_by_name_required("StatefulPartitionedCall")?;

        let mut args = SessionRunArgs::new();
        args.add_feed(&input_op, 0, &input_tensor);
        let output_token = args.request_fetch(&output_op, 0);

        self.session.run(&mut args)?;

        let output_tensor: Tensor<f64> = args.fetch(output_token).unwrap();
        let output_value = output_tensor[0];
        Ok(output_value == 1.0)
    }
}

pub struct MLScheduler {
    inference_model: TensorFlowModel,
}

impl MLScheduler {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            inference_model: TensorFlowModel::new(model_path)?,
        })
    }

    pub fn migrate_inference(&self, cpu: &i32, cpu_idle: &i32, cpu_not_idle: &i32, src_dom_load: &f64, dst_dom_load: &f64) -> bool {
        let input_vec = vec![f64::from(*cpu), f64::from(*cpu_idle), f64::from(*cpu_not_idle), *src_dom_load, *dst_dom_load];
        self.inference_model.predict(input_vec).unwrap()
    }
}