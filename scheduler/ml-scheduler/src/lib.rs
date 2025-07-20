pub mod bpf_interface;
pub mod scheduler_integration;

use anyhow::Result;
use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor, SavedModelBundle};

pub struct TensorFlowModel {
    graph: Graph,
    session: Session,
}

impl TensorFlowModel {
    pub fn new(model_dir: &str) -> Result<Self> {
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(),
            &["serve"],
            &mut graph,
            model_dir,
        )?;
        let session = bundle.session;
        
        Ok(TensorFlowModel { graph, session })
    }

    pub fn predict(&self, input_data: Vec<f64>) -> Result<bool> {
        let input_tensor = Tensor::new(&[input_data.len() as u64]).with_values(&input_data)?;

        let input_op = self.graph.operation_by_name_required("serving_default_input")?;
        let output_op = self.graph.operation_by_name_required("StatefulPartitionedCall")?;

        let mut args = SessionRunArgs::new();
        args.add_feed(&input_op, 0, &input_tensor);
        let output_token = args.request_fetch(&output_op, 0);

        self.session.run(&mut args)?;

        let output_tensor: Tensor<f64> = args.fetch(output_token)?;
        let output_value = output_tensor[0];
        Ok(output_value == 1.0)
    }
}

#[derive(Debug)]
pub struct MigrationFeatures {
    pub cpu: i32,
    pub cpu_idle: i32,
    pub cpu_not_idle: i32,
    pub src_dom_load: f64,
    pub dst_dom_load: f64,
}

pub struct MLScheduler {
    model: TensorFlowModel,
}

impl MLScheduler {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = TensorFlowModel::new(model_path)?;
        Ok(MLScheduler { model })
    }

    pub fn should_migrate(&self, features: &MigrationFeatures) -> Result<bool> {
        self.migrate_inference(
            &features.cpu,
            &features.cpu_idle,
            &features.cpu_not_idle,
            &features.src_dom_load,
            &features.dst_dom_load,
        )
    }
    
    fn migrate_inference(&self, cpu: &i32, cpu_idle: &i32, cpu_not_idle: &i32, src_dom_load: &f64, dst_dom_load: &f64) -> Result<bool> {
        let input_vec = vec![
            *cpu as f64,
            *cpu_idle as f64,
            *cpu_not_idle as f64,
            *src_dom_load,
            *dst_dom_load,
        ];
        
        self.model.predict(input_vec)
    }
}