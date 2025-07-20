// Copyright (c) Meta Platforms, Inc. and affiliates.
// Machine Learning based support load balancer
// Inspired by the paper: https://arxiv.org/abs/2407.10077

extern crate tensorflow;

use anyhow::Result;
use tensorflow::Graph;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

pub struct TensorFlowModel {
    graph: Graph,
    session: Session,
}

impl TensorFlowModel {
    pub fn new(model_dir: &str) -> Result<Self> {
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

/// Task context structure matching BPF interface
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TaskCtx {
    pub pid: u32,
    pub dom_id: u32,
    pub cpu: i32,
    pub cpu_idle: i32,
    pub cpu_not_idle: i32,
    pub src_dom_load: u64,
    pub dst_dom_load: u64,
}

pub struct MLLoadBalancer {
    inference_model: TensorFlowModel,
}

impl MLLoadBalancer {
    pub fn new(model_path: &str) -> Result<Self> {
        Ok(Self {
            inference_model: TensorFlowModel::new(model_path)?,
        })
    }

    /// Exact implementation of migrate_inference from the fork
    pub fn migrate_inference(&self, cpu: &i32, cpu_idle: &i32, cpu_not_idle: &i32, src_dom_load: &f64, dst_dom_load: &f64) -> bool {
        let input_vec = vec![f64::from(*cpu), f64::from(*cpu_idle), f64::from(*cpu_not_idle), *src_dom_load, *dst_dom_load];
        self.inference_model.predict(input_vec).unwrap_or(false)
    }

    /// Make migration decision based on task context
    pub fn should_migrate_task(&self, task: &TaskCtx) -> bool {
        let src_load = task.src_dom_load as f64;
        let dst_load = task.dst_dom_load as f64;
        
        self.migrate_inference(
            &task.cpu,
            &task.cpu_idle,
            &task.cpu_not_idle,
            &src_load,
            &dst_load,
        )
    }
}