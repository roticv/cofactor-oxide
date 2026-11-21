//! Off macOS there's no GPU bucket fill: the siever fills its own.

use std::sync::Arc;

use cofox_core::nfs::{BucketFill, BucketPlan};

pub enum Gpu {}

impl Gpu {
    pub fn new(_: &[BucketPlan; 2]) -> Option<Arc<Self>> {
        None
    }

    pub fn name(&self) -> String {
        match *self {}
    }

    pub fn filler(self: &Arc<Self>) -> Box<dyn BucketFill> {
        match **self {}
    }
}
