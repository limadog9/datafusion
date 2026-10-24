// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Expose aggregate inputs to the projection-based subquery rewrites.

use std::mem;
use std::sync::Arc;

use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{Column, Result};
use datafusion_expr::expr_rewriter::NamePreserver;
use datafusion_expr::{Aggregate, Expr, LogicalPlan, Projection};

use crate::OptimizerConfig;

/// Move row-level aggregate inputs containing supported subqueries into a
/// projection. The calling rule subsequently decorrelates that projection
/// during its top-down traversal, without adding another optimizer pass.
pub(crate) fn extract_aggregate_subqueries(
    mut aggregate: Aggregate,
    config: &dyn OptimizerConfig,
    contains_subquery: impl Fn(&Expr) -> bool,
) -> Result<Transformed<LogicalPlan>> {
    // Grouping-set containers cannot be evaluated as projection expressions.
    if !aggregate
        .group_expr
        .iter()
        .filter(|expr| !matches!(expr, Expr::GroupingSet(_)))
        .any(&contains_subquery)
        && !aggregate
            .aggr_expr
            .iter()
            .any(|expr| aggregate_inputs_contain_subquery(expr, &contains_subquery))
    {
        return Ok(Transformed::no(LogicalPlan::Aggregate(aggregate)));
    }

    let input_schema = aggregate.input.schema();
    let mut projection: Vec<_> = input_schema
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    let input_len = projection.len();
    let name_preserver = NamePreserver::new_for_projection();
    {
        let mut extract = |expr: Expr| {
            if matches!(expr, Expr::GroupingSet(_)) || !contains_subquery(&expr) {
                return Ok(Transformed::no(expr));
            }
            let alias = loop {
                let alias = config.alias_generator().next("__aggregate_subquery");
                if !input_schema.has_column_with_unqualified_name(&alias) {
                    break alias;
                }
            };
            projection.push(expr.alias(alias.clone()));
            Ok(Transformed::yes(Expr::Column(Column::from_name(alias))))
        };

        // Repeated grouping keys must share their projected column, otherwise
        // they become distinct expressions with the same output field name.
        let mut extracted_groups: Vec<(Expr, Expr)> = vec![];
        aggregate.group_expr = mem::take(&mut aggregate.group_expr)
            .into_iter()
            .map(|expr| {
                if matches!(expr, Expr::GroupingSet(_)) || !contains_subquery(&expr) {
                    return Ok(expr);
                }
                if let Some((_, rewritten)) = extracted_groups
                    .iter()
                    .find(|(original, _)| original == &expr)
                {
                    return Ok(rewritten.clone());
                }
                let original = expr.clone();
                let name = name_preserver.save(&expr);
                let rewritten = name.restore(extract(expr)?.data);
                extracted_groups.push((original, rewritten.clone()));
                Ok(rewritten)
            })
            .collect::<Result<_>>()?;

        aggregate.aggr_expr = mem::take(&mut aggregate.aggr_expr)
            .into_iter()
            .map(|expr| {
                if !aggregate_inputs_contain_subquery(&expr, &contains_subquery) {
                    return Ok(expr);
                }
                let name = name_preserver.save(&expr);
                // Only arguments, FILTER, and ORDER BY are evaluated per input
                // row. The aggregate function itself stays above the projection.
                expr.transform_up(|expr| {
                    if matches!(expr, Expr::AggregateFunction(_)) {
                        expr.map_children(&mut extract)
                    } else {
                        Ok(Transformed::no(expr))
                    }
                })
                .data()
                .map(|expr| name.restore(expr))
            })
            .collect::<Result<_>>()?;
    }

    if projection.len() == input_len {
        return Ok(Transformed::no(LogicalPlan::Aggregate(aggregate)));
    }

    let input =
        LogicalPlan::Projection(Projection::try_new(projection, aggregate.input)?);
    Ok(Transformed::yes(LogicalPlan::Aggregate(
        Aggregate::try_new(Arc::new(input), aggregate.group_expr, aggregate.aggr_expr)?,
    )))
}

/// Check row inputs without traversing the aggregate-function wrapper.
fn aggregate_inputs_contain_subquery(
    expr: &Expr,
    contains_subquery: &impl Fn(&Expr) -> bool,
) -> bool {
    match expr {
        Expr::Alias(alias) => {
            aggregate_inputs_contain_subquery(&alias.expr, contains_subquery)
        }
        Expr::AggregateFunction(function) => {
            function.params.args.iter().any(contains_subquery)
                || function
                    .params
                    .filter
                    .as_deref()
                    .is_some_and(contains_subquery)
                || function
                    .params
                    .order_by
                    .iter()
                    .any(|sort| contains_subquery(&sort.expr))
        }
        _ => contains_subquery(expr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptimizerContext;
    use crate::test::test_table_scan;
    use datafusion_expr::{ExprFunctionExt, LogicalPlanBuilder, col, exists};
    use datafusion_functions_aggregate::expr_fn::sum;

    fn contains_predicate_subquery(expr: &Expr) -> bool {
        expr.exists(|expr| Ok(matches!(expr, Expr::Exists(_) | Expr::InSubquery(_))))
            .unwrap()
    }

    #[test]
    fn no_subqueries_preserve_plan_and_schema() -> Result<()> {
        let LogicalPlan::Aggregate(aggregate) =
            LogicalPlanBuilder::from(test_table_scan()?)
                .aggregate(vec![col("a")], vec![sum(col("b")).alias("total")])?
                .build()?
        else {
            unreachable!()
        };
        let original = aggregate.clone();
        let schema = Arc::clone(&aggregate.schema);
        let result = extract_aggregate_subqueries(
            aggregate,
            &OptimizerContext::new(),
            contains_predicate_subquery,
        )?;
        assert!(!result.transformed);
        assert_eq!(result.data, LogicalPlan::Aggregate(original));
        assert!(Arc::ptr_eq(result.data.schema(), &schema));
        Ok(())
    }

    #[test]
    fn input_alias_collision_preserves_output_schema() -> Result<()> {
        let input = LogicalPlanBuilder::from(test_table_scan()?)
            .project(vec![col("a").alias("__aggregate_subquery_1"), col("b")])?
            .build()?;
        let predicate = exists(Arc::new(test_table_scan()?));
        let aggregate = Aggregate::try_new(
            Arc::new(input),
            vec![col("test.b")],
            vec![
                sum(col("__aggregate_subquery_1"))
                    .filter(predicate)
                    .build()?,
            ],
        )?;
        let schema = Arc::clone(&aggregate.schema);
        let result = extract_aggregate_subqueries(
            aggregate,
            &OptimizerContext::new(),
            contains_predicate_subquery,
        )?;
        assert!(result.transformed);
        assert_eq!(result.data.schema(), &schema);
        let LogicalPlan::Aggregate(aggregate) = result.data else {
            unreachable!()
        };
        let LogicalPlan::Projection(projection) = aggregate.input.as_ref() else {
            unreachable!()
        };
        assert_eq!(
            projection.schema.field_names(),
            vec!["__aggregate_subquery_1", "test.b", "__aggregate_subquery_2"]
        );
        Ok(())
    }
}
