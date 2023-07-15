"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from data_prep.nodes import get_data, preprocess_companies, preprocess_shuttles, create_model_input_table

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(get_data, None, ["companies", "shuttles", "reviews"], name="get_data_node"),
            node(preprocess_companies, "companies", "companies", name="companies_node"),
            node(preprocess_shuttles, "shuttles", "shuttles", name="shuttles_node"),
            node(create_model_input_table, ["shuttles", "companies", "reviews"], "model_input_table", name="create_model_input_table_node")
        ]
        )