"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table,
    get_data,
    preprocess_companies,
    preprocess_shuttles,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_data,
                inputs=None,
                outputs=["companies", "shuttles", "reviews"],
                name="obtener_datos",
            ),
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs="prep_companies",
                name="preprocesar_companies",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="prep_shuttles",
                name="preprocesar_shuttles",
            ),
            node(
                func=create_model_input_table,
                inputs=["prep_shuttles", "prep_companies", "reviews"],
                outputs="model_input_table",
                name="crear_model_input_table",
            ),
        ]
    )
